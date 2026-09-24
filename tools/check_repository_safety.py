"""Block unreviewed paths and sensitive content in staged files or pushed history."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import PurePosixPath

MAX_BYTES = 1024 * 1024
SPECIAL_FILES = {
    ".gitignore",
    ".gitattributes",
    ".githooks/pre-commit",
    ".githooks/pre-push",
    ".github/workflows/ci.yml",
    "LICENSE",
}
SOURCE_SUFFIXES = {".py", ".toml", ".md"}
PRIVATE_DIRECTORIES = {
    "data",
    "datasets",
    "initialdata",
    "results",
    "provenance",
    "submission",
    "delivery",
    "drafts",
    "figures",
    "plotting_data",
    "plotting_data_payload_v3",
    "__pycache__",
    ".build_work",
    ".venv-rl-gpu",
    ".venv",
    "legacy",
    "reviews",
    "upgrades",
    "evidence",
    "qa",
    "demo_outputs",
}
CONTENT_RULES = {
    "private key": rb"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----",
    "GitHub credential": rb"(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})",
    "API credential": rb"(?:sk-(?:proj-)?[A-Za-z0-9_-]{24,}|AKIA[A-Z0-9]{16})",
    "literal secret": (
        rb"""(?i)(?:api[_-]?key|access[_-]?token|password|secret)"""
        rb"""\s*[:=]\s*["'][A-Za-z0-9_+/=-]{20,}["']"""
    ),
    "credential in URL": rb"https?://[^\s/]+:[^\s/]+@",
    "Git LFS pointer requiring separate review": rb"(?m)^version https://git-lfs.github.com/spec/v1$",
    "personal email": rb"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
    "personal filesystem path": rb"(?i)(?:[A-Z]:[\\/]+Users[\\/]+|/Users/|/home/)[A-Za-z0-9_.-]+",
}


def git(*args: str, data: bytes | None = None) -> bytes:
    result = subprocess.run(["git", *args], input=data, capture_output=True)
    if result.returncode:
        raise RuntimeError("Git operation failed: " + " ".join(args[:2]))
    return result.stdout


def allowlist(body: bytes) -> set[str]:
    paths = set()
    for line in body.decode("utf-8-sig").splitlines():
        if not line.startswith("!/") or line.endswith("/"):
            continue
        path = line[2:]
        if any(character in path for character in "*?[]\\") or ".." in PurePosixPath(path).parts:
            raise ValueError("Only exact, repository-relative allowlist entries are permitted.")
        paths.add(path)
    if not paths:
        raise ValueError("The repository has no explicit file allowlist.")
    return paths


def path_problem(path: str, approved: set[str]) -> str | None:
    if path not in approved:
        return "not in the reviewed file allowlist"
    item = PurePosixPath(path)
    if any(part.casefold() in PRIVATE_DIRECTORIES for part in item.parts):
        return "private data or artifact directory"
    if path not in SPECIAL_FILES and item.suffix.casefold() not in SOURCE_SUFFIXES:
        return "data, binary or unapproved file type"
    return None


def content_problems(body: bytes) -> list[str]:
    if b"\x00" in body:
        return ["binary content"]
    try:
        body.decode("utf-8-sig")
    except UnicodeDecodeError:
        return ["non-UTF-8 content"]
    return [label for label, pattern in CONTENT_RULES.items() if re.search(pattern, body)]


def entries(revision: str | None) -> list[tuple[str, str, str]]:
    result = []
    raw = git("ls-tree", "-r", "-z", revision) if revision else git("ls-files", "--stage", "-z")
    for record in raw.split(b"\x00"):
        if not record:
            continue
        metadata, path = record.split(b"\t", 1)
        mode, middle, last = metadata.decode("ascii").split()
        if revision:
            object_id = last
            if middle != "blob":
                raise ValueError("Submodules and other non-file objects are not allowed.")
        else:
            object_id = middle
            if last != "0":
                raise ValueError("Resolve staged conflicts before committing.")
        result.append((mode, object_id, path.decode("utf-8")))
    return result


def read_blobs(object_ids: set[str]) -> dict[str, bytes | None]:
    if not object_ids:
        return {}
    request = ("\n".join(sorted(object_ids)) + "\n").encode("ascii")
    sizes = git("cat-file", "--batch-check", data=request).decode("ascii").splitlines()
    result = {}
    small = []
    for record in sizes:
        object_id, kind, size = record.split()
        if kind != "blob":
            raise ValueError("Expected a file blob.")
        if int(size) > MAX_BYTES:
            result[object_id] = None
        else:
            small.append(object_id)
    if small:
        raw = git("cat-file", "--batch", data=("\n".join(small) + "\n").encode("ascii"))
        offset = 0
        for expected in small:
            boundary = raw.index(b"\n", offset)
            object_id, kind, size = raw[offset:boundary].decode("ascii").split()
            if object_id != expected or kind != "blob":
                raise ValueError("Unexpected Git object response.")
            start = boundary + 1
            end = start + int(size)
            result[object_id] = raw[start:end]
            offset = end + 1
    return result


def check(revision: str | None, cache: dict[str, bytes | None]) -> tuple[int, list[str]]:
    rows = entries(revision)
    cache.update(read_blobs({object_id for _, object_id, _ in rows} - cache.keys()))
    policy = next((cache[oid] for _, oid, path in rows if path == ".gitignore"), None)
    if policy is None:
        raise ValueError("Every commit must include the file allowlist.")
    approved = allowlist(policy)
    failures = []
    for mode, object_id, path in rows:
        reason = path_problem(path, approved)
        if mode not in {"100644", "100755"}:
            reason = "symlinks and non-regular files are not allowed"
        body = cache[object_id]
        if body is None:
            reason = "file exceeds the 1 MiB source limit"
        if reason:
            failures.append(f"{path}: {reason}")
        elif body is not None:
            failures.extend(f"{path}: {problem}" for problem in content_problems(body))
    return len(rows), failures


def self_test() -> None:
    safe = "Paper/SourceCode/src/taici/metrics.py"
    approved = {safe, "measurements.csv", "Paper/results/summary.py"}
    assert path_problem(safe, approved) is None
    assert path_problem("LICENSE", {"LICENSE"}) is None
    assert path_problem(".github/workflows/ci.yml", {".github/workflows/ci.yml"}) is None
    assert path_problem("unreviewed.yml", {"unreviewed.yml"})
    assert path_problem("new.py", approved)
    assert path_problem("measurements.csv", approved)
    assert path_problem("Paper/results/summary.py", approved)
    assert content_problems(b"gh" + b"p_" + b"A" * 36)
    assert content_problems(b"person" + b"@" + b"example.com")
    assert content_problems(b"-----BEGIN " + b"PRIVATE KEY-----")
    assert content_problems(b"data\x00payload")
    assert content_problems(b"C:" + b"/Users/" + b"someone/private")
    assert not content_problems(b"def sum_values(values):\n    return sum(values)\n")
    assert allowlist(b"*\n!/Paper/\n!/" + safe.encode() + b"\n") == {safe}
    print("Repository safety self-test: PASS")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--staged", action="store_true")
    group.add_argument("--history", action="store_true")
    group.add_argument("--pre-push", action="store_true")
    group.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    revisions: list[str | None] = [None]
    if args.history or args.pre_push:
        tips = ["HEAD"]
        if args.pre_push:
            tips = []
            for line in sys.stdin:
                local_ref, local_sha, remote_ref, remote_sha = line.split()
                if set(local_sha) != {"0"}:
                    tips.append(local_sha)
            if not tips:
                return 0
        revisions = sorted(set(git("rev-list", *tips).decode("ascii").splitlines()))
    cache: dict[str, bytes | None] = {}
    failures = []
    checked = 0
    for revision in revisions:
        count, found = check(revision, cache)
        checked += count
        label = revision[:12] if revision else "index"
        failures.extend(f"{label}: {item}" for item in found)
    if failures:
        print("Repository safety check blocked the operation:", file=sys.stderr)
        for failure in failures:
            print("  " + failure, file=sys.stderr)
        return 1
    print(f"Repository safety check: PASS ({len(revisions)} snapshot(s), {checked} file entries)")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, ValueError, UnicodeError) as exc:
        print("Repository safety check failed: " + str(exc), file=sys.stderr)
        raise SystemExit(1) from None
