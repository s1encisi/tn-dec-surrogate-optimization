from __future__ import annotations

from datetime import date, datetime
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np
import pandas as pd

from .config import project_path


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_workbook_sheet(
    workbook: str | Path,
    sheet_name: str,
    header_row_zero_based: int,
) -> pd.DataFrame:
    workbook_path = project_path(workbook)
    if not workbook_path.exists():
        raise FileNotFoundError(f"Workbook not found: {workbook_path}")
    frame = pd.read_excel(
        workbook_path,
        sheet_name=sheet_name,
        header=header_row_zero_based,
        engine="openpyxl",
    )
    frame.columns = [str(column).strip() for column in frame.columns]
    if not pd.Index(frame.columns).is_unique:
        duplicates = pd.Index(frame.columns)[pd.Index(frame.columns).duplicated()].tolist()
        raise ValueError(f"Duplicate columns after stripping whitespace: {duplicates}")
    return frame


def coerce_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype("boolean")
    mapping = {
        "1": True,
        "0": False,
        "true": True,
        "false": False,
        "yes": True,
        "no": False,
        "是": True,
        "否": False,
    }
    converted = series.map(
        lambda value: pd.NA if pd.isna(value) else mapping.get(str(value).strip().lower(), value)
    )
    invalid = converted.notna() & ~converted.isin([True, False])
    if bool(invalid.any()):
        values = sorted({str(value) for value in series.loc[invalid].unique()})
        raise ValueError(f"Invalid boolean values in {series.name}: {values}")
    return converted.astype("boolean")


def normalize_processed_frame(frame: pd.DataFrame, date_column: str = "Date") -> pd.DataFrame:
    result = frame.copy()
    result[date_column] = pd.to_datetime(result[date_column], errors="raise").dt.normalize()
    bool_columns = [
        "has_preprocessing_record",
        "normal_operation_modeling",
        "TP_out_label_observed",
        "TN_out_label_observed",
        "DEC_label_observed",
    ]
    for column in bool_columns:
        if column in result.columns:
            result[column] = coerce_bool(result[column])
    return result


def _json_default(value: Any) -> Any:
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if pd.isna(value):
        return None
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _clean_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _clean_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean_json(item) for item in value]
    if isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
        return None
    if isinstance(value, np.ndarray):
        return [_clean_json(item) for item in value.tolist()]
    return value


def write_json(payload: Any, path: str | Path) -> None:
    destination = project_path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(
        _clean_json(payload),
        ensure_ascii=False,
        indent=2,
        default=_json_default,
        allow_nan=False,
    )
    descriptor, temp_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, destination)
    except Exception:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise
