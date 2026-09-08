from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .denoise import continuous_valid_segments


ALIGNMENT_SOURCE_SHIFT = {"L0": 1, "L1": 2, "L2": 3}


def target_feature_spec(feature_config: dict[str, Any], target: str) -> dict[str, Any]:
    try:
        return feature_config["targets"][target]
    except KeyError as exc:
        raise KeyError(f"No feature registry for target {target}") from exc


def build_tabular_next_day_features(
    frame: pd.DataFrame,
    feature_config: dict[str, Any],
    target: str,
    version: str,
    alignment: str,
) -> tuple[pd.DataFrame, list[str]]:
    if version not in {"F_A", "F_B", "F_C"}:
        raise ValueError(f"Unknown feature version: {version}")
    if alignment not in ALIGNMENT_SOURCE_SHIFT:
        raise ValueError(f"Unknown alignment: {alignment}")
    data = frame.sort_values("Date").reset_index(drop=True).copy()
    spec = target_feature_spec(feature_config, target)
    source_shift = ALIGNMENT_SOURCE_SHIFT[alignment]
    dates = pd.to_datetime(data["Date"])

    result = pd.DataFrame(
        {
            "Date": dates,
            "actual": pd.to_numeric(data[target], errors="coerce"),
            "study_partition": data["study_partition"],
        }
    )
    day_of_year = dates.dt.dayofyear.astype(float)
    result["doy_sin"] = np.sin(2.0 * np.pi * day_of_year / 365.2425)
    result["doy_cos"] = np.cos(2.0 * np.pi * day_of_year / 365.2425)
    feature_names = ["doy_sin", "doy_cos"]

    exogenous = [*spec["g2_influent"], *spec["g3_process"]]
    for column in exogenous:
        feature_name = f"source_{column}"
        result[feature_name] = pd.to_numeric(data[column], errors="coerce").shift(source_shift)
        feature_names.append(feature_name)

    target_lags = [int(value) for value in spec["g4_target_lags"]]
    if version in {"F_B", "F_C"}:
        for lag in target_lags:
            feature_name = f"lag{lag}_{target}"
            result[feature_name] = pd.to_numeric(data[target], errors="coerce").shift(lag)
            feature_names.append(feature_name)

    if version == "F_C":
        load_feature = str(spec["g5_load_sensitivity"][0])
        _, concentration = load_feature.split("_x_", maxsplit=1)
        q_name = "source_Q"
        concentration_name = f"source_{concentration}"
        if q_name not in result or concentration_name not in result:
            raise ValueError(f"Cannot construct {load_feature} for {target}")
        result[load_feature] = result[q_name] * result[concentration_name]
        feature_names.append(load_feature)

    valid_target = data["is_normal_operation"].fillna(False).astype(bool) & data[target].notna()
    segment = continuous_valid_segments(dates, data["is_normal_operation"], data[target].notna())
    source_normal = data["is_normal_operation"].astype(bool).shift(
        source_shift, fill_value=False
    )
    eligible = valid_target & source_normal
    if version in {"F_B", "F_C"}:
        max_lag = max(target_lags)
        eligible &= segment.ge(0) & segment.eq(segment.shift(max_lag))
    result["eligible"] = eligible

    expected_dimension_key = f"expected_{version}_dimension"
    if version in {"F_A", "F_B"} and expected_dimension_key in spec:
        expected = int(spec[expected_dimension_key])
        if len(feature_names) != expected:
            raise ValueError(
                f"Feature dimension mismatch for {target}/{version}: "
                f"expected {expected}, observed {len(feature_names)}"
            )
    maximum = int(feature_config["constraints"]["maximum_table_features_before_missing_indicators"])
    if len(feature_names) > maximum:
        raise ValueError(f"Feature count {len(feature_names)} exceeds pre-registered maximum {maximum}")
    return result, feature_names


def build_feature_registry_table(
    feature_config: dict[str, Any],
    target: str,
    version: str,
    alignment: str,
) -> pd.DataFrame:
    spec = target_feature_spec(feature_config, target)
    source_shift = ALIGNMENT_SOURCE_SHIFT[alignment]
    records: list[dict[str, Any]] = [
        {
            "feature": "doy_sin",
            "source_variable": "Date",
            "formula": "sin(2*pi*day_of_year/365.2425)",
            "group": "G1_calendar",
            "engineering_rationale": "annual seasonality of temperature and biological activity",
            "availability": "known for target date",
            "lag_days": 0,
            "tasks": version,
            "leakage_risk": "low",
        },
        {
            "feature": "doy_cos",
            "source_variable": "Date",
            "formula": "cos(2*pi*day_of_year/365.2425)",
            "group": "G1_calendar",
            "engineering_rationale": "annual seasonality of temperature and biological activity",
            "availability": "known for target date",
            "lag_days": 0,
            "tasks": version,
            "leakage_risk": "low",
        },
    ]
    for column in spec["g2_influent"]:
        records.append(
            {
                "feature": f"source_{column}",
                "source_variable": column,
                "formula": f"{column}[target_date-{source_shift}d]",
                "group": "G2_influent",
                "engineering_rationale": "pre-registered hydraulic or influent disturbance",
                "availability": "retrospective daily value; within-day cutoff pending confirmation",
                "lag_days": source_shift,
                "tasks": version,
                "leakage_risk": "timestamp semantics pending",
            }
        )
    for column in spec["g3_process"]:
        records.append(
            {
                "feature": f"source_{column}",
                "source_variable": column,
                "formula": f"{column}[target_date-{source_shift}d]",
                "group": "G3_process",
                "engineering_rationale": "pre-registered process state or operation proxy",
                "availability": "retrospective daily value; action/state semantics pending",
                "lag_days": source_shift,
                "tasks": version,
                "leakage_risk": "not a confirmed controllable action",
            }
        )
    if version in {"F_B", "F_C"}:
        for lag in spec["g4_target_lags"]:
            records.append(
                {
                    "feature": f"lag{lag}_{target}",
                    "source_variable": target,
                    "formula": f"{target}[target_date-{lag}d]",
                    "group": "G4_target_history",
                    "engineering_rationale": "daily persistence and weekly operating cycle",
                    "availability": "rolling one-step use of previously observed target",
                    "lag_days": int(lag),
                    "tasks": version,
                    "leakage_risk": "requires prior laboratory result by cutoff",
                }
            )
    return pd.DataFrame.from_records(records)
