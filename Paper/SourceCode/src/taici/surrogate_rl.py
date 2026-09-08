"""Historical-support constrained surrogate RL for TN_out and DEC sensitivity.

This module does not alter the blocked plant-deployment gate. PPA and DO are
virtual continuous controls inside a retrospective proxy environment; MLSS is
replayed as an exogenous slow state. All reported outcomes are proxy estimates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from pymoo.indicators.hv import HV
from pymoo.indicators.igd_plus import IGDPlus
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from .final_proxy import FinalProxyBundle
from .paired_random_windows import FEATURE_KEY, PairedFeatureSet


CONTROL_FEATURES = ("PPA", "DO")
SLOW_STATE_FEATURE = "MLSS"
DEFAULT_PREFERENCES = (0.25, 0.50, 0.75)


class SurrogateRLError(ValueError):
    """Raised when the exploratory surrogate-RL contract is violated."""


@dataclass(frozen=True)
class ScenarioData:
    frame: pd.DataFrame
    union_feature_names: tuple[str, ...]
    observation_feature_names: tuple[str, ...]
    observation_mean: np.ndarray
    observation_scale: np.ndarray
    action_low: np.ndarray
    action_high: np.ndarray
    action_rate: np.ndarray
    objective_low: np.ndarray
    objective_high: np.ndarray
    support_scaler: StandardScaler
    support_model: NearestNeighbors
    support_threshold: float
    preferences: tuple[float, ...]
    train_starts: tuple[int, ...]
    validation_starts: tuple[int, ...]
    test_starts: tuple[int, ...]
    episode_horizon: int
    metadata: Mapping[str, Any]
    support_mode: str = "joint_legacy"
    context_feature_names: tuple[str, ...] = ()
    action_context_feature_names: tuple[str, ...] = ()
    context_support_scaler: StandardScaler | None = None
    context_support_model: NearestNeighbors | None = None
    context_support_threshold: float | None = None
    action_context_scaler: StandardScaler | None = None
    action_context_model: NearestNeighbors | None = None
    support_train_actions: np.ndarray | None = None
    conditional_action_lower_quantile: float = 0.10
    conditional_action_upper_quantile: float = 0.90
    abstain_on_context_ood: bool = False


def _daily_action_history(initial_data: pd.DataFrame) -> pd.DataFrame:
    """Recover the frozen daily PPA/DO audit series from materialized columns."""

    specifications = (
        (3, "PPA_older", "DO_older"),
        (2, "PPA_recent", "DO_recent"),
        (1, "PPA_historical_action", "DO_historical_action"),
        (0, "PPA_current_observed", "DO_current_observed"),
    )
    records: list[pd.DataFrame] = []
    for offset, ppa_column, do_column in specifications:
        part = pd.DataFrame(
            {
                "Date": initial_data["Date"] - pd.Timedelta(days=offset),
                "PPA": pd.to_numeric(initial_data[ppa_column], errors="coerce"),
                "DO": pd.to_numeric(initial_data[do_column], errors="coerce"),
            }
        )
        records.append(part)
    stacked = pd.concat(records, ignore_index=True).sort_values("Date")
    for column in ("PPA", "DO"):
        conflicts = stacked.groupby("Date", observed=True)[column].apply(
            lambda values: values.dropna().nunique() > 1
        )
        if conflicts.any():
            raise SurrogateRLError(f"Conflicting {column} audit values share a date.")
    return stacked.groupby("Date", observed=True, as_index=True)[["PPA", "DO"]].first()


def _period_starts(
    frame: pd.DataFrame,
    *,
    start: str,
    end: str,
    horizon: int,
    stride: int,
) -> tuple[int, ...]:
    dates = pd.DatetimeIndex(pd.to_datetime(frame["Date"])).normalize()
    lower = pd.Timestamp(start).normalize()
    upper = pd.Timestamp(end).normalize()
    candidates: list[int] = []
    for position, date in enumerate(dates):
        if date < lower or date > upper or position + horizon > len(dates):
            continue
        episode_dates = dates[position : position + horizon]
        if episode_dates[-1] > upper:
            continue
        expected = pd.date_range(date, periods=horizon, freq="D")
        if episode_dates.equals(expected):
            candidates.append(position)
    if stride <= 1:
        return tuple(candidates)
    selected: list[int] = []
    next_allowed = pd.Timestamp.min
    for position in candidates:
        date = dates[position]
        if date >= next_allowed:
            selected.append(position)
            next_allowed = date + pd.Timedelta(days=int(stride))
    return tuple(selected)


def _historical_observation_matrix(
    frame: pd.DataFrame, base_features: Sequence[str]
) -> np.ndarray:
    return np.column_stack(
        [
            frame.loc[:, list(base_features)].to_numpy(float),
            frame["MLSS_current"].to_numpy(float),
            frame["PPA_older"].to_numpy(float),
            frame["PPA_recent"].to_numpy(float),
            frame["DO_older"].to_numpy(float),
            frame["DO_recent"].to_numpy(float),
        ]
    )


def _named_numeric_matrix(frame: pd.DataFrame, names: Sequence[str]) -> np.ndarray:
    missing = set(names).difference(frame.columns)
    if missing:
        raise SurrogateRLError(f"Support context columns are missing: {sorted(missing)}")
    values = frame.loc[:, list(names)].to_numpy(float)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise SurrogateRLError("Support context contains non-finite values.")
    return values


def _support_matrix(
    frame: pd.DataFrame, union_features: Sequence[str], actions: np.ndarray
) -> np.ndarray:
    return np.column_stack(
        [frame.loc[:, list(union_features)].to_numpy(float), np.asarray(actions, dtype=float)]
    )


def build_scenario_data(
    feature_set: PairedFeatureSet,
    initial_data: pd.DataFrame,
    final_bundle: FinalProxyBundle,
    rl_config: Mapping[str, Any],
) -> ScenarioData:
    """Build reproducible historical trajectories and training-only constraints."""

    horizon = int(rl_config["episode_horizon_days"])
    if horizon < 2:
        raise SurrogateRLError("Episode horizon must be at least two days.")
    preferences = tuple(float(value) for value in rl_config["preference_weights"])
    if preferences != DEFAULT_PREFERENCES:
        raise SurrogateRLError(f"Preferences must be exactly {DEFAULT_PREFERENCES}.")
    tn_bundle = feature_set.bundles["TN_out"]
    dec_bundle = feature_set.bundles["DEC"]
    dates = pd.DatetimeIndex(pd.to_datetime(tn_bundle.anchor["Date"])).normalize()
    if not dates.equals(pd.DatetimeIndex(pd.to_datetime(dec_bundle.anchor["Date"]))):
        raise SurrogateRLError("TN and DEC scenarios must share one initial-data panel.")
    union_features = tuple(tn_bundle.feature_names[FEATURE_KEY])
    if len(final_bundle.tn_feature_names) != len(union_features):
        raise SurrogateRLError("Final TN feature count differs from the scenario schema.")
    base_features = tuple(feature for feature in union_features if feature not in CONTROL_FEATURES)
    X = tn_bundle.matrix(FEATURE_KEY).copy()
    X.insert(0, "Date", dates)

    source = initial_data.copy()
    source["Date"] = pd.to_datetime(source["Date"], errors="raise").dt.normalize()
    source["decision_date"] = pd.to_datetime(
        source["decision_date"], errors="raise"
    ).dt.normalize()
    direct_columns = (
        "Date",
        "decision_date",
        "PPA_older",
        "PPA_recent",
        "PPA_historical_action",
        "DO_older",
        "DO_recent",
        "DO_historical_action",
        "MLSS_current",
    )
    missing = set(direct_columns).difference(source.columns)
    if missing:
        raise SurrogateRLError(f"Initial dataset lacks RL state columns: {sorted(missing)}")
    frame = X.merge(
        source.loc[:, list(direct_columns)], on="Date", how="left", validate="one_to_one"
    )
    state_columns = direct_columns[2:]
    frame.loc[:, list(state_columns)] = frame.loc[:, list(state_columns)].apply(
        pd.to_numeric, errors="coerce"
    )
    finite_state = np.isfinite(frame.loc[:, list(state_columns)].to_numpy(float)).all(axis=1)
    frame = frame.loc[finite_state].sort_values("Date").reset_index(drop=True)
    train_mask = frame["Date"].between(
        pd.Timestamp(str(rl_config["train_start"])),
        pd.Timestamp(str(rl_config["train_end"])),
    )
    if train_mask.sum() < 100:
        raise SurrogateRLError("Too few training scenarios to define the environment.")
    fill_values = frame.loc[train_mask, list(union_features)].median(numeric_only=True)
    missing_before = frame.loc[:, list(union_features)].isna().sum().to_dict()
    frame.loc[:, list(union_features)] = frame.loc[:, list(union_features)].fillna(fill_values)
    if not np.isfinite(frame.loc[:, list(union_features)].to_numpy(float)).all():
        raise SurrogateRLError("Training-only median filling did not make scenarios finite.")

    historical_actions = frame.loc[
        train_mask, ["PPA_historical_action", "DO_historical_action"]
    ].to_numpy(float)
    lower_q = float(rl_config["action_lower_quantile"])
    upper_q = float(rl_config["action_upper_quantile"])
    action_low = np.quantile(historical_actions, lower_q, axis=0)
    action_high = np.quantile(historical_actions, upper_q, axis=0)
    daily_actions = _daily_action_history(source)
    daily_train = daily_actions.loc[
        pd.Timestamp(str(rl_config["train_start"])) - pd.Timedelta(days=3) : pd.Timestamp(
            str(rl_config["train_end"])
        ),
        ["PPA", "DO"],
    ].copy()
    if len(daily_train) != 731:
        raise SurrogateRLError(f"Expected 731 training-period daily actions, observed {len(daily_train)}.")
    changes = daily_train.apply(pd.to_numeric, errors="coerce").diff().abs()
    action_rate = changes.quantile(float(rl_config["action_rate_quantile"])).to_numpy(float)
    if not (
        np.isfinite(action_low).all()
        and np.isfinite(action_high).all()
        and np.isfinite(action_rate).all()
        and np.all(action_high > action_low)
        and np.all(action_rate > 0)
    ):
        raise SurrogateRLError("Empirical action bounds or rates are invalid.")

    train_frame = frame.loc[train_mask].reset_index(drop=True)
    objective_predictions = np.column_stack(
        [
            final_bundle.predict_tn(train_frame.loc[:, list(union_features)]),
            final_bundle.predict_dec(
                train_frame.loc[:, list(dec_bundle.feature_names[FEATURE_KEY])]
            ),
        ]
    )
    objective_low = np.quantile(
        objective_predictions, float(rl_config["objective_lower_quantile"]), axis=0
    )
    objective_high = np.quantile(
        objective_predictions, float(rl_config["objective_upper_quantile"]), axis=0
    )
    if not np.all(objective_high > objective_low):
        raise SurrogateRLError("Objective normalization range is degenerate.")

    train_support = _support_matrix(train_frame, union_features, historical_actions)
    support_scaler = StandardScaler().fit(train_support)
    scaled_support = support_scaler.transform(train_support)
    neighbors = int(rl_config["support_neighbors"])
    if neighbors < 2 or neighbors >= len(train_support):
        raise SurrogateRLError("support_neighbors is incompatible with training rows.")
    support_model = NearestNeighbors(n_neighbors=neighbors + 1).fit(scaled_support)
    distances, _ = support_model.kneighbors(scaled_support)
    loo_mean = distances[:, 1 : neighbors + 1].mean(axis=1)
    support_threshold = float(
        np.quantile(loo_mean, float(rl_config["support_threshold_quantile"]))
    )
    if not np.isfinite(support_threshold) or support_threshold <= 0:
        raise SurrogateRLError("The support threshold is invalid.")

    support_mode = str(rl_config.get("support_mode", "joint_legacy"))
    context_feature_names: tuple[str, ...] = ()
    action_context_feature_names: tuple[str, ...] = ()
    context_support_scaler: StandardScaler | None = None
    context_support_model: NearestNeighbors | None = None
    context_support_threshold: float | None = None
    action_context_scaler: StandardScaler | None = None
    action_context_model: NearestNeighbors | None = None
    support_train_actions: np.ndarray | None = None
    conditional_lower = float(rl_config.get("conditional_action_lower_quantile", 0.10))
    conditional_upper = float(rl_config.get("conditional_action_upper_quantile", 0.90))
    abstain_on_context_ood = bool(rl_config.get("abstain_on_context_ood", False))
    if support_mode == "split_context_conditional_action_v2":
        excluded = {
            str(value) for value in rl_config.get("context_excluded_features", ())
        }
        context_feature_names = tuple(
            feature for feature in base_features if feature not in excluded
        )
        if "MLSS_current" not in context_feature_names:
            context_feature_names = (*context_feature_names, "MLSS_current")
        action_context_feature_names = (
            *context_feature_names,
            "PPA_older",
            "PPA_recent",
            "DO_older",
            "DO_recent",
        )
        context_train = _named_numeric_matrix(train_frame, context_feature_names)
        context_support_scaler = StandardScaler().fit(context_train)
        scaled_context = context_support_scaler.transform(context_train)
        context_support_model = NearestNeighbors(n_neighbors=neighbors + 1).fit(
            scaled_context
        )
        context_distances, _ = context_support_model.kneighbors(scaled_context)
        context_loo_mean = context_distances[:, 1 : neighbors + 1].mean(axis=1)
        context_support_threshold = float(
            np.quantile(
                context_loo_mean,
                float(rl_config["support_threshold_quantile"]),
            )
        )
        conditional_neighbors = int(rl_config.get("conditional_action_neighbors", 20))
        if conditional_neighbors < 5 or conditional_neighbors >= len(train_frame):
            raise SurrogateRLError("conditional_action_neighbors is incompatible with training rows.")
        if not (0.0 <= conditional_lower < conditional_upper <= 1.0):
            raise SurrogateRLError("Conditional action quantiles are invalid.")
        action_context_train = _named_numeric_matrix(
            train_frame, action_context_feature_names
        )
        action_context_scaler = StandardScaler().fit(action_context_train)
        action_context_model = NearestNeighbors(
            n_neighbors=conditional_neighbors
        ).fit(action_context_scaler.transform(action_context_train))
        support_train_actions = historical_actions.copy()
        if (
            not np.isfinite(context_support_threshold)
            or context_support_threshold <= 0
        ):
            raise SurrogateRLError("The context support threshold is invalid.")
    elif support_mode != "joint_legacy":
        raise SurrogateRLError(f"Unknown support_mode: {support_mode}")

    observation_names = (
        *base_features,
        "MLSS_current",
        "PPA_older",
        "PPA_recent",
        "DO_older",
        "DO_recent",
    )
    observation_train = _historical_observation_matrix(train_frame, base_features)
    observation_mean = observation_train.mean(axis=0)
    observation_scale = observation_train.std(axis=0, ddof=0)
    observation_scale = np.where(observation_scale > 1e-12, observation_scale, 1.0)

    train_starts = _period_starts(
        frame,
        start=str(rl_config["train_start"]),
        end=str(rl_config["train_end"]),
        horizon=horizon,
        stride=1,
    )
    evaluation_stride = int(rl_config["evaluation_episode_stride_days"])
    validation_starts = _period_starts(
        frame,
        start=str(rl_config["validation_start"]),
        end=str(rl_config["validation_end"]),
        horizon=horizon,
        stride=evaluation_stride,
    )
    test_starts = _period_starts(
        frame,
        start=str(rl_config["test_start"]),
        end=str(rl_config["test_end"]),
        horizon=horizon,
        stride=evaluation_stride,
    )
    if min(len(train_starts), len(validation_starts), len(test_starts)) < 3:
        raise SurrogateRLError(
            "The scenario split lacks sufficient complete episodes: "
            f"{len(train_starts)}, {len(validation_starts)}, {len(test_starts)}"
        )
    return ScenarioData(
        frame=frame,
        union_feature_names=union_features,
        observation_feature_names=tuple(observation_names),
        observation_mean=observation_mean,
        observation_scale=observation_scale,
        action_low=action_low,
        action_high=action_high,
        action_rate=action_rate,
        objective_low=objective_low,
        objective_high=objective_high,
        support_scaler=support_scaler,
        support_model=support_model,
        support_threshold=support_threshold,
        preferences=preferences,
        train_starts=train_starts,
        validation_starts=validation_starts,
        test_starts=test_starts,
        episode_horizon=horizon,
        metadata={
            "feature_missing_before_training_median_fill": missing_before,
            "feature_fill_values": {key: float(value) for key, value in fill_values.items()},
            "action_bounds_role": "training_period_empirical_quantiles_not_engineering_limits",
            "action_rate_role": "training_period_empirical_absolute_change_quantile",
            "support_role": (
                "split_exogenous_context_and_conditional_action_support"
                if support_mode == "split_context_conditional_action_v2"
                else "training_period_standardized_knn_loo_threshold"
            ),
            "MLSS_role": "historical_exogenous_slow_state_not_action",
        },
        support_mode=support_mode,
        context_feature_names=context_feature_names,
        action_context_feature_names=action_context_feature_names,
        context_support_scaler=context_support_scaler,
        context_support_model=context_support_model,
        context_support_threshold=context_support_threshold,
        action_context_scaler=action_context_scaler,
        action_context_model=action_context_model,
        support_train_actions=support_train_actions,
        conditional_action_lower_quantile=conditional_lower,
        conditional_action_upper_quantile=conditional_upper,
        abstain_on_context_ood=abstain_on_context_ood,
    )


class SurrogateRLEnv(gym.Env[np.ndarray, np.ndarray]):
    """Seven-day continuous-control proxy environment with a state queue."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        scenario: ScenarioData,
        proxies: FinalProxyBundle,
        *,
        start_positions: Sequence[int] | None = None,
        fixed_preference: float | None = None,
        support_penalty: float = 2.0,
        repair_penalty: float = 0.05,
        invalid_penalty: float = 10.0,
    ) -> None:
        super().__init__()
        self.scenario = scenario
        self.proxies = proxies
        self.start_positions = tuple(
            int(value) for value in (start_positions or scenario.train_starts)
        )
        if not self.start_positions:
            raise SurrogateRLError("At least one episode start is required.")
        if fixed_preference is not None and float(fixed_preference) not in scenario.preferences:
            raise SurrogateRLError("A fixed preference must be one of the registered weights.")
        self.fixed_preference = None if fixed_preference is None else float(fixed_preference)
        self.support_penalty = float(support_penalty)
        self.repair_penalty = float(repair_penalty)
        self.invalid_penalty = float(invalid_penalty)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(-10.0, 10.0, shape=(15,), dtype=np.float32)
        self._position = 0
        self._start = 0
        self._step_count = 0
        self._preference = 0.5
        self._ppa_queue = np.zeros(2, dtype=float)
        self._do_queue = np.zeros(2, dtype=float)
        self._previous_action = np.zeros(2, dtype=float)

    def _row(self) -> pd.Series:
        return self.scenario.frame.iloc[self._position]

    def _observation(self) -> np.ndarray:
        row = self._row()
        base_features = [
            feature
            for feature in self.scenario.union_feature_names
            if feature not in CONTROL_FEATURES
        ]
        raw = np.concatenate(
            [
                row.loc[base_features].to_numpy(float),
                np.array([float(row["MLSS_current"])]),
                self._ppa_queue,
                self._do_queue,
            ]
        )
        normalized = (raw - self.scenario.observation_mean) / self.scenario.observation_scale
        observation = np.concatenate([normalized, np.array([self._preference])])
        return np.clip(observation, -10.0, 10.0).astype(np.float32)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        options = options or {}
        if "start_position" in options:
            start = int(options["start_position"])
            if start not in self.start_positions:
                raise SurrogateRLError("Requested start_position is not registered for this env.")
        else:
            start = int(self.np_random.choice(np.asarray(self.start_positions, dtype=int)))
        if "preference" in options:
            preference = float(options["preference"])
        elif self.fixed_preference is not None:
            preference = self.fixed_preference
        else:
            preference = float(self.np_random.choice(np.asarray(self.scenario.preferences)))
        if preference not in self.scenario.preferences:
            raise SurrogateRLError("Episode preference is not registered.")
        self._start = start
        self._position = start
        self._step_count = 0
        self._preference = preference
        row = self._row()
        self._ppa_queue = np.array([row["PPA_older"], row["PPA_recent"]], dtype=float)
        self._do_queue = np.array([row["DO_older"], row["DO_recent"]], dtype=float)
        self._previous_action = np.array(
            [float(row["PPA_recent"]), float(row["DO_recent"])], dtype=float
        )
        info = {
            "start_position": start,
            "start_date": pd.Timestamp(row["Date"]).date().isoformat(),
            "preference_TN": preference,
            "preference_DEC": 1.0 - preference,
        }
        return self._observation(), info

    def _feature_frame(self, action: np.ndarray) -> pd.DataFrame:
        row = self._row()
        features = row.loc[list(self.scenario.union_feature_names)].copy()
        features["PPA"] = float(
            np.mean([self._ppa_queue[0], self._ppa_queue[1], action[0]])
        )
        features["DO"] = float(
            np.mean([self._do_queue[0], self._do_queue[1], action[1]])
        )
        return pd.DataFrame([features], columns=self.scenario.union_feature_names)

    def _support_distance(self, features: pd.DataFrame, action: np.ndarray) -> float:
        matrix = np.column_stack(
            [features.loc[:, list(self.scenario.union_feature_names)].to_numpy(float), action[None, :]]
        )
        scaled = self.scenario.support_scaler.transform(matrix)
        distances, _ = self.scenario.support_model.kneighbors(
            scaled, n_neighbors=self.scenario.support_model.n_neighbors - 1
        )
        return float(distances.mean())

    def _support_context_values(self, names: Sequence[str]) -> np.ndarray:
        row = self._row()
        overrides = {
            "PPA_older": float(self._ppa_queue[0]),
            "PPA_recent": float(self._ppa_queue[1]),
            "DO_older": float(self._do_queue[0]),
            "DO_recent": float(self._do_queue[1]),
        }
        values = np.asarray(
            [overrides.get(name, float(row[name])) for name in names], dtype=float
        )
        if not np.isfinite(values).all():
            raise SurrogateRLError("The current support context is non-finite.")
        return values

    def _context_support(self) -> tuple[float, bool]:
        scenario = self.scenario
        if (
            scenario.context_support_scaler is None
            or scenario.context_support_model is None
            or scenario.context_support_threshold is None
        ):
            raise SurrogateRLError("Split context support was not fitted.")
        raw = self._support_context_values(scenario.context_feature_names)[None, :]
        scaled = scenario.context_support_scaler.transform(raw)
        requested = scenario.context_support_model.n_neighbors
        distances, _ = scenario.context_support_model.kneighbors(
            scaled, n_neighbors=requested
        )
        row = distances[0]
        if row[0] <= 1e-12 and len(row) > 1:
            row = row[1:]
        else:
            row = row[:-1]
        distance = float(row.mean())
        return distance, bool(distance <= scenario.context_support_threshold)

    def _conditional_action_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        scenario = self.scenario
        if (
            scenario.action_context_scaler is None
            or scenario.action_context_model is None
            or scenario.support_train_actions is None
        ):
            raise SurrogateRLError("Conditional action support was not fitted.")
        raw = self._support_context_values(scenario.action_context_feature_names)[None, :]
        scaled = scenario.action_context_scaler.transform(raw)
        _, indices = scenario.action_context_model.kneighbors(scaled)
        local_actions = scenario.support_train_actions[indices[0]]
        local_low = np.quantile(
            local_actions, scenario.conditional_action_lower_quantile, axis=0
        )
        local_high = np.quantile(
            local_actions, scenario.conditional_action_upper_quantile, axis=0
        )
        lower = np.maximum(scenario.action_low, local_low)
        upper = np.minimum(scenario.action_high, local_high)
        invalid = upper <= lower + 1e-12
        lower[invalid] = scenario.action_low[invalid]
        upper[invalid] = scenario.action_high[invalid]
        return lower, upper

    def _step_split_support(
        self, raw_action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        scenario = self.scenario
        raw_action = np.clip(raw_action, -1.0, 1.0)
        proposed = self._previous_action + raw_action * scenario.action_rate
        globally_clipped = np.clip(proposed, scenario.action_low, scenario.action_high)
        absolute_bound_repair = np.abs(globally_clipped - proposed)
        context_distance, context_valid = self._context_support()
        conditional_low, conditional_high = self._conditional_action_bounds()
        conditional_clipped = np.clip(
            globally_clipped, conditional_low, conditional_high
        )
        conditional_repair = np.abs(conditional_clipped - globally_clipped)
        bounded_anchor = np.clip(
            self._previous_action, scenario.action_low, scenario.action_high
        )
        initial_anchor_repair = np.abs(bounded_anchor - self._previous_action)
        abstained = bool(scenario.abstain_on_context_ood and not context_valid)
        chosen = bounded_anchor if abstained else conditional_clipped
        chosen_features = self._feature_frame(chosen)
        action_valid = bool(
            np.all(chosen >= scenario.action_low - 1e-12)
            and np.all(chosen <= scenario.action_high + 1e-12)
        )
        conditional_action_valid = bool(
            np.all(chosen >= conditional_low - 1e-12)
            and np.all(chosen <= conditional_high + 1e-12)
        )
        support_valid = bool(context_valid and action_valid and conditional_action_valid)
        tn_prediction = float(self.proxies.predict_tn(chosen_features)[0])
        dec_prediction = float(self.proxies.predict_dec(chosen_features)[0])
        objective = np.array([tn_prediction, dec_prediction], dtype=float)
        normalized = (objective - scenario.objective_low) / (
            scenario.objective_high - scenario.objective_low
        )
        total_repair = np.abs(chosen - proposed)
        rate_normalized_repair = float(
            np.sum(total_repair / np.maximum(scenario.action_rate, 1e-12))
            + 2.0 * float(abstained)
        )
        reward = -float(
            self._preference * normalized[0]
            + (1.0 - self._preference) * normalized[1]
        )
        reward -= self.repair_penalty * rate_normalized_repair
        context_excess = max(
            0.0,
            context_distance / max(float(scenario.context_support_threshold), 1e-12)
            - 1.0,
        )
        reward -= self.support_penalty * context_excess**2
        if not context_valid:
            reward -= self.invalid_penalty
        repaired = bool(np.any(total_repair > 1e-12))
        current_row = self._row()
        info = {
            "Date": pd.Timestamp(current_row["Date"]).date().isoformat(),
            "decision_date": pd.Timestamp(current_row["decision_date"]).date().isoformat(),
            "preference_TN": self._preference,
            "preference_DEC": 1.0 - self._preference,
            "proposal_PPA": float(proposed[0]),
            "proposal_DO": float(proposed[1]),
            "applied_PPA": float(chosen[0]),
            "applied_DO": float(chosen[1]),
            "model_input_PPA": float(chosen_features.iloc[0]["PPA"]),
            "model_input_DO": float(chosen_features.iloc[0]["DO"]),
            "model_input_MLSS": float(chosen_features.iloc[0][SLOW_STATE_FEATURE]),
            "predicted_TN_out": tn_prediction,
            "predicted_DEC": dec_prediction,
            "support_distance": context_distance,
            "support_threshold": float(scenario.context_support_threshold),
            "support_valid": support_valid,
            "context_support_valid": context_valid,
            "conditional_action_valid": conditional_action_valid,
            "context_abstained": abstained,
            "conditional_low_PPA": float(conditional_low[0]),
            "conditional_high_PPA": float(conditional_high[0]),
            "conditional_low_DO": float(conditional_low[1]),
            "conditional_high_DO": float(conditional_high[1]),
            "support_backtrack_fraction": 0.0 if repaired else 1.0,
            "absolute_bound_repair_PPA": float(absolute_bound_repair[0]),
            "absolute_bound_repair_DO": float(absolute_bound_repair[1]),
            "conditional_repair_PPA": float(conditional_repair[0]),
            "conditional_repair_DO": float(conditional_repair[1]),
            "total_action_repair_PPA": float(total_repair[0]),
            "total_action_repair_DO": float(total_repair[1]),
            "initial_anchor_repair_PPA": float(initial_anchor_repair[0]),
            "initial_anchor_repair_DO": float(initial_anchor_repair[1]),
            "reward": float(reward),
            "proxy_simulation_only": True,
        }
        self._ppa_queue = np.array([self._ppa_queue[1], chosen[0]], dtype=float)
        self._do_queue = np.array([self._do_queue[1], chosen[1]], dtype=float)
        self._previous_action = chosen
        self._step_count += 1
        truncated = self._step_count >= scenario.episode_horizon
        terminated = False
        if not truncated:
            self._position += 1
        return self._observation(), float(reward), terminated, truncated, info

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        raw_action = np.asarray(action, dtype=float).reshape(-1)
        if raw_action.shape != (2,) or not np.isfinite(raw_action).all():
            raise SurrogateRLError("Action must contain two finite values.")
        if self.scenario.support_mode == "split_context_conditional_action_v2":
            return self._step_split_support(raw_action)
        raw_action = np.clip(raw_action, -1.0, 1.0)
        proposed = self._previous_action + raw_action * self.scenario.action_rate
        clipped = np.clip(proposed, self.scenario.action_low, self.scenario.action_high)
        absolute_repair = np.abs(clipped - proposed)
        # Historical episode initialization can lie outside the training-period
        # q05-q95 experimental action domain.  The backtracking anchor must itself
        # be in-domain; otherwise fraction=0 could silently apply an out-of-bounds
        # action even though ``clipped`` was valid.
        bounded_anchor = np.clip(
            self._previous_action,
            self.scenario.action_low,
            self.scenario.action_high,
        )
        initial_anchor_repair = np.abs(bounded_anchor - self._previous_action)
        chosen = bounded_anchor.copy()
        chosen_features = self._feature_frame(chosen)
        chosen_distance = self._support_distance(chosen_features, chosen)
        support_fraction = 0.0
        for fraction in (1.0, 0.75, 0.5, 0.25, 0.0):
            candidate = bounded_anchor + fraction * (clipped - bounded_anchor)
            candidate_features = self._feature_frame(candidate)
            distance = self._support_distance(candidate_features, candidate)
            if distance <= self.scenario.support_threshold:
                chosen = candidate
                chosen_features = candidate_features
                chosen_distance = distance
                support_fraction = fraction
                break
        if np.any(chosen < self.scenario.action_low) or np.any(
            chosen > self.scenario.action_high
        ):
            raise SurrogateRLError("Action repair returned an out-of-domain action.")
        support_valid = bool(chosen_distance <= self.scenario.support_threshold)
        tn_prediction = float(self.proxies.predict_tn(chosen_features)[0])
        dec_prediction = float(self.proxies.predict_dec(chosen_features)[0])
        objective = np.array([tn_prediction, dec_prediction], dtype=float)
        normalized = (objective - self.scenario.objective_low) / (
            self.scenario.objective_high - self.scenario.objective_low
        )
        rate_normalized_repair = float(
            np.sum(absolute_repair / np.maximum(self.scenario.action_rate, 1e-12))
            + 2.0 * (1.0 - support_fraction)
        )
        reward = -float(
            self._preference * normalized[0] + (1.0 - self._preference) * normalized[1]
        )
        reward -= self.repair_penalty * rate_normalized_repair
        support_excess = max(0.0, chosen_distance / self.scenario.support_threshold - 1.0)
        reward -= self.support_penalty * support_excess**2
        if not support_valid:
            reward -= self.invalid_penalty
        current_row = self._row()
        info = {
            "Date": pd.Timestamp(current_row["Date"]).date().isoformat(),
            "decision_date": pd.Timestamp(current_row["decision_date"]).date().isoformat(),
            "preference_TN": self._preference,
            "preference_DEC": 1.0 - self._preference,
            "proposal_PPA": float(proposed[0]),
            "proposal_DO": float(proposed[1]),
            "applied_PPA": float(chosen[0]),
            "applied_DO": float(chosen[1]),
            "model_input_PPA": float(chosen_features.iloc[0]["PPA"]),
            "model_input_DO": float(chosen_features.iloc[0]["DO"]),
            "model_input_MLSS": float(chosen_features.iloc[0][SLOW_STATE_FEATURE]),
            "predicted_TN_out": tn_prediction,
            "predicted_DEC": dec_prediction,
            "support_distance": chosen_distance,
            "support_threshold": self.scenario.support_threshold,
            "support_valid": support_valid,
            "support_backtrack_fraction": support_fraction,
            "absolute_bound_repair_PPA": float(absolute_repair[0]),
            "absolute_bound_repair_DO": float(absolute_repair[1]),
            "initial_anchor_repair_PPA": float(initial_anchor_repair[0]),
            "initial_anchor_repair_DO": float(initial_anchor_repair[1]),
            "reward": float(reward),
            "proxy_simulation_only": True,
        }
        self._ppa_queue = np.array([self._ppa_queue[1], chosen[0]], dtype=float)
        self._do_queue = np.array([self._do_queue[1], chosen[1]], dtype=float)
        self._previous_action = chosen
        self._step_count += 1
        truncated = self._step_count >= self.scenario.episode_horizon
        terminated = False
        if not truncated:
            self._position += 1
        observation = self._observation()
        return observation, float(reward), terminated, truncated, info


def rollout_policy(
    model: Any,
    env: SurrogateRLEnv,
    *,
    algorithm: str,
    training_seed: int,
    start_positions: Sequence[int],
    preferences: Sequence[float],
) -> pd.DataFrame:
    """Evaluate a learned policy on fixed scenarios and preferences."""

    records: list[dict[str, Any]] = []
    for preference in preferences:
        for episode_id, start in enumerate(start_positions, start=1):
            observation, _ = env.reset(
                options={"start_position": int(start), "preference": float(preference)}
            )
            done = False
            step_id = 0
            while not done:
                action, _ = model.predict(observation, deterministic=True)
                observation, _, terminated, truncated, info = env.step(action)
                step_id += 1
                records.append(
                    {
                        "method": algorithm,
                        "method_role": "learned_RL_policy",
                        "training_seed": int(training_seed),
                        "preference_TN": float(preference),
                        "episode_id": episode_id,
                        "episode_start_position": int(start),
                        "step": step_id,
                        **info,
                    }
                )
                done = terminated or truncated
    return pd.DataFrame.from_records(records)


def rollout_simple_baseline(
    kind: str,
    env: SurrogateRLEnv,
    *,
    start_positions: Sequence[int],
    preferences: Sequence[float],
    seed: int = 20260823,
) -> pd.DataFrame:
    """Evaluate keep-previous or random-feasible action baselines."""

    if kind not in {"KeepPrevious", "RandomFeasible"}:
        raise SurrogateRLError(f"Unknown simple baseline: {kind}")
    rng = np.random.default_rng(int(seed))
    records: list[dict[str, Any]] = []
    for preference in preferences:
        for episode_id, start in enumerate(start_positions, start=1):
            observation, _ = env.reset(
                options={"start_position": int(start), "preference": float(preference)}
            )
            done = False
            step_id = 0
            while not done:
                action = (
                    np.zeros(2, dtype=np.float32)
                    if kind == "KeepPrevious"
                    else rng.uniform(-1.0, 1.0, size=2).astype(np.float32)
                )
                observation, _, terminated, truncated, info = env.step(action)
                step_id += 1
                records.append(
                    {
                        "method": kind,
                        "method_role": "transparent_baseline",
                        "training_seed": int(seed),
                        "preference_TN": float(preference),
                        "episode_id": episode_id,
                        "episode_start_position": int(start),
                        "step": step_id,
                        **info,
                    }
                )
                done = terminated or truncated
    return pd.DataFrame.from_records(records)


def rollout_historical_baseline(
    scenario: ScenarioData,
    proxies: FinalProxyBundle,
    *,
    start_positions: Sequence[int],
    preferences: Sequence[float],
) -> pd.DataFrame:
    """Evaluate the observed historical PPA/DO trajectory without action projection."""

    records: list[dict[str, Any]] = []
    for preference in preferences:
        for episode_id, start in enumerate(start_positions, start=1):
            for step in range(scenario.episode_horizon):
                row = scenario.frame.iloc[int(start) + step]
                features = pd.DataFrame(
                    [row.loc[list(scenario.union_feature_names)]],
                    columns=scenario.union_feature_names,
                )
                action = np.array(
                    [row["PPA_historical_action"], row["DO_historical_action"]], dtype=float
                )
                if scenario.support_mode == "split_context_conditional_action_v2":
                    if (
                        scenario.context_support_scaler is None
                        or scenario.context_support_model is None
                        or scenario.context_support_threshold is None
                        or scenario.action_context_scaler is None
                        or scenario.action_context_model is None
                        or scenario.support_train_actions is None
                    ):
                        raise SurrogateRLError("Split support artifacts are incomplete.")
                    context = _named_numeric_matrix(
                        pd.DataFrame([row]), scenario.context_feature_names
                    )
                    scaled_context = scenario.context_support_scaler.transform(context)
                    requested = scenario.context_support_model.n_neighbors
                    distances, _ = scenario.context_support_model.kneighbors(
                        scaled_context, n_neighbors=requested
                    )
                    distance_row = distances[0]
                    if distance_row[0] <= 1e-12 and len(distance_row) > 1:
                        distance_row = distance_row[1:]
                    else:
                        distance_row = distance_row[:-1]
                    distance = float(distance_row.mean())
                    context_valid = bool(distance <= scenario.context_support_threshold)
                    action_context = _named_numeric_matrix(
                        pd.DataFrame([row]), scenario.action_context_feature_names
                    )
                    scaled_action_context = scenario.action_context_scaler.transform(
                        action_context
                    )
                    _, neighbor_indices = scenario.action_context_model.kneighbors(
                        scaled_action_context
                    )
                    local_actions = scenario.support_train_actions[neighbor_indices[0]]
                    local_low = np.maximum(
                        scenario.action_low,
                        np.quantile(
                            local_actions,
                            scenario.conditional_action_lower_quantile,
                            axis=0,
                        ),
                    )
                    local_high = np.minimum(
                        scenario.action_high,
                        np.quantile(
                            local_actions,
                            scenario.conditional_action_upper_quantile,
                            axis=0,
                        ),
                    )
                    invalid_bounds = local_high <= local_low + 1e-12
                    local_low[invalid_bounds] = scenario.action_low[invalid_bounds]
                    local_high[invalid_bounds] = scenario.action_high[invalid_bounds]
                    conditional_valid = bool(
                        np.all(action >= local_low - 1e-12)
                        and np.all(action <= local_high + 1e-12)
                    )
                    support_valid = bool(context_valid and conditional_valid)
                    support_threshold = float(scenario.context_support_threshold)
                else:
                    support = _support_matrix(
                        pd.DataFrame([row]), scenario.union_feature_names, action[None, :]
                    )
                    scaled = scenario.support_scaler.transform(support)
                    distances, _ = scenario.support_model.kneighbors(
                        scaled, n_neighbors=scenario.support_model.n_neighbors - 1
                    )
                    distance = float(distances.mean())
                    context_valid = bool(distance <= scenario.support_threshold)
                    conditional_valid = context_valid
                    support_valid = context_valid
                    support_threshold = float(scenario.support_threshold)
                objective = np.array(
                    [proxies.predict_tn(features)[0], proxies.predict_dec(features)[0]], dtype=float
                )
                normalized = (objective - scenario.objective_low) / (
                    scenario.objective_high - scenario.objective_low
                )
                reward = -float(
                    preference * normalized[0] + (1.0 - preference) * normalized[1]
                )
                records.append(
                    {
                        "method": "HistoricalObserved",
                        "method_role": "observed_behavior_reference_not_feasible_action_policy",
                        "training_seed": 0,
                        "preference_TN": float(preference),
                        "episode_id": episode_id,
                        "episode_start_position": int(start),
                        "step": step + 1,
                        "Date": pd.Timestamp(row["Date"]).date().isoformat(),
                        "decision_date": pd.Timestamp(row["decision_date"]).date().isoformat(),
                        "preference_DEC": 1.0 - preference,
                        "proposal_PPA": float(action[0]),
                        "proposal_DO": float(action[1]),
                        "applied_PPA": float(action[0]),
                        "applied_DO": float(action[1]),
                        "model_input_PPA": float(features.iloc[0]["PPA"]),
                        "model_input_DO": float(features.iloc[0]["DO"]),
                        "model_input_MLSS": float(features.iloc[0][SLOW_STATE_FEATURE]),
                        "predicted_TN_out": float(objective[0]),
                        "predicted_DEC": float(objective[1]),
                        "support_distance": distance,
                        "support_threshold": support_threshold,
                        "support_valid": support_valid,
                        "context_support_valid": context_valid,
                        "conditional_action_valid": conditional_valid,
                        "context_abstained": False,
                        "support_backtrack_fraction": 1.0,
                        "absolute_bound_repair_PPA": 0.0,
                        "absolute_bound_repair_DO": 0.0,
                        "conditional_repair_PPA": 0.0,
                        "conditional_repair_DO": 0.0,
                        "total_action_repair_PPA": 0.0,
                        "total_action_repair_DO": 0.0,
                        "initial_anchor_repair_PPA": 0.0,
                        "initial_anchor_repair_DO": 0.0,
                        "reward": reward,
                        "proxy_simulation_only": True,
                    }
                )
    return pd.DataFrame.from_records(records)


def summarize_episodes(trajectories: pd.DataFrame) -> pd.DataFrame:
    grouping = [
        "method",
        "method_role",
        "training_seed",
        "preference_TN",
        "episode_id",
        "episode_start_position",
    ]
    aggregation: dict[str, tuple[str, Any]] = {
        "start_date": ("Date", "min"),
        "end_date": ("Date", "max"),
        "days": ("step", "size"),
        "mean_TN_out": ("predicted_TN_out", "mean"),
        "mean_DEC": ("predicted_DEC", "mean"),
        "cumulative_DEC": ("predicted_DEC", "sum"),
        "cumulative_reward": ("reward", "sum"),
        "mean_PPA": ("applied_PPA", "mean"),
        "mean_DO": ("applied_DO", "mean"),
        "support_valid_rate": ("support_valid", "mean"),
        "mean_support_distance": ("support_distance", "mean"),
        "repair_rate": (
            "support_backtrack_fraction",
            lambda x: float(np.mean(x < 1.0)),
        ),
    }
    optional = {
        "context_support_valid_rate": ("context_support_valid", "mean"),
        "conditional_action_valid_rate": ("conditional_action_valid", "mean"),
        "context_abstention_rate": ("context_abstained", "mean"),
        "mean_total_action_repair_PPA": ("total_action_repair_PPA", "mean"),
        "mean_total_action_repair_DO": ("total_action_repair_DO", "mean"),
    }
    aggregation.update(
        {name: spec for name, spec in optional.items() if spec[0] in trajectories.columns}
    )
    summary = trajectories.groupby(grouping, observed=True).agg(**aggregation).reset_index()
    return summary


def non_dominated_mask(objectives: np.ndarray) -> np.ndarray:
    values = np.asarray(objectives, dtype=float)
    if values.ndim != 2 or values.shape[1] != 2 or not np.isfinite(values).all():
        raise SurrogateRLError("Pareto objectives must be a finite n-by-2 matrix.")
    indices = NonDominatedSorting().do(values, only_non_dominated_front=True)
    mask = np.zeros(len(values), dtype=bool)
    mask[np.asarray(indices, dtype=int)] = True
    return mask


def algorithm_multiobjective_summary(
    episode_summary: pd.DataFrame,
    *,
    objective_low: np.ndarray,
    objective_high: np.ndarray,
    reference_point: Sequence[float] = (1.1, 1.1),
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compute empirical Pareto points, hypervolume and IGD+ without a total score."""

    points = (
        episode_summary.groupby(
            ["method", "method_role", "training_seed", "preference_TN"], observed=True
        )
        .agg(
            mean_TN_out=("mean_TN_out", "mean"),
            mean_DEC=("mean_DEC", "mean"),
            sd_episode_TN=("mean_TN_out", "std"),
            sd_episode_DEC=("mean_DEC", "std"),
            n_episodes=("episode_id", "size"),
            support_valid_rate=("support_valid_rate", "mean"),
            repair_rate=("repair_rate", "mean"),
        )
        .reset_index()
    )
    raw = points[["mean_TN_out", "mean_DEC"]].to_numpy(float)
    normalized = (raw - np.asarray(objective_low, float)) / (
        np.asarray(objective_high, float) - np.asarray(objective_low, float)
    )
    points["normalized_TN"] = normalized[:, 0]
    points["normalized_DEC"] = normalized[:, 1]
    points["non_dominated_global"] = non_dominated_mask(normalized)
    reference_front = np.unique(
        normalized[points["non_dominated_global"].to_numpy(bool)], axis=0
    )
    hv_indicator = HV(ref_point=np.asarray(reference_point, dtype=float))
    records: list[dict[str, Any]] = []
    for (method, role), group in points.groupby(["method", "method_role"], observed=True):
        method_values = group[["normalized_TN", "normalized_DEC"]].to_numpy(float)
        method_front = method_values[non_dominated_mask(method_values)]
        inside_reference = np.all(method_front < np.asarray(reference_point, dtype=float), axis=1)
        hv = float(hv_indicator(method_front[inside_reference])) if inside_reference.any() else 0.0
        igd_plus = float(IGDPlus(reference_front)(method_front))
        records.append(
            {
                "method": method,
                "method_role": role,
                "hypervolume": hv,
                "IGD_plus_to_empirical_union_front": igd_plus,
                "non_dominated_points": int(len(method_front)),
                "total_points": int(len(method_values)),
                "mean_support_valid_rate": float(group["support_valid_rate"].mean()),
                "mean_repair_rate": float(group["repair_rate"].mean()),
                "ranking_rule": "HV_desc_then_IGDplus_asc_no_composite_score",
            }
        )
    algorithm = pd.DataFrame.from_records(records).sort_values(
        ["hypervolume", "IGD_plus_to_empirical_union_front"],
        ascending=[False, True],
    )
    algorithm["HV_rank"] = np.arange(1, len(algorithm) + 1)
    rl_only = algorithm.loc[algorithm["method_role"].eq("learned_RL_policy")]
    if not rl_only.empty:
        best = rl_only.iloc[0]["method"]
        worst = rl_only.iloc[-1]["method"]
        algorithm["headline_role"] = "other_complete_result"
        algorithm.loc[algorithm["method"].eq(best), "headline_role"] = "best_RL_by_HV"
        algorithm.loc[algorithm["method"].eq(worst), "headline_role"] = (
            "worst_RL_posthoc_comparator_not_formal_baseline"
        )
    else:
        algorithm["headline_role"] = "not_applicable"
    return points, algorithm.reset_index(drop=True), pd.DataFrame(
        reference_front, columns=["normalized_TN", "normalized_DEC"]
    )


def benchmark_proxy_step(env: SurrogateRLEnv, *, steps: int = 100) -> dict[str, float]:
    """Small deterministic benchmark used before selecting the training budget."""

    import time

    observation, _ = env.reset(seed=123)
    durations: list[float] = []
    completed = 0
    while completed < int(steps):
        start = time.perf_counter()
        observation, _, terminated, truncated, _ = env.step(np.zeros(2, dtype=np.float32))
        durations.append(time.perf_counter() - start)
        completed += 1
        if terminated or truncated:
            observation, _ = env.reset()
    values = np.asarray(durations, dtype=float)
    return {
        "steps": float(len(values)),
        "mean_seconds_per_step": float(values.mean()),
        "p50_seconds_per_step": float(np.quantile(values, 0.5)),
        "p95_seconds_per_step": float(np.quantile(values, 0.95)),
        "estimated_seconds_for_90000_steps_proxy_only": float(values.mean() * 90_000),
    }
