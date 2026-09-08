from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from taici.surrogate_rl import (
    CONTROL_FEATURES,
    DEFAULT_PREFERENCES,
    SLOW_STATE_FEATURE,
    ScenarioData,
    SurrogateRLEnv,
    SurrogateRLError,
    algorithm_multiobjective_summary,
    non_dominated_mask,
)


UNION_FEATURES = (
    "doy_sin",
    "doy_cos",
    "time_index_days",
    "Q",
    "COD",
    "TN_in",
    "NH3N",
    "T",
    *CONTROL_FEATURES,
    SLOW_STATE_FEATURE,
)


class _RecordingProxies:
    def __init__(self) -> None:
        self.tn_frames: list[pd.DataFrame] = []
        self.dec_frames: list[pd.DataFrame] = []

    def predict_tn(self, frame: pd.DataFrame) -> np.ndarray:
        self.tn_frames.append(frame.copy())
        return frame["PPA"].to_numpy(float)

    def predict_dec(self, frame: pd.DataFrame) -> np.ndarray:
        self.dec_frames.append(frame.copy())
        return frame["DO"].to_numpy(float)


def _scenario() -> ScenarioData:
    records: list[dict[str, object]] = []
    for position, date in enumerate(pd.date_range("2025-01-01", periods=3, freq="D")):
        record: dict[str, object] = {
            feature: float(position + feature_index + 1)
            for feature_index, feature in enumerate(UNION_FEATURES)
        }
        record.update(
            {
                "Date": date,
                "decision_date": date - pd.Timedelta(days=1),
                "PPA_older": 1.0 + position,
                "PPA_recent": 2.0 + position,
                "PPA_historical_action": 3.0 + position,
                "DO_older": 3.0 + position,
                "DO_recent": 4.0 + position,
                "DO_historical_action": 5.0 + position,
                "MLSS_current": 100.0 + position,
            }
        )
        records.append(record)
    frame = pd.DataFrame.from_records(records)
    support = np.column_stack(
        [
            frame.loc[:, list(UNION_FEATURES)].to_numpy(float),
            frame[["PPA_historical_action", "DO_historical_action"]].to_numpy(float),
        ]
    )
    support_scaler = StandardScaler().fit(support)
    support_model = NearestNeighbors(n_neighbors=2).fit(support_scaler.transform(support))
    base_features = tuple(feature for feature in UNION_FEATURES if feature not in CONTROL_FEATURES)
    observation_names = (
        *base_features,
        "MLSS_current",
        "PPA_older",
        "PPA_recent",
        "DO_older",
        "DO_recent",
    )
    return ScenarioData(
        frame=frame,
        union_feature_names=UNION_FEATURES,
        observation_feature_names=observation_names,
        observation_mean=np.zeros(14, dtype=float),
        observation_scale=np.ones(14, dtype=float),
        action_low=np.array([0.0, 0.0]),
        action_high=np.array([10.0, 10.0]),
        action_rate=np.array([1.0, 1.0]),
        objective_low=np.array([0.0, 0.0]),
        objective_high=np.array([10.0, 10.0]),
        support_scaler=support_scaler,
        support_model=support_model,
        support_threshold=1e9,
        preferences=DEFAULT_PREFERENCES,
        train_starts=(0,),
        validation_starts=(0,),
        test_starts=(0,),
        episode_horizon=2,
        metadata={"MLSS_role": "historical_exogenous_slow_state_not_action"},
    )


def test_surrogate_env_uses_two_controls_and_advances_input_queue() -> None:
    scenario = _scenario()
    proxies = _RecordingProxies()
    env = SurrogateRLEnv(
        scenario,
        proxies,  # type: ignore[arg-type]
        start_positions=(0,),
        fixed_preference=0.5,
    )

    observation, reset_info = env.reset(seed=7, options={"start_position": 0})
    assert observation.shape == (15,)
    assert observation.dtype == np.float32
    assert env.action_space.shape == (2,)
    assert reset_info["preference_TN"] == 0.5

    _, reward, terminated, truncated, first_info = env.step(np.zeros(2, dtype=np.float32))
    first_frame = proxies.tn_frames[-1]
    assert np.isfinite(reward)
    assert not terminated
    assert not truncated
    assert np.isclose(first_frame.iloc[0]["PPA"], np.mean([1.0, 2.0, 2.0]))
    assert np.isclose(first_frame.iloc[0]["DO"], np.mean([3.0, 4.0, 4.0]))
    assert first_frame.iloc[0][SLOW_STATE_FEATURE] == scenario.frame.iloc[0][SLOW_STATE_FEATURE]
    assert first_info["proxy_simulation_only"] is True

    _, _, terminated, truncated, _ = env.step(np.zeros(2, dtype=np.float32))
    second_frame = proxies.tn_frames[-1]
    assert not terminated
    assert truncated
    assert np.isclose(second_frame.iloc[0]["PPA"], 2.0)
    assert np.isclose(second_frame.iloc[0]["DO"], 4.0)


def test_surrogate_env_rejects_non_two_dimensional_action() -> None:
    env = SurrogateRLEnv(_scenario(), _RecordingProxies(), start_positions=(0,))  # type: ignore[arg-type]
    env.reset(seed=1)

    with pytest.raises(SurrogateRLError, match="two finite values"):
        env.step(np.zeros(3, dtype=np.float32))


def test_surrogate_env_repairs_out_of_domain_historical_anchor() -> None:
    scenario = _scenario()
    scenario.frame.loc[0, "PPA_recent"] = 20.0
    scenario.frame.loc[0, "DO_recent"] = -5.0
    env = SurrogateRLEnv(
        scenario,
        _RecordingProxies(),  # type: ignore[arg-type]
        start_positions=(0,),
    )
    env.reset(seed=2)

    _, _, _, _, info = env.step(np.zeros(2, dtype=np.float32))

    assert scenario.action_low[0] <= info["applied_PPA"] <= scenario.action_high[0]
    assert scenario.action_low[1] <= info["applied_DO"] <= scenario.action_high[1]
    assert info["initial_anchor_repair_PPA"] > 0
    assert info["initial_anchor_repair_DO"] > 0


def _split_support_scenario(*, context_threshold: float) -> ScenarioData:
    scenario = _scenario()
    context_names = ("doy_sin",)
    context = scenario.frame.loc[:, list(context_names)].to_numpy(float)
    context_scaler = StandardScaler().fit(context)
    context_model = NearestNeighbors(n_neighbors=2).fit(
        context_scaler.transform(context)
    )
    action_context_scaler = StandardScaler().fit(context)
    action_context_model = NearestNeighbors(n_neighbors=2).fit(
        action_context_scaler.transform(context)
    )
    return replace(
        scenario,
        support_mode="split_context_conditional_action_v2",
        context_feature_names=context_names,
        action_context_feature_names=context_names,
        context_support_scaler=context_scaler,
        context_support_model=context_model,
        context_support_threshold=context_threshold,
        action_context_scaler=action_context_scaler,
        action_context_model=action_context_model,
        support_train_actions=np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]]),
        conditional_action_lower_quantile=0.10,
        conditional_action_upper_quantile=0.90,
        abstain_on_context_ood=True,
    )


def test_split_support_repairs_actions_within_local_conditional_range() -> None:
    scenario = _split_support_scenario(context_threshold=1e9)
    env = SurrogateRLEnv(
        scenario,
        _RecordingProxies(),  # type: ignore[arg-type]
        start_positions=(0,),
    )
    env.reset(seed=3)

    _, _, _, _, info = env.step(np.ones(2, dtype=np.float32))

    assert info["context_support_valid"] is True
    assert info["conditional_action_valid"] is True
    assert info["context_abstained"] is False
    assert info["total_action_repair_PPA"] > 0
    assert info["conditional_low_PPA"] <= info["applied_PPA"] <= info["conditional_high_PPA"]


def test_split_support_abstains_when_context_is_out_of_domain() -> None:
    scenario = _split_support_scenario(context_threshold=0.01)
    scenario.frame.loc[0, "doy_sin"] = 1_000.0
    env = SurrogateRLEnv(
        scenario,
        _RecordingProxies(),  # type: ignore[arg-type]
        start_positions=(0,),
    )
    env.reset(seed=4)

    _, _, _, _, info = env.step(np.ones(2, dtype=np.float32))

    assert info["context_support_valid"] is False
    assert info["context_abstained"] is True
    assert info["support_valid"] is False
    assert info["applied_PPA"] == 2.0
    assert info["applied_DO"] == 4.0


def test_non_dominated_mask_marks_only_pareto_points() -> None:
    mask = non_dominated_mask(
        np.array(
            [
                [1.0, 3.0],
                [2.0, 2.0],
                [3.0, 1.0],
                [3.0, 3.0],
            ]
        )
    )

    assert mask.tolist() == [True, True, True, False]


def test_igd_plus_reference_front_deduplicates_repeated_preference_points() -> None:
    def episode_summary(repeated_preferences: tuple[float, ...]) -> pd.DataFrame:
        records: list[dict[str, object]] = []
        for preference in repeated_preferences:
            records.append(
                {
                    "method": "RepeatedBaseline",
                    "method_role": "transparent_baseline",
                    "training_seed": 0,
                    "preference_TN": preference,
                    "episode_id": 1,
                    "mean_TN_out": 2.0,
                    "mean_DEC": 8.0,
                    "support_valid_rate": 1.0,
                    "repair_rate": 0.0,
                }
            )
        for preference, tn, dec in ((0.25, 5.0, 5.0), (0.5, 8.0, 2.0)):
            records.append(
                {
                    "method": "Comparator",
                    "method_role": "learned_RL_policy",
                    "training_seed": 11,
                    "preference_TN": preference,
                    "episode_id": 1,
                    "mean_TN_out": tn,
                    "mean_DEC": dec,
                    "support_valid_rate": 1.0,
                    "repair_rate": 0.0,
                }
            )
        return pd.DataFrame.from_records(records)

    _, single_metrics, single_union = algorithm_multiobjective_summary(
        episode_summary((0.25,)),
        objective_low=np.array([0.0, 0.0]),
        objective_high=np.array([10.0, 10.0]),
    )
    _, repeated_metrics, repeated_union = algorithm_multiobjective_summary(
        episode_summary((0.25, 0.5, 0.75)),
        objective_low=np.array([0.0, 0.0]),
        objective_high=np.array([10.0, 10.0]),
    )

    assert len(repeated_union) == len(repeated_union.drop_duplicates()) == 3
    pd.testing.assert_frame_equal(single_union, repeated_union)
    single_igd = single_metrics.set_index("method")["IGD_plus_to_empirical_union_front"]
    repeated_igd = repeated_metrics.set_index("method")["IGD_plus_to_empirical_union_front"]
    pd.testing.assert_series_equal(single_igd.sort_index(), repeated_igd.sort_index())
