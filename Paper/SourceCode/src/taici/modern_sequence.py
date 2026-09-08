"""Leakage-safe, lightweight modern sequence baselines.

The architectures in this module are compact local implementations intended for controlled
small-data comparisons.  In particular, ``PATCHTST_LIKE`` and ``NBEATS_LIKE`` are inspired by
the corresponding architectural ideas; they are not the authors' official implementations and
they are not TimePFN.  This distinction is recorded in every fitted result.

The module is deliberately file-agnostic.  Callers supply in-memory arrays, and prediction rows
never participate in imputation, scaling, early stopping, or epoch selection.
"""

from __future__ import annotations

import copy
import random
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F


ModernArchitecture = Literal["TCN", "PATCHTST_LIKE", "NBEATS_LIKE"]


@dataclass(frozen=True)
class ModernSequenceConfig:
    """Conservative CPU settings for small sequence-regression datasets."""

    seeds: tuple[int, ...] = (11, 23, 37, 53, 71)
    max_epochs: int = 120
    patience: int = 15
    validation_fraction: float = 0.15
    batch_size: int = 32
    hidden_size: int = 32
    dropout: float = 0.10
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    minimum_improvement: float = 1e-7
    refit_full_training: bool = True
    patch_length: int = 4
    patch_stride: int = 2
    transformer_layers: int = 1
    transformer_heads: int = 2
    nbeats_blocks: int = 2


@dataclass(frozen=True)
class StrictCausalWindows:
    """Past-only windows and eligibility flags aligned to requested target dates."""

    sequences: np.ndarray
    target_dates: pd.DatetimeIndex
    eligible: np.ndarray


@dataclass(frozen=True)
class ModernSequenceResult:
    prediction: np.ndarray
    info: dict[str, int | float | bool | str]


@dataclass(frozen=True)
class ModernSequenceEnsembleResult:
    mean_prediction: np.ndarray
    prediction_sd: np.ndarray
    seed_predictions: dict[int, np.ndarray]
    seed_info: list[dict[str, int | float | bool | str]]

    @property
    def prediction(self) -> np.ndarray:
        return self.mean_prediction


def build_strict_causal_windows(
    dates: Sequence[object],
    daily_X: np.ndarray | pd.DataFrame,
    raw_y: Sequence[float],
    *,
    target_dates: Sequence[object] | None = None,
    window: int = 14,
    normal: Sequence[bool] | None = None,
    include_target_history: bool = True,
) -> StrictCausalWindows:
    """Build windows containing exactly ``d-window`` through ``d-1`` for target day ``d``.

    The current target day's features and target are never placed in a window.  Historical raw
    targets may be appended as the final channel because they are available at one-step-ahead
    inference time.  Eligibility requires an unbroken daily history, finite historical targets
    when used, and normal operation across the history and target day when ``normal`` is supplied.
    Feature NaNs are retained for training-fold-only median imputation.
    """

    if window < 1:
        raise ValueError("window must be positive")
    date_index = pd.DatetimeIndex(pd.to_datetime(list(dates), errors="raise")).normalize()
    if not len(date_index):
        raise ValueError("dates cannot be empty")
    if date_index.has_duplicates or not date_index.is_monotonic_increasing:
        raise ValueError("dates must be unique and strictly increasing")

    features = np.asarray(daily_X, dtype=float)
    targets = np.asarray(raw_y, dtype=float)
    if features.ndim != 2 or features.shape[0] != len(date_index):
        raise ValueError("daily_X must be two-dimensional and aligned with dates")
    if targets.ndim != 1 or len(targets) != len(date_index):
        raise ValueError("raw_y must be one-dimensional and aligned with dates")
    if np.isinf(features).any() or np.isinf(targets).any():
        raise ValueError("daily_X and raw_y cannot contain infinite values")

    if normal is None:
        normal_mask = np.ones(len(date_index), dtype=bool)
    else:
        normal_mask = pd.Series(normal, dtype="boolean").fillna(False).to_numpy(dtype=bool)
        if normal_mask.ndim != 1 or len(normal_mask) != len(date_index):
            raise ValueError("normal must be one-dimensional and aligned with dates")

    requested = (
        date_index
        if target_dates is None
        else pd.DatetimeIndex(pd.to_datetime(list(target_dates), errors="raise")).normalize()
    )
    if requested.has_duplicates or not requested.is_monotonic_increasing:
        raise ValueError("target_dates must be unique and strictly increasing")

    channels = features.shape[1] + int(include_target_history)
    sequences = np.full((len(requested), window, channels), np.nan, dtype=float)
    eligible = np.zeros(len(requested), dtype=bool)
    positions = {date: row for row, date in enumerate(date_index)}

    for output_row, target_date in enumerate(requested):
        history_dates = pd.date_range(
            target_date - pd.Timedelta(days=window),
            target_date - pd.Timedelta(days=1),
            freq="D",
        )
        history_positions = [positions.get(date) for date in history_dates]
        target_position = positions.get(target_date)
        if target_position is None or any(row is None for row in history_positions):
            continue
        history_rows = np.asarray(history_positions, dtype=int)
        sequences[output_row, :, : features.shape[1]] = features[history_rows]
        if include_target_history:
            sequences[output_row, :, -1] = targets[history_rows]
        finite_history = bool(np.isfinite(targets[history_rows]).all())
        normal_history = bool(normal_mask[target_position] and normal_mask[history_rows].all())
        eligible[output_row] = normal_history and (finite_history or not include_target_history)

    return StrictCausalWindows(
        sequences=sequences,
        target_dates=requested,
        eligible=eligible,
    )


@dataclass(frozen=True)
class _ArrayTransform:
    median: np.ndarray
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, values: np.ndarray) -> _ArrayTransform:
        axes = tuple(range(values.ndim - 1))
        observed = np.isfinite(values).sum(axis=axes)
        if bool((observed == 0).any()):
            missing = np.flatnonzero(observed == 0).tolist()
            raise ValueError(f"Training features are entirely missing in channels: {missing}")
        median = np.nanmedian(values, axis=axes)
        filled = np.where(np.isnan(values), median, values)
        mean = filled.mean(axis=axes)
        scale = filled.std(axis=axes)
        return cls(median, mean, np.where(scale > 0.0, scale, 1.0))

    def transform(self, values: np.ndarray) -> np.ndarray:
        if np.isinf(values).any():
            raise ValueError("Features cannot contain infinite values")
        filled = np.where(np.isnan(values), self.median, values)
        return ((filled - self.mean) / self.scale).astype(np.float32, copy=False)


@dataclass(frozen=True)
class _TargetTransform:
    mean: float
    scale: float

    @classmethod
    def fit(cls, values: np.ndarray) -> _TargetTransform:
        mean = float(values.mean())
        scale = float(values.std())
        return cls(mean, scale if scale > 0.0 else 1.0)

    def transform(self, values: np.ndarray) -> np.ndarray:
        return ((values - self.mean) / self.scale).astype(np.float32, copy=False)

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        return np.asarray(values, dtype=float) * self.scale + self.mean


class _CausalConv1d(nn.Conv1d):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        left_padding = self.dilation[0] * (self.kernel_size[0] - 1)
        return super().forward(F.pad(inputs, (left_padding, 0)))


class _TCNRegressor(nn.Module):
    def __init__(self, channels: int, hidden: int, dropout: float) -> None:
        super().__init__()
        self.input_projection = nn.Conv1d(channels, hidden, kernel_size=1)
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    _CausalConv1d(hidden, hidden, kernel_size=3, dilation=dilation),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    _CausalConv1d(hidden, hidden, kernel_size=3, dilation=dilation),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                )
                for dilation in (1, 2)
            ]
        )
        self.output = nn.Linear(hidden, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = self.input_projection(inputs.transpose(1, 2))
        for block in self.blocks:
            hidden = hidden + block(hidden)
        return self.output(hidden[:, :, -1]).squeeze(-1)


class _PatchTSTLikeRegressor(nn.Module):
    """Compact channel-independent patch Transformer, not official PatchTST."""

    def __init__(
        self,
        channels: int,
        window: int,
        hidden: int,
        dropout: float,
        patch_length: int,
        patch_stride: int,
        layers: int,
        heads: int,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.patch_length = min(patch_length, window)
        self.patch_stride = patch_stride
        patch_count = 1 + max(0, (window - self.patch_length) // patch_stride)
        self.patch_projection = nn.Linear(self.patch_length, hidden)
        self.position = nn.Parameter(torch.zeros(1, patch_count, hidden))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=heads,
            dim_feedforward=max(2 * hidden, 16),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.output = nn.Sequential(
            nn.LayerNorm(channels * hidden), nn.Linear(channels * hidden, 1)
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch, _, channels = inputs.shape
        by_channel = inputs.transpose(1, 2)
        patches = by_channel.unfold(2, self.patch_length, self.patch_stride)
        tokens = self.patch_projection(patches).reshape(batch * channels, patches.shape[2], -1)
        encoded = self.encoder(tokens + self.position[:, : tokens.shape[1]])
        representation = encoded[:, -1].reshape(batch, channels * encoded.shape[-1])
        return self.output(representation).squeeze(-1)


class _NBeatsLikeBlock(nn.Module):
    def __init__(self, input_size: int, hidden: int, dropout: float) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(input_size, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.backcast = nn.Linear(hidden, input_size)
        self.forecast = nn.Linear(hidden, 1)

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.body(inputs)
        return self.backcast(hidden), self.forecast(hidden).squeeze(-1)


class _NBeatsLikeRegressor(nn.Module):
    """Residual backcast/forecast stack inspired by N-BEATS, not official N-BEATS."""

    def __init__(self, input_size: int, hidden: int, dropout: float, blocks: int) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [_NBeatsLikeBlock(input_size, hidden, dropout) for _ in range(blocks)]
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        residual = inputs.flatten(start_dim=1)
        forecast = torch.zeros(inputs.shape[0], device=inputs.device, dtype=inputs.dtype)
        for block in self.blocks:
            backcast, increment = block(residual)
            residual = residual - backcast
            forecast = forecast + increment
        return forecast


def _normalise_architecture(architecture: str) -> ModernArchitecture:
    value = architecture.upper().replace("-", "_")
    aliases = {"PATCHTST": "PATCHTST_LIKE", "NBEATS": "NBEATS_LIKE"}
    value = aliases.get(value, value)
    if value not in {"TCN", "PATCHTST_LIKE", "NBEATS_LIKE"}:
        raise ValueError(f"Unsupported modern sequence architecture: {architecture}")
    return value  # type: ignore[return-value]


def _validate_config(config: ModernSequenceConfig) -> None:
    if not config.seeds or len(config.seeds) != len(set(config.seeds)):
        raise ValueError("seeds must be non-empty and unique")
    if config.max_epochs < 1 or config.patience < 1 or config.batch_size < 1:
        raise ValueError("max_epochs, patience, and batch_size must be positive")
    if not 0.0 < config.validation_fraction < 0.5:
        raise ValueError("validation_fraction must lie strictly between 0 and 0.5")
    positive = (
        config.hidden_size,
        config.patch_length,
        config.patch_stride,
        config.transformer_layers,
        config.transformer_heads,
        config.nbeats_blocks,
    )
    if any(value < 1 for value in positive):
        raise ValueError("architecture sizes must be positive")
    if config.hidden_size % config.transformer_heads:
        raise ValueError("hidden_size must be divisible by transformer_heads")


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def _make_model(
    architecture: ModernArchitecture,
    sequence_shape: tuple[int, int],
    config: ModernSequenceConfig,
) -> nn.Module:
    window, channels = sequence_shape
    if architecture == "TCN":
        return _TCNRegressor(channels, config.hidden_size, config.dropout)
    if architecture == "PATCHTST_LIKE":
        return _PatchTSTLikeRegressor(
            channels,
            window,
            config.hidden_size,
            config.dropout,
            config.patch_length,
            config.patch_stride,
            config.transformer_layers,
            config.transformer_heads,
        )
    return _NBeatsLikeRegressor(
        window * channels,
        config.hidden_size,
        config.dropout,
        config.nbeats_blocks,
    )


def _train_epochs(
    model: nn.Module,
    train_X: torch.Tensor,
    train_y: torch.Tensor,
    *,
    epochs: int,
    config: ModernSequenceConfig,
    seed: int,
    validation: tuple[torch.Tensor, torch.Tensor] | None,
) -> tuple[nn.Module, int, float]:
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    loss_function = nn.MSELoss()
    generator = torch.Generator(device="cpu").manual_seed(seed)
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 1
    best_loss = float("inf")
    stale = 0

    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(len(train_X), generator=generator)
        for start in range(0, len(order), config.batch_size):
            rows = order[start : start + config.batch_size]
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(model(train_X[rows]), train_y[rows])
            loss.backward()
            optimizer.step()

        if validation is None:
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            continue

        model.eval()
        with torch.no_grad():
            validation_loss = float(loss_function(model(validation[0]), validation[1]).item())
        if validation_loss < best_loss - config.minimum_improvement:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= config.patience:
                break

    model.load_state_dict(best_state)
    return model, best_epoch, best_loss


def fit_predict_modern_sequence(
    train_X: np.ndarray,
    train_y: Sequence[float],
    predict_X: np.ndarray,
    architecture: str,
    *,
    seed: int = 11,
    config: ModernSequenceConfig | None = None,
) -> ModernSequenceResult:
    """Fit one modern sequence baseline and predict without consulting prediction outcomes."""

    settings = config or ModernSequenceConfig()
    _validate_config(settings)
    model_name = _normalise_architecture(architecture)
    features = np.asarray(train_X, dtype=float)
    predictions = np.asarray(predict_X, dtype=float)
    targets = np.asarray(train_y, dtype=float)
    if features.ndim != 3 or predictions.ndim != 3:
        raise ValueError("train_X and predict_X must have shape (rows, window, channels)")
    if features.shape[1:] != predictions.shape[1:]:
        raise ValueError("train_X and predict_X must share window and channel dimensions")
    if targets.ndim != 1 or len(targets) != len(features):
        raise ValueError("train_y must be one-dimensional and aligned with train_X")
    if len(features) < 8:
        raise ValueError("At least eight training rows are required")
    if not np.isfinite(targets).all():
        raise ValueError("train_y must be finite")

    validation_rows = max(1, int(np.ceil(len(features) * settings.validation_fraction)))
    fit_rows = len(features) - validation_rows
    if fit_rows < 4:
        raise ValueError("Chronological early-stopping split leaves fewer than four fit rows")

    _seed_everything(seed)
    selection_feature_transform = _ArrayTransform.fit(features[:fit_rows])
    selection_target_transform = _TargetTransform.fit(targets[:fit_rows])
    selection_train_X = torch.from_numpy(selection_feature_transform.transform(features[:fit_rows]))
    selection_train_y = torch.from_numpy(selection_target_transform.transform(targets[:fit_rows]))
    validation_X = torch.from_numpy(selection_feature_transform.transform(features[fit_rows:]))
    validation_y = torch.from_numpy(selection_target_transform.transform(targets[fit_rows:]))
    model = _make_model(model_name, features.shape[1:], settings)
    model, best_epoch, best_validation_loss = _train_epochs(
        model,
        selection_train_X,
        selection_train_y,
        epochs=settings.max_epochs,
        config=settings,
        seed=seed,
        validation=(validation_X, validation_y),
    )

    final_feature_transform = selection_feature_transform
    final_target_transform = selection_target_transform
    preprocessing_rows = fit_rows
    if settings.refit_full_training:
        _seed_everything(seed)
        final_feature_transform = _ArrayTransform.fit(features)
        final_target_transform = _TargetTransform.fit(targets)
        full_X = torch.from_numpy(final_feature_transform.transform(features))
        full_y = torch.from_numpy(final_target_transform.transform(targets))
        model = _make_model(model_name, features.shape[1:], settings)
        model, _, _ = _train_epochs(
            model,
            full_X,
            full_y,
            epochs=best_epoch,
            config=settings,
            seed=seed,
            validation=None,
        )
        preprocessing_rows = len(features)

    model.eval()
    with torch.no_grad():
        scaled_prediction = model(
            torch.from_numpy(final_feature_transform.transform(predictions))
        ).numpy()
    prediction = final_target_transform.inverse_transform(scaled_prediction)
    implementation = {
        "TCN": "local lightweight causal-convolution TCN baseline",
        "PATCHTST_LIKE": "local PatchTST-inspired baseline; not official PatchTST or TimePFN",
        "NBEATS_LIKE": "local N-BEATS-inspired baseline; not official N-BEATS",
    }[model_name]
    return ModernSequenceResult(
        prediction=prediction,
        info={
            "architecture": model_name,
            "implementation_scope": implementation,
            "seed": seed,
            "device": "cpu",
            "training_rows": len(features),
            "early_stopping_fit_rows": fit_rows,
            "early_stopping_validation_rows": validation_rows,
            "prediction_rows_used_for_selection": 0,
            "best_epoch": best_epoch,
            "best_scaled_validation_mse": best_validation_loss,
            "refit_full_training": settings.refit_full_training,
            "preprocessing_fit_rows_final": preprocessing_rows,
        },
    )


def fit_predict_modern_sequence_ensemble(
    train_X: np.ndarray,
    train_y: Sequence[float],
    predict_X: np.ndarray,
    architecture: str,
    *,
    config: ModernSequenceConfig | None = None,
) -> ModernSequenceEnsembleResult:
    """Average independently seeded fits using the pre-registered configuration seeds."""

    settings = config or ModernSequenceConfig()
    _validate_config(settings)
    results = {
        seed: fit_predict_modern_sequence(
            train_X,
            train_y,
            predict_X,
            architecture,
            seed=seed,
            config=settings,
        )
        for seed in settings.seeds
    }
    stacked = np.vstack([result.prediction for result in results.values()])
    return ModernSequenceEnsembleResult(
        mean_prediction=stacked.mean(axis=0),
        prediction_sd=stacked.std(axis=0, ddof=0),
        seed_predictions={seed: result.prediction for seed, result in results.items()},
        seed_info=[result.info for result in results.values()],
    )
