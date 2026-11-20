"""Reference medium, speed bounds and confidence weights for FTTT."""

from __future__ import annotations
import numpy as np
from ftttbench.core.schema import AlgorithmConfig, FTTTCase


def reference_sound_speed(case: FTTTCase, config: AlgorithmConfig) -> float:
    value = config.parameters.reference_sound_speed_mps
    if value is None:
        value = case.metadata.get("reference_sound_speed_mps", 1500.0)
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError("reference_sound_speed_mps must be positive")
    return value


def speed_bounds(config: AlgorithmConfig) -> tuple[float, float]:
    bounds = config.parameters.sound_speed_bounds_mps
    low, high = float(bounds[0]), float(bounds[1])
    if not np.isfinite(low) or not np.isfinite(high) or low <= 0 or high <= low:
        raise ValueError("sound_speed_bounds_mps must be [positive_low, higher_high]")
    return low, high


def configured_ray_weights(
    case: FTTTCase,
    projector,
    mask: np.ndarray | None,
    config: AlgorithmConfig,
) -> np.ndarray:
    """Return per-ray confidence weights after optional config shaping."""

    threshold = float(config.parameters.min_ray_weight)
    power = float(config.parameters.ray_weight_power)
    return ray_weights(case, projector, mask, min_weight=threshold, weight_power=power)


def ray_weights(
    case: FTTTCase,
    projector,
    mask: np.ndarray | None = None,
    *,
    min_weight: float = 0.0,
    weight_power: float = 1.0,
) -> np.ndarray:
    """Return per-ray confidence weights clipped to [0, 1].

    ``min_weight`` and ``weight_power`` let algorithms sharpen feature-derived
    confidence without editing the feature case. They default to the historical
    behavior: no threshold and no exponentiation.
    """

    if min_weight < 0.0 or min_weight > 1.0:
        raise ValueError("min_ray_weight/ray_weight_threshold must be in [0, 1]")
    if weight_power <= 0.0:
        raise ValueError("ray_weight_power must be positive")

    source = case.measurement.ray_weights
    if source is None:
        weights = np.ones(projector.n_rays, dtype=float)
    else:
        weights = np.asarray(source, dtype=float).reshape(-1)
        if weights.size != projector.n_rays:
            raise ValueError(
                "measurement.ray_weights shape must match transmitter/receiver ray shape"
            )
        weights = np.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
        weights = np.clip(weights, 0.0, 1.0)
    if min_weight > 0.0:
        weights = np.where(weights >= float(min_weight), weights, 0.0)
    if weight_power != 1.0:
        weights = np.power(weights, float(weight_power))
    if mask is not None:
        weights = np.where(np.asarray(mask, dtype=bool), weights, 0.0)
    return weights
