from __future__ import annotations
from typing import Any, Literal
from enum import StrEnum
import numpy as np
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SerializeAsAny,
    field_validator,
    model_validator,
)

from .run_controls import RunControls
from .execution import Deployment, EvaluationSettings, ImageEvaluationSettings
from ..algorithms.parameters import Parameters


class ResultStatus(StrEnum):
    COMPLETED = "completed"
    BUDGET_EXHAUSTED = "budget_exhausted"
    FAILED = "failed"


def as_float_array(value):
    """Convert input data to a real-valued NumPy array."""
    if value is None:
        return None
    array = np.asarray(value)
    if np.iscomplexobj(array):
        raise ValueError("FTTT data must be real-valued")
    return array.astype(np.float64)


def as_bool_array(value):
    """Accept only boolean values or binary values (0 and 1)."""
    if value is None:
        return None
    array = np.asarray(value)
    if not np.isin(array, [0, 1]).all():
        raise ValueError("Masks must contain only boolean values or 0/1")
    return array.astype(bool)


class ArrayModel(BaseModel):
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
    )


class GridSpec(ArrayModel):
    # Axis order: (ny, nx), (dy, dx), and (y0, x0).
    shape: tuple[int, int]
    spacing_m: tuple[float, float]
    origin_m: tuple[float, float] = (0.0, 0.0)
    roi_mask: np.ndarray | None = None

    @field_validator("roi_mask", mode="before")
    @classmethod
    def convert_mask(cls, value):
        return as_bool_array(value)

    @model_validator(mode="after")
    def check_grid(self):
        if min(self.shape) < 2:
            raise ValueError(
                "The Eikonal grid requires at least 2 cells along each axis"
            )
        if not np.isfinite(self.spacing_m).all() or min(self.spacing_m) <= 0:
            raise ValueError("spacing_m must contain finite positive values in meters")
        if not np.isfinite(self.origin_m).all():
            raise ValueError("origin_m must contain finite values")
        if self.roi_mask is not None:
            if self.roi_mask.shape != self.shape:
                raise ValueError("roi_mask must have the same shape as grid.shape")
        return self


class GeometrySpec(ArrayModel):
    type: Literal["ring", "linear", "custom"] = "ring"

    # Each row contains one transducer's [y, x] coordinates in meters.
    tx_pos_m: np.ndarray
    rx_pos_m: np.ndarray
    radius_m: float | None = Field(default=None, gt=0, allow_inf_nan=False)

    @field_validator("tx_pos_m", "rx_pos_m", mode="before")
    @classmethod
    def convert_positions(cls, value):
        return as_float_array(value)

    @model_validator(mode="after")
    def check_positions(self):
        for name in ("tx_pos_m", "rx_pos_m"):
            positions = getattr(self, name)
            if positions.ndim != 2 or positions.shape[1] != 2 or len(positions) == 0:
                raise ValueError(f"{name} must have shape (n, 2) with n > 0")
            if not np.isfinite(positions).all():
                raise ValueError(f"{name} must contain only finite values")
        return self


class MeasurementSpec(ArrayModel):
    domain: Literal["features"] = "features"
    observable: Literal["first_arrival"] = "first_arrival"

    # Provide exactly one representation, each with shape (n_tx, n_rx).
    tof_s: np.ndarray | None = None
    delta_tof_s: np.ndarray | None = None

    # True indicates a channel available for subsequent data processing.
    valid_mask: np.ndarray
    # Optional confidence weights for first-arrival picks, in [0, 1].
    ray_weights: np.ndarray | None = None

    @field_validator("tof_s", "delta_tof_s", "ray_weights", mode="before")
    @classmethod
    def convert_times(cls, value):
        return as_float_array(value)

    @field_validator("valid_mask", mode="before")
    @classmethod
    def convert_mask(cls, value):
        return as_bool_array(value)

    @model_validator(mode="after")
    def check_measurement(self):
        if (self.tof_s is None) == (self.delta_tof_s is None):
            raise ValueError("Provide exactly one of tof_s and delta_tof_s")

        times = self.tof_s if self.tof_s is not None else self.delta_tof_s
        if times.ndim != 2:
            raise ValueError("Travel-time arrays must have shape (n_tx, n_rx)")
        if self.valid_mask.shape != times.shape:
            raise ValueError(
                "valid_mask must have the same shape as the travel-time array"
            )
        if not self.valid_mask.any():
            raise ValueError("At least one valid channel is required")
        if not np.isfinite(times[self.valid_mask]).all():
            raise ValueError(
                "Travel times on valid channels must not contain NaN or Inf"
            )
        if self.tof_s is not None:
            if np.any(times[self.valid_mask] < 0):
                raise ValueError("Absolute travel times must be nonnegative")
        if self.ray_weights is not None:
            weights = self.ray_weights
            if weights.shape != times.shape:
                raise ValueError("ray_weights must match the travel-time array")
            if not np.isfinite(weights).all() or np.any((weights < 0) | (weights > 1)):
                raise ValueError(
                    "ray_weights must be finite confidence values in [0, 1]"
                )

        # Differential travel times may be negative.
        # Invalid channels may contain NaN.
        return self


class GroundTruthSpec(ArrayModel):
    sound_speed_mps: np.ndarray | None = None

    @field_validator("sound_speed_mps", mode="before")
    @classmethod
    def convert_speed(cls, value):
        return as_float_array(value)

    @model_validator(mode="after")
    def check_speed(self):
        speed = self.sound_speed_mps
        if speed is not None:
            if not np.isfinite(speed).all() or np.any(speed <= 0):
                raise ValueError(
                    "Ground truth sound speeds must be finite positive values "
                    "in meters per second"
                )
        return self


class FTTTCase(ArrayModel):
    case_id: str = Field(min_length=1)
    grid: GridSpec
    geometry: GeometrySpec
    measurement: MeasurementSpec
    ground_truth: GroundTruthSpec = Field(default_factory=GroundTruthSpec)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def check_case(self):
        expected = (
            len(self.geometry.tx_pos_m),
            len(self.geometry.rx_pos_m),
        )
        if self.measurement.valid_mask.shape != expected:
            raise ValueError(
                "The measurement shape must match the transducer counts; "
                f"expected {expected}"
            )

        truth = self.ground_truth.sound_speed_mps
        if truth is not None and truth.shape != self.grid.shape:
            raise ValueError("The ground truth sound-speed image must match grid.shape")

        c0 = self.metadata.get("reference_sound_speed_mps")
        if c0 is not None:
            c0 = float(c0)
            if not np.isfinite(c0) or c0 <= 0:
                raise ValueError(
                    "The reference sound speed must be finite and positive"
                )
        if self.measurement.delta_tof_s is not None:
            if c0 is None:
                raise ValueError(
                    "Differential travel times require "
                    "metadata['reference_sound_speed_mps']"
                )
            # This version supports only analytical reference travel times
            # in a homogeneous background.
            definition = self.metadata.get("reference_tof_definition")
            if definition != "distance_over_c0":
                raise ValueError(
                    "This version requires reference_tof_definition='distance_over_c0'"
                )
        return self


class AlgorithmConfig(ArrayModel):
    """Scientific parameters, execution limits and evaluation policy."""

    name: str | None = None
    diagnostics: bool = Field(False, strict=True)
    parameters: SerializeAsAny[Parameters] | dict[str, Any] = Field(
        default_factory=dict
    )
    run_controls: RunControls = Field(default_factory=RunControls)
    deployment: Deployment = Field(default_factory=Deployment)
    evaluation: EvaluationSettings = Field(default_factory=EvaluationSettings)
    image_evaluation: ImageEvaluationSettings = Field(
        default_factory=ImageEvaluationSettings
    )
    metadata: dict[str, Any] = Field(default_factory=dict)


class ReconstructionResult(ArrayModel):
    algorithm: str
    case_id: str
    status: ResultStatus = ResultStatus.COMPLETED
    experiment_id: str | None = None
    variant: Literal["dip", "inr", "dps"] | None = None

    sound_speed_mps: np.ndarray | None = None
    runtime_s: float | None = Field(default=0.0, ge=0, allow_inf_nan=False)
    failure_reason: str | None = None
    metrics: dict[str, Any] = Field(default_factory=dict)
    artifacts: dict[str, str] = Field(default_factory=dict)

    @field_validator("sound_speed_mps", mode="before")
    @classmethod
    def convert_speed(cls, value):
        return as_float_array(value)

    @model_validator(mode="after")
    def check_result(self):
        reason = self.metrics.get("stop_reason")
        if self.status == ResultStatus.COMPLETED:
            if reason in {
                "max_iterations",
                "time_budget",
                "forward_calls_budget",
                "adjoint_calls_budget",
            }:
                self.status = ResultStatus.BUDGET_EXHAUSTED
            elif reason in {
                "numerical_failure",
                "linear_solver_breakdown",
                "line_search_failed",
            }:
                self.status = ResultStatus.FAILED
                self.failure_reason = (
                    self.failure_reason or self.metrics.get("numerical_error") or reason
                )
        speed = self.sound_speed_mps
        if self.status == "failed":
            if not self.failure_reason:
                raise ValueError("Failed results must provide failure_reason")
        elif speed is None:
            raise ValueError(
                "Non-failed results must provide a reconstructed sound-speed image"
            )

        if speed is not None:
            if speed.ndim != 2 or not np.isfinite(speed).all() or np.any(speed <= 0):
                raise ValueError(
                    "The reconstructed sound-speed image must be a 2-D array "
                    "of finite positive values"
                )
        return self
