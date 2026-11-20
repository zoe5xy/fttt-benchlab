"""Authoritative native parameter models; constraints are numerical, not presets.

Null defaults denote case-dependent values and are omitted at the solver boundary.
Budgets and evaluation policies are validated separately in ``configuration``.
"""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, model_validator

Positive = Annotated[float, Field(strict=True, gt=0)]
Nonnegative = Annotated[float, Field(strict=True, ge=0)]
PositiveInt = Annotated[int, Field(strict=True, gt=0)]
NonnegativeInt = Annotated[int, Field(strict=True, ge=0)]
Bounds = tuple[Positive, Positive]


def parameter(default, description, units="1", exposure="advanced", **kwargs):
    return Field(
        default=default,
        description=description,
        json_schema_extra={"units": units, "exposure": exposure},
        **kwargs,
    )


class Parameters(BaseModel):
    model_config = ConfigDict(
        extra="forbid", allow_inf_nan=False, validate_default=True
    )

    @model_validator(mode="after")
    def ordered_bounds(self):
        bounds = getattr(self, "sound_speed_bounds_mps", None)
        if bounds is not None and bounds[0] >= bounds[1]:
            raise ValueError("sound_speed_bounds_mps requires positive low < high")
        return self


class SoundSpeedParameters(Parameters):
    reference_sound_speed_mps: Positive | None = parameter(
        None,
        "Measurement reference; case metadata, otherwise 1500 m/s.",
        "m/s",
        "internal",
    )
    sound_speed_bounds_mps: Bounds = parameter(
        (1300.0, 1700.0),
        "Ordered numerical projection bounds, not a calibrated physiological range.",
        "m/s",
        "agent",
    )
    roi_update_only: bool = parameter(
        False, "Restrict updates to the supplied ROI, if present."
    )
