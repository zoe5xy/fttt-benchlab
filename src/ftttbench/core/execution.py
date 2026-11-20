from typing import Annotated, Literal
from pydantic import BaseModel, ConfigDict, Field


class Deployment(BaseModel):
    model_config = ConfigDict(extra="forbid")
    scratch_root: str | None = None
    neural_device: str = Field("cpu", pattern=r"^(cpu|cuda(:[0-9]+)?)$")


class EvaluationSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    receiver_indices: list[Annotated[int, Field(ge=0, strict=True)]] | None = None
    frequency_indices: list[Annotated[int, Field(ge=0, strict=True)]] | None = None
    receiver_fraction: float = Field(0, ge=0, lt=1)
    frequency_fraction: float = Field(0, ge=0, lt=1)
    seed: int = Field(0, ge=0, strict=True)
    exclude_reciprocal: bool = True


class ImageEvaluationSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    primary_region: Literal["tissue", "full_image"] = "tissue"
    water_speed_mps: float = Field(1500, gt=0)
    water_tolerance_mps: float = Field(0.1, ge=0)
    data_range_mps: float | None = Field(None, gt=0)
