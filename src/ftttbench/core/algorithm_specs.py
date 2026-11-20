"""Static, versioned research algorithm descriptions and Agent admission.

Registry consumers use explicit physical variants, never naming heuristics.
Static schemas contain no case data or deployment-specific paths.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import json

from pydantic import ConfigDict, create_model

from ftttbench.core.schema import AlgorithmConfig


@dataclass(frozen=True)
class Variant:
    id: str
    model: str
    selectors: dict = field(default_factory=dict)
    observation_domains: tuple[str, ...] = ("features",)
    required_observations: tuple[str, ...] = ("delta_tof_s",)
    observation_alternatives: tuple[tuple[str, ...], ...] = ()
    runtime_requirements: tuple[str, ...] = ("numpy", "scipy")
    limitations: tuple[str, ...] = ()
    iteration_unit: str = "iteration"
    deterministic: str = (
        "deterministic_for_fixed_input_config_and_numerical_environment"
    )
    agent_parameters: bool = True


@dataclass(frozen=True)
class AlgorithmSpecification:
    algorithm_id: str
    family: str
    description: str
    variants: tuple[Variant, ...]
    supported_geometries: tuple[str, ...] = ("ring", "linear", "custom")

    def variant(self, variant_id=None):
        if variant_id is None:
            return self.variants[0]
        for variant in self.variants:
            if variant.id == variant_id:
                return variant
        raise ValueError(f"unknown variant {variant_id!r} for {self.algorithm_id}")


def agent_schema(model, selectors=None, *, enabled=True):
    """Build the schema from exposed fields only, including only their definitions."""
    fields = {
        name: (info.annotation, info)
        for name, info in model.model_fields.items()
        if enabled and (info.json_schema_extra or {}).get("exposure") == "agent"
    }
    schema = create_model(
        f"{model.__name__}Agent",
        __config__=ConfigDict(extra="forbid"),
        **fields,
    ).model_json_schema()
    for key, value in schema["properties"].items():
        value.update(getattr(model, "agent_schema_overrides", {}).get(key, {}))
        if key in (selectors or {}):
            value["const"] = selectors[key]
            value["default"] = selectors[key]
    return schema, {
        key: value["default"]
        for key, value in schema["properties"].items()
        if "default" in value
    }


SPECS = {
    "fttt": AlgorithmSpecification(
        "fttt",
        "eikonal",
        "First-arrival travel-time DIP, INR and pretrained-prior DPS baselines.",
        tuple(
            Variant(
                method,
                "Eikonal_first_arrival_" + method,
                selectors={"variant": method},
                required_observations=(),
                observation_alternatives=(("tof_s",), ("delta_tof_s",)),
                runtime_requirements=("torch>=2.0", "numpy", "scipy")
                + (
                    ("compatible_usct.fttt.prior.v1_checkpoint",)
                    if method == "dps"
                    else ()
                ),
                iteration_unit=(
                    "reverse sampling update"
                    if method == "dps"
                    else "Adam parameter update"
                ),
                deterministic="fixed_seed_input_config_device_and_torch_environment",
                limitations=(
                    "First-arrival geometrical acoustics; Eikonal physics runs on CPU.",
                    "Discrete derivatives are local to the active fast-marching stencil.",
                    "Requires declared first-arrival semantics; correlation/phase delays are not interchangeable.",
                )
                + (
                    (
                        "Requires a trained compatible prior or dataset manifest; missing weights guide to Yang Song score_sde_pytorch.",
                        "Sampling residuals need not decrease monotonically; objective-plateau stopping is unsupported.",
                    )
                    if method == "dps"
                    else ()
                ),
            )
            for method in ("dip", "inr", "dps")
        ),
    )
}

for _variant in SPECS["fttt"].variants:
    SPECS[_variant.id] = replace(
        SPECS["fttt"],
        algorithm_id=_variant.id,
        description=f"First-arrival travel-time {_variant.id.upper()} reconstruction.",
        variants=(_variant,),
    )


def make_agent_config(
    algorithm_id,
    parameters=None,
    *,
    variant=None,
    run_controls=None,
    trusted_parameters=None,
):
    """Only declared Agent knobs plus independent host-supplied context."""
    from ftttbench.core.registry import get_algorithm_entry

    entry = get_algorithm_entry(algorithm_id)
    selected = entry.specification.variant(variant)
    description = entry.describe(selected.id)
    supplied = dict(parameters or {})
    forbidden = supplied.keys() - set(description["allowed_parameters"])
    if forbidden:
        raise ValueError(f"parameters not exposed to Agent: {sorted(forbidden)}")
    for key, value in supplied.items():
        choices = description["config_schema"]["properties"][key].get("enum")
        if choices is not None and value not in choices:
            raise ValueError(f"parameter {key} must be one of {choices}")
    values = {**(trusted_parameters or {}), **supplied}
    for key, value in selected.selectors.items():
        if key in values and values[key] != value:
            raise ValueError(f"parameter {key} conflicts with selected variant")
        values[key] = value
    controls = dict(run_controls or {})
    if controls.keys() - set(entry.supported_budgets):
        raise ValueError("unsupported Agent run control")
    model = entry.parameter_model(values)
    typed = model.model_validate_json(json.dumps(values, allow_nan=False), strict=True)
    return AlgorithmConfig(name=algorithm_id, parameters=typed, run_controls=controls)


def case_capabilities(
    algorithm_id,
    case,
    *,
    variant=None,
    runtime_available=None,
):
    """Case facts are separate from static specifications and clinical validity."""
    selected = SPECS[algorithm_id].variant(variant)
    return {
        "schema_version": "fttt.case_capabilities.v1",
        "algorithm_id": algorithm_id,
        "variant": selected.id,
        "case_id": case.case_id,
        "observable": case.measurement.observable,
        "observation_mode": "differential"
        if case.measurement.delta_tof_s is not None
        else "absolute",
        "num_valid_channels": int(case.measurement.valid_mask.sum()),
        "reference_sound_speed_mps": case.metadata.get("reference_sound_speed_mps"),
        "reference_tof_definition": case.metadata.get("reference_tof_definition"),
        "runtime_available": runtime_available,
        "runtime_availability_source": (
            "not_checked" if runtime_available is None else "deployment_reported"
        ),
        "validation_is_independent_test": False,
    }
