"""Only standalone first-arrival algorithms are registered here."""

from __future__ import annotations
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol
from .schema import AlgorithmConfig, ReconstructionResult, FTTTCase
from .algorithm_specs import AlgorithmSpecification, SPECS, agent_schema
from .run_controls import BUDGET_FIELDS, RunControls
from ftttbench.algorithms.fttt import FTTTAlgorithm
from ftttbench.algorithms.fttt.parameters import (
    DIPParameters,
    INRParameters,
    DPSParameters,
)


class Algorithm(Protocol):
    name: str

    def run(self, case: FTTTCase, config: AlgorithmConfig) -> ReconstructionResult:
        """Run one reconstruction from prepared observations and configuration."""


@dataclass(frozen=True)
class AlgorithmEntry:
    name: str
    factory: Callable[[], Algorithm]
    specification: AlgorithmSpecification
    models: dict
    iteration_defaults: dict
    supported_budgets: tuple = BUDGET_FIELDS
    numerical_stopping: bool = True
    deadline_semantics: str = (
        "cooperative; launching process must enforce a hard deadline"
    )
    budget_descriptions: dict[str, str] = field(default_factory=dict)

    @property
    def variants(self):
        return self.specification.variants

    def parameter_model(self, parameters=None):
        selected = (
            parameters.get("variant")
            if isinstance(parameters, dict)
            else getattr(parameters, "variant", None)
        )
        variant = self.specification.variant(selected)
        return self.models[variant.id]

    def default_iterations(self, parameters=None, case=None):
        selected = (
            parameters.get("variant")
            if isinstance(parameters, dict)
            else getattr(parameters, "variant", None)
        )
        rule = self.iteration_defaults[self.specification.variant(selected).id]
        if rule == "sampling_steps":
            return (
                parameters.get("sampling_steps", 64)
                if isinstance(parameters, dict)
                else getattr(parameters, "sampling_steps", 64)
            )
        return rule

    def validate_controls(self, controls):
        unsupported = {
            key
            for key in BUDGET_FIELDS
            if key not in self.supported_budgets and getattr(controls, key) is not None
        }
        if not self.numerical_stopping:
            unsupported.update(
                controls.model_dump(exclude_defaults=True).keys() - set(BUDGET_FIELDS)
            )
        if unsupported:
            raise ValueError(
                f"{self.name} does not support run controls: {sorted(unsupported)}"
            )

    def budget_schema(self):
        schema = RunControls.model_json_schema()
        schema["properties"] = {
            key: value
            for key, value in schema["properties"].items()
            if key in self.supported_budgets
        }
        for key, description in self.budget_descriptions.items():
            schema["properties"][key]["description"] = description
        return schema

    @property
    def description(self):
        return self.specification.description

    @property
    def tags(self):
        return (self.specification.family,)

    def describe(self, variant: str | None = None) -> dict:
        spec = self.specification
        selected = spec.variant(variant)
        model = self.models[selected.id]
        schema, defaults = agent_schema(
            model, selected.selectors, enabled=selected.agent_parameters
        )
        return {
            "schema_version": "fttt.algorithm.v1",
            "algorithm_id": self.name,
            "family": spec.family,
            "variant": selected.id,
            "variants": [
                {"id": item.id, "model": item.model} for item in self.variants
            ],
            "description": self.description,
            "mathematical_model": selected.model,
            "required_observation_domains": list(selected.observation_domains),
            "required_observations": {
                "all_of": list(selected.required_observations),
                "any_of": [list(group) for group in selected.observation_alternatives],
            },
            "supported_geometries": list(spec.supported_geometries),
            "parameter_model": model.__name__,
            "config_schema": schema,
            "allowed_parameters": sorted(schema["properties"]),
            "default_parameters": defaults,
            "compute_budget": {
                "separate_from_parameters": True,
                "default_max_iterations": self.default_iterations(selected.selectors),
                "online_controls_supported": bool(self.supported_budgets),
                "numerical_convergence_stopping_supported": self.numerical_stopping,
                "run_controls_schema": self.budget_schema(),
                "host_limits_schema": self.budget_schema(),
                "deadline_semantics": self.deadline_semantics,
            },
            "runtime_requirements": list(selected.runtime_requirements),
            "runtime_availability": "not_checked_static_description",
            "limitations": [
                "Research-grade within tested models/ranges; not clinical validation.",
                "Geometry acceptance does not establish imaging performance for every layout.",
                *selected.limitations,
            ],
            "iteration_unit": selected.iteration_unit,
            "deterministic_status": selected.deterministic,
        }


_FAMILY_REGISTRY = {
    "fttt": AlgorithmEntry(
        "fttt",
        FTTTAlgorithm,
        SPECS["fttt"],
        {"dip": DIPParameters, "inr": INRParameters, "dps": DPSParameters},
        {"dip": 1000, "inr": 1000, "dps": "sampling_steps"},
    )
}

# FTTT is the shared family implementation, not an additional method.
_REGISTRY = {}
for _name, _model, _iterations in (
    ("dip", DIPParameters, 1000),
    ("inr", INRParameters, 1000),
    ("dps", DPSParameters, "sampling_steps"),
):
    _REGISTRY[_name] = AlgorithmEntry(
        _name,
        lambda name=_name: FTTTAlgorithm(name),
        SPECS[_name],
        {_name: _model},
        {_name: _iterations},
    )


def get_algorithm_entry(name: str) -> AlgorithmEntry:
    if name == "fttt":
        return _FAMILY_REGISTRY[name]
    try:
        return _REGISTRY[name]
    except KeyError as exc:
        raise KeyError(
            f"unknown algorithm {name!r}; available: {', '.join(sorted(_REGISTRY))}"
        ) from exc


def get_algorithm(name: str) -> Algorithm:
    return get_algorithm_entry(name).factory()


def list_algorithms() -> list[AlgorithmEntry]:
    return list(_REGISTRY.values())
