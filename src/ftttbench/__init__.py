"""Independent first-arrival travel-time framework."""

from .core.schema import (
    AlgorithmConfig,
    FTTTCase,
    GridSpec,
    GeometrySpec,
    GroundTruthSpec,
    MeasurementSpec,
    ReconstructionResult,
)
from .core.run_controls import RunControls
from .core.config import prepare_config
from .core.registry import get_algorithm, get_algorithm_entry, list_algorithms
from .benchmark.runner import run, suite, evaluate
from .core.io import read_case_hdf5, write_case_hdf5


__version__ = "0.1.0"


def describe(algorithm="dip", variant=None):
    return get_algorithm_entry(algorithm).describe(variant)


__all__ = [
    "AlgorithmConfig",
    "FTTTCase",
    "GridSpec",
    "GeometrySpec",
    "GroundTruthSpec",
    "MeasurementSpec",
    "ReconstructionResult",
    "RunControls",
    "prepare_config",
    "get_algorithm",
    "get_algorithm_entry",
    "list_algorithms",
    "run",
    "suite",
    "evaluate",
    "describe",
    "read_case_hdf5",
    "write_case_hdf5",
]
