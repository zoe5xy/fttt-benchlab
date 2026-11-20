"""Admission and persisted failure records for typed and NumPy inputs."""

import json

import numpy as np
import pytest

from ftttbench import MeasurementSpec, prepare_config, run
from ftttbench.data.synthetic import make_first_arrival_case


def read_strict_json(path):
    def reject_nonfinite(value):
        raise ValueError(f"Nonstandard JSON number: {value}")

    return json.loads(path.read_text(), parse_constant=reject_nonfinite)


@pytest.mark.parametrize("invalid", ["residual_scale", "deployment"])
def test_prepared_config_revalidates_mutated_nested_models(invalid):
    config = prepare_config("inr", {"run_controls": {"max_iterations": 0}})
    if invalid == "residual_scale":
        config.parameters = config.parameters.model_copy(
            update={"residual_scale_s": 0.0}
        )
    else:
        config.deployment.neural_device = "mps"

    with pytest.raises(ValueError):
        prepare_config("inr", config)


def test_numpy_reference_speed_can_be_persisted(tmp_path):
    pytest.importorskip("torch")
    case = make_first_arrival_case(shape=(8, 8), n_transducers=4)
    case.metadata["reference_sound_speed_mps"] = np.float32(1500.0)

    result = run(
        "inr",
        case,
        {"run_controls": {"max_iterations": 0}},
        tmp_path,
        score=False,
        preview=False,
    )

    assert result.status == "budget_exhausted", result.failure_reason
    record = read_strict_json(tmp_path / "run.json")
    assert record["status"] == "budget_exhausted"
    assert record["measurement"]["reference_sound_speed_mps"] == 1500.0
    assert (
        read_strict_json(tmp_path / "metrics.json")["stop_reason"] == "max_iterations"
    )


def test_invalid_absolute_reference_speed_returns_persisted_failure(tmp_path):
    pytest.importorskip("torch")
    case = make_first_arrival_case(shape=(8, 8), n_transducers=4)
    distance = np.linalg.norm(
        case.geometry.tx_pos_m[:, None] - case.geometry.rx_pos_m[None], axis=-1
    )
    case.measurement = MeasurementSpec(
        tof_s=case.measurement.delta_tof_s + distance / 1500.0,
        valid_mask=case.measurement.valid_mask,
    )
    case.metadata["reference_sound_speed_mps"] = float("nan")

    result = run(
        "inr",
        case,
        {"run_controls": {"max_iterations": 0}},
        tmp_path,
        score=False,
        preview=False,
    )

    assert result.status == "failed"
    assert result.failure_reason
    record = read_strict_json(tmp_path / "run.json")
    assert record["status"] == "failed"
    assert record["failure_reason"] == result.failure_reason
    read_strict_json(tmp_path / "metrics.json")
