"""Public workflows: first-arrival contracts, persisted runs and portable configs."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest
import yaml

from ftttbench import (
    evaluate,
    get_algorithm,
    list_algorithms,
    prepare_config,
    read_case_hdf5,
    run,
    suite,
    write_case_hdf5,
)
from ftttbench.core.io import read_result_hdf5
from ftttbench.core.schema import FTTTCase, GroundTruthSpec, MeasurementSpec
from ftttbench.data.synthetic import make_grid, make_ring_geometry
from ftttbench.operators.eikonal import EikonalForward


REPOSITORY = Path(__file__).resolve().parents[1]


def first_arrival_case(shape=(8, 8), *, missing_channels=False):
    grid = make_grid(shape)
    geometry = make_ring_geometry(4, radius_m=0.75 * max(shape) * 1e-3)
    speed = np.full(shape, 1500.0)
    speed[2:5, 2:5] = 1450.0
    distance = np.linalg.norm(
        geometry.tx_pos_m[:, None] - geometry.rx_pos_m[None], axis=-1
    )
    times = EikonalForward(grid, geometry).forward(1 / speed).reshape(4, 4)
    delta = times - distance / 1500.0
    valid_mask = distance > 0
    if missing_channels:
        delta[~valid_mask] = np.nan
    return FTTTCase(
        case_id="workflow_case",
        grid=grid,
        geometry=geometry,
        measurement=MeasurementSpec(
            delta_tof_s=delta,
            valid_mask=valid_mask,
            ray_weights=np.ones_like(delta),
        ),
        ground_truth=GroundTruthSpec(sound_speed_mps=speed),
        metadata={
            "reference_sound_speed_mps": 1500.0,
            "reference_tof_definition": "distance_over_c0",
            "forward_model": "eikonal",
            "measurement_provenance": "synthetic",
        },
    )


def small_config(algorithm):
    parameters = {
        "inr": {"hidden_features": 8, "hidden_layers": 1},
        "dip": {"base_channels": 4, "input_channels": 2},
        "dps": {"sampling_steps": 2},
    }[algorithm]
    return {
        "name": algorithm,
        "parameters": parameters,
        "run_controls": {"max_iterations": 2},
        "deployment": {"neural_device": "cpu"},
    }


def test_public_algorithms_and_saved_inr_run(tmp_path):
    assert {entry.name for entry in list_algorithms()} == {"dip", "inr", "dps"}
    pytest.importorskip("torch")
    case = first_arrival_case()
    out = tmp_path / "inr"
    result = run("inr", case, small_config("inr"), out, preview=False)
    assert result.status != "failed", result.failure_reason
    assert result.metrics["stopping"]["completed_iterations"] == 2
    assert result.sound_speed_mps.shape == case.grid.shape
    assert np.isfinite(result.sound_speed_mps).all()
    assert all(
        (out / name).is_file() for name in ("result.h5", "run.json", "metrics.json")
    )
    restored = read_result_hdf5(out / "result.h5")
    np.testing.assert_array_equal(restored.sound_speed_mps, result.sound_speed_mps)
    record = json.loads((out / "run.json").read_text())
    assert record["algorithm"] == "inr"
    assert record["variant"] == "inr"
    report = evaluate(tmp_path)
    assert report["passed"], report["records"]
    assert len(report["records"]) == 1
    assert report["records"][0]["case_id"] == case.case_id


def test_masked_nan_and_differential_reference_roundtrip(tmp_path):
    case = first_arrival_case(missing_channels=True)
    write_case_hdf5(case, tmp_path / "case.h5")
    restored = read_case_hdf5(tmp_path / "case.h5")
    np.testing.assert_array_equal(
        restored.measurement.valid_mask, case.measurement.valid_mask
    )
    np.testing.assert_array_equal(
        restored.measurement.ray_weights, case.measurement.ray_weights
    )
    np.testing.assert_allclose(
        restored.measurement.delta_tof_s, case.measurement.delta_tof_s, equal_nan=True
    )
    assert restored.metadata["reference_tof_definition"] == "distance_over_c0"


@pytest.mark.parametrize(
    "invalid",
    [
        "valid_nan",
        "both_representations",
        "missing_reference",
        "wrong_reference",
        "geometry_shape",
    ],
)
def test_reject_inconsistent_first_arrival_case(invalid):
    payload = first_arrival_case().model_dump()
    if invalid == "valid_nan":
        payload["measurement"]["delta_tof_s"][0, 1] = np.nan
    elif invalid == "both_representations":
        payload["measurement"]["tof_s"] = np.zeros((4, 4))
    elif invalid == "missing_reference":
        payload["metadata"].pop("reference_tof_definition")
    elif invalid == "wrong_reference":
        payload["metadata"]["reference_tof_definition"] = "eikonal_background"
    else:
        payload["geometry"]["rx_pos_m"] = payload["geometry"]["rx_pos_m"][:-1]
    with pytest.raises(ValueError):
        FTTTCase.model_validate(payload)


def cli_json(cwd, *arguments):
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        filter(None, (str(REPOSITORY / "src"), environment.get("PYTHONPATH")))
    )
    environment.update(
        OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1"
    )
    process = subprocess.run(
        [sys.executable, "-m", "ftttbench", *arguments],
        cwd=cwd,
        env=environment,
        text=True,
        capture_output=True,
        timeout=60,
    )
    assert process.returncode == 0, process.stdout + process.stderr
    return json.loads(process.stdout)


def test_cli_simulation_and_discovery_from_another_directory(tmp_path):
    assert set(cli_json(tmp_path, "list-algorithms")) == {"dip", "inr", "dps"}
    created = cli_json(
        tmp_path,
        "simulate",
        "--out",
        "cases/small.h5",
        "--shape",
        "8",
        "8",
        "--n-transducers",
        "4",
    )
    inspected = cli_json(tmp_path, "inspect-case", "--case", "cases/small.h5")
    assert inspected["case_id"] == created["case_id"]
    assert inspected["shape"] == [8, 8]
    assert inspected["observations"] == [4, 4]


def test_prior_manifest_is_relative_to_yaml(tmp_path, monkeypatch):
    configs = tmp_path / "configuration"
    configs.mkdir()
    manifest = configs / "priors" / "manifest.json"
    manifest.parent.mkdir()
    shutil.copy2(
        REPOSITORY / "diffusion/weights/synthetic_ellipses/manifest.json", manifest
    )
    config_path = configs / "dps.yaml"
    config = small_config("dps")
    config["parameters"]["prior_manifest"] = "priors/manifest.json"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    prepared = prepare_config("dps", config_path)
    assert Path(prepared.parameters.prior_manifest) == manifest.resolve()


def test_suite_resolves_case_configs_priors_and_outputs_from_its_file(
    tmp_path, monkeypatch
):
    pytest.importorskip("torch")
    project = tmp_path / "benchmark"
    (project / "cases").mkdir(parents=True)
    (project / "configs").mkdir()
    shutil.copytree(
        REPOSITORY / "diffusion/weights/synthetic_ellipses",
        project / "configs/priors",
    )
    write_case_hdf5(first_arrival_case((16, 16)), project / "cases/case.h5")
    for name in ("dip", "inr", "dps"):
        config = small_config(name)
        config["run_controls"]["max_iterations"] = 1
        if name == "dps":
            config["parameters"]["prior_manifest"] = "priors/manifest.json"
        (project / f"configs/{name}.yaml").write_text(
            yaml.safe_dump(config), encoding="utf-8"
        )
    definition = {
        "run_id": "portable",
        "case_glob": "cases/*.h5",
        "outputs": {"root": "outputs"},
        "algorithms": [
            {"id": name, "name": name, "config": f"configs/{name}.yaml"}
            for name in ("dip", "inr", "dps")
        ],
    }
    suite_path = project / "suite.yaml"
    suite_path.write_text(yaml.safe_dump(definition), encoding="utf-8")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    report = suite(suite_path)
    assert report["passed"], report["records"]
    assert report["num_cases"] == 1
    assert Path(report["run_root"]) == project / "outputs/portable"
    assert {record["algorithm"] for record in report["records"]} == {
        "dip",
        "inr",
        "dps",
    }
    for name in ("dip", "inr", "dps"):
        run_record = project / "outputs/portable" / name / "workflow_case/run.json"
        record = json.loads(run_record.read_text())
        assert record["experiment_id"] == name
        assert record["completed_iterations"] == 1


@pytest.mark.parametrize("invalid", ["algorithm_mismatch", "valid_nan", "observable"])
def test_low_level_algorithm_revalidates_its_request(invalid):
    case = first_arrival_case()
    config = small_config("inr" if invalid == "algorithm_mismatch" else "dip")
    config["run_controls"]["max_iterations"] = 0
    prepared = prepare_config(config["name"], config)
    if invalid == "valid_nan":
        case.measurement.delta_tof_s[0, 1] = np.nan
    elif invalid == "observable":
        case.measurement.observable = "pressure"
    with pytest.raises(ValueError):
        get_algorithm("dip").run(case, prepared)


def test_suite_rejects_duplicate_case_ids_before_running_or_writing(
    tmp_path, monkeypatch
):
    from ftttbench.benchmark import runner

    cases = tmp_path / "cases"
    cases.mkdir()
    case = first_arrival_case()
    write_case_hdf5(case, cases / "first.h5")
    write_case_hdf5(case, cases / "second.h5")
    config_path = tmp_path / "inr.yaml"
    config_path.write_text(yaml.safe_dump(small_config("inr")), encoding="utf-8")
    suite_path = tmp_path / "suite.yaml"
    suite_path.write_text(
        yaml.safe_dump(
            {
                "run_id": "duplicate",
                "case_glob": "cases/*.h5",
                "outputs": {"root": "outputs"},
                "algorithms": [{"id": "inr", "name": "inr", "config": "inr.yaml"}],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        runner,
        "run",
        lambda *args, **kwargs: pytest.fail("suite ran before validating all case IDs"),
    )
    with pytest.raises(ValueError):
        suite(suite_path)
    assert not any(path.is_file() for path in (tmp_path / "outputs").rglob("*"))
