# FTTT BenchLab

English | [简体中文](README.zh-CN.md)

A standalone codebase for two-dimensional First-arrival Travel-time Tomography (FTTT).
It reconstructs sound speed maps from transducer positions and first-arrival travel times,
with tools for managing cases, configurations, runs, and results.
The Python package is `ftttbench`, and the command-line entry point is `fttt`.
The complete `fttt-benchlab` directory contains everything needed to install and run the code;
no other code repositories are required.

There are exactly three public reconstruction algorithms:

| ID | Method | Image representation | Pretrained weights required |
| --- | --- | --- | --- |
| `dip` | Deep Image Prior | SkipUNet with a fixed random input | No |
| `inr` | Implicit Neural Representation | SIREN with coordinate inputs | No |
| `dps` | Diffusion Posterior Sampling | VP diffusion prior with travel-time observation guidance | Yes |

All three methods share a differentiable Eikonal first-arrival forward model and are invoked
as `dip`, `inr`, and `dps`, respectively.

## Installation

Python 3.12 or later is required. Run the following commands inside the received
`fttt-benchlab` directory:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[neural,viz]'
fttt list-algorithms --json
fttt describe-algorithm inr
```

The base installation, `python -m pip install -e .`, includes NumPy, SciPy, h5py,
Pydantic, PyYAML, and scikit-image. It supports algorithm discovery and case generation
and reading. Running any of the three neural methods also requires `neural` (PyTorch).
The `performance` extra provides Numba, and `viz` provides Matplotlib/Pillow.
The `dev` extra provides pytest, build, and Ruff; install it separately with
`python -m pip install -e '.[dev]'` for development or testing.
Install a CUDA build of PyTorch appropriate for your driver environment if needed;
the numerical Eikonal solver currently runs on the CPU.

## Start with a synthetic case

```bash
fttt simulate --out data/demo.h5
fttt inspect --case data/demo.h5
fttt run inr --case data/demo.h5 --config configs/algorithms/fttt_inr.yaml --out runs/demo_inr
fttt evaluate --run runs/demo_inr
```

You can also use `python -m ftttbench` in place of `fttt`. Run commands require an empty
output directory; choose a new directory for each experiment.

The following Python example provides a short workflow without starting with the longer
default iteration budgets:

```python
from ftttbench import run, write_case_hdf5
from ftttbench.data.synthetic import make_first_arrival_case

case = make_first_arrival_case(shape=(16, 16), n_transducers=12)
write_case_hdf5(case, "data/python_demo.h5")
result = run(
    "inr", case,
    {"parameters": {"seed": 0}, "run_controls": {"max_iterations": 30}},
    out="runs/python_inr", preview=False,
)
if str(result.status) == "failed":
    raise RuntimeError(result.failure_reason)
print(result.sound_speed_mps.shape, result.metrics["data_relative_residual"])
```

Run the three-method demo and a batch comparison:

```bash
python examples/fttt_demo.py --out runs/three_methods --iterations 30 --sampling-steps 16
fttt bench --suite configs/benchmarks/synthetic.yaml
```

The batch configuration reads synthetic cases from `data/` by default, so run `simulate`
first as shown above. File paths in YAML configurations are resolved relative to the
directory containing that YAML file. For example, the DPS algorithm configuration uses
`../../diffusion/weights/synthetic_ellipses/manifest.json`.

Each run saves `result.h5`, `run.json`, and `metrics.json`. After numerical results are
saved successfully, it also writes `index.json`; when previews are enabled, it writes
`preview.png`. Batch evaluation summarizes existing run records without rerunning the
solvers. Inspect `status`, `failure_reason`, `stages`, and `stop_reason`, and compare
forward/adjoint call counts and elapsed time as well as reconstruction metrics:
the three algorithms have different costs per iteration.

## Data conventions

An `FTTTCase` consists of `grid`, `geometry`, `measurement`, optional `ground_truth`,
and `metadata`. Arrays and coordinates use `[y, x]` order, and grid shapes are `(ny, nx)`.
The `origin_m` field gives the boundary coordinate of the first pixel; add half the grid
spacing to obtain pixel centers. All lengths are in m, travel times in s, and sound speeds
in m/s.

The `measurement` must use `domain="features"` and `observable="first_arrival"`.
Provide exactly one of `tof_s` (absolute first-arrival travel times) and `delta_tof_s`
(differential first-arrival travel times), together with a `valid_mask` of the same
`(n_tx, n_rx)` shape. Valid channels must contain finite values, and absolute travel times
must be nonnegative; invalid channels may contain NaN.
Differential travel times also require the following metadata:

```python
metadata = {
    "reference_sound_speed_mps": 1500.0,
    "reference_tof_definition": "distance_over_c0",
}
```

The differential travel time is defined as
`first-arrival travel time - transmitter–receiver distance / c0`.
Raw pressure waveforms, cross-correlation delays, and phase-slope delays cannot be used
directly as first-arrival travel times. See [Development and data integration](docs/development.md)
(Chinese) for an example of constructing a case from experimental data.

## Scope of DPS weights

The included `synthetic_ellipses` prior was trained on **16×16 synthetic images** and is
provided only for interface checks and synthetic experiments. The `openbreastus` and
`nbpslice2d` manifests are `untrained` placeholders; corresponding experimental-data
priors are not available. Grid dimensions, normalization ranges, and model architecture
must match the checkpoint. The loader does not resize automatically or substitute random
weights when a checkpoint is missing. DIP and INR do not depend on these weights.

Use the built-in training entry point below. The output path must not already exist:

```bash
python -m ftttbench.algorithms.fttt.train_prior --output checkpoints/my_prior.pt --steps 5000 --shape 16 16
```

See [Algorithm documentation](docs/algorithm.md) (Chinese) for the full algorithm
descriptions, weight requirements, and scope. The synthetic demos in this codebase do not
establish reconstruction quality on experimental data.

## Directory layout and validation

```text
configs/algorithms/       YAML configurations for the three algorithms
configs/benchmarks/       Batch configurations for cases × algorithms
src/ftttbench/core/       Types, configuration, registry, HDF5, budgets, and provenance
src/ftttbench/operators/  Eikonal forward model, linearization, and PyTorch gradient bridge
src/ftttbench/models/     SkipUNet, SIREN, TinyUNet
src/ftttbench/solvers/    Numerical loops for DIP, INR, and DPS
src/ftttbench/algorithms/ FTTT physics adaptation, losses, weights, and training entry point
src/ftttbench/benchmark/  Run execution, saving, and aggregation
examples/                Minimal reproducible demos
tests/                   Data contract, physics, algorithm, and command-line checks
diffusion/weights/       Dataset manifests and the synthetic prior
```

```bash
python -m pip install -e '.[dev]'
python -m pytest -q
```

Tests cover numerical gradient/adjoint consistency, masks and held-out channels,
stopping budgets, and weight validation. Passing these tests confirms the corresponding
interfaces and numerical checks; it does not establish reconstruction quality on
experimental data.

See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for third-party code attribution
and licensing information.

See [Data format](docs/data_format.md) (Chinese) for data fields and file structure, and
[Validation records](docs/validation.md) (Chinese) for the previously recorded CPU synthetic workflow results.
