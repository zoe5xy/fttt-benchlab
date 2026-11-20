"""Independent FTTT discovery, simulation, reconstruction and evaluation."""

import argparse
import json
from pathlib import Path

from . import describe, evaluate, list_algorithms, run, suite
from .core.io import read_case_hdf5, write_case_hdf5


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    listing = sub.add_parser("list-algorithms")
    listing.add_argument("--json", action="store_true")
    info = sub.add_parser("describe-algorithm")
    info.add_argument(
        "algorithm", nargs="?", default="dip", choices=["dip", "inr", "dps", "fttt"]
    )
    info.add_argument("--variant", choices=["dip", "inr", "dps"])
    sim = sub.add_parser("simulate")
    sim.add_argument("--out", type=Path, required=True)
    sim.add_argument("--shape", nargs=2, type=int, default=[16, 16])
    sim.add_argument("--n-transducers", type=int, default=12)
    inspect = sub.add_parser("inspect", aliases=["inspect-case"])
    inspect.add_argument("--case", type=Path, required=True)
    execute = sub.add_parser("run")
    execute.add_argument(
        "algorithm", nargs="?", default="inr", choices=["dip", "inr", "dps", "fttt"]
    )
    execute.add_argument("--case", type=Path, required=True)
    execute.add_argument("--config", type=Path, required=True)
    execute.add_argument("--out", type=Path, required=True)
    execute.add_argument("--preview", action="store_true")
    batch = sub.add_parser("bench", aliases=["suite"])
    batch.add_argument("--suite", type=Path, required=True)
    scoring = sub.add_parser("evaluate")
    scoring.add_argument("--run", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "list-algorithms":
            value = (
                [describe(entry.name) for entry in list_algorithms()]
                if args.json
                else [x.name for x in list_algorithms()]
            )
        elif args.command == "describe-algorithm":
            value = describe(args.algorithm, args.variant)
        elif args.command == "simulate":
            from .data.synthetic import make_first_arrival_case

            if args.out.exists():
                raise FileExistsError(f"refusing to overwrite {args.out}")
            case = make_first_arrival_case(tuple(args.shape), args.n_transducers)
            args.out.parent.mkdir(parents=True, exist_ok=True)
            write_case_hdf5(case, args.out)
            value = {"case_id": case.case_id, "path": str(args.out)}
        elif args.command in ("inspect", "inspect-case"):
            case = read_case_hdf5(args.case)
            value = {
                "case_id": case.case_id,
                "shape": case.grid.shape,
                "observable": case.measurement.observable,
                "num_valid_channels": int(case.measurement.valid_mask.sum()),
                "observation_mode": "differential"
                if case.measurement.delta_tof_s is not None
                else "absolute",
                "reference_sound_speed_mps": case.metadata.get(
                    "reference_sound_speed_mps"
                ),
                "reference_tof_definition": case.metadata.get(
                    "reference_tof_definition"
                ),
                "observations": (
                    case.measurement.delta_tof_s.shape
                    if case.measurement.delta_tof_s is not None
                    else case.measurement.tof_s.shape
                ),
            }
        elif args.command == "run":
            if args.out.exists() and any(args.out.iterdir()):
                raise ValueError("output directory must be empty")
            result = run(
                args.algorithm, args.case, args.config, args.out, preview=args.preview
            )
            value = {
                "status": str(result.status),
                "stop_reason": result.metrics.get("stop_reason"),
                "failure_reason": result.failure_reason,
                "out": str(args.out),
            }
            if str(result.status) == "failed":
                print(json.dumps(value, ensure_ascii=False, indent=2))
                return 1
        elif args.command in ("bench", "suite"):
            value = suite(args.suite)
        else:
            value = evaluate(args.run)
        print(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))
        if isinstance(value, dict) and "passed" in value and not value["passed"]:
            return 1
        return 0
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        parser.exit(1, str(exc) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
