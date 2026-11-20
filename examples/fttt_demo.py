"""Run DIP, INR and a supplied synthetic-prior DPS through one Eikonal operator."""

import argparse
import json
from pathlib import Path
import numpy as np
from ftttbench import run, write_case_hdf5
from ftttbench.data.synthetic import make_first_arrival_case
from ftttbench.algorithms.fttt.weight_bank import reference_prior_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("runs/three_baselines"))
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--sampling-steps", type=int, default=16)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
    )
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("choose a new output directory")
    args.out.mkdir(parents=True)
    case = make_first_arrival_case()
    write_case_hdf5(case, args.out / "case.h5")
    records = []
    for variant in ("dip", "inr", "dps"):
        parameters = {"variant": variant, "seed": 0}
        if variant == "dps":
            parameters.update(
                sampling_steps=args.sampling_steps,
            )
            if args.checkpoint is not None:
                parameters["checkpoint"] = str(args.checkpoint.resolve())
            else:
                parameters["prior_manifest"] = str(reference_prior_manifest())
        config = {
            "parameters": parameters,
            "run_controls": {
                "max_iterations": (
                    args.sampling_steps if variant == "dps" else args.iterations
                )
            },
        }
        result = run(variant, case, config, args.out / variant, preview=False)
        if str(result.status) == "failed":
            raise RuntimeError(result.failure_reason)
        record = {
            "method": variant,
            "status": str(result.status),
            "stop_reason": result.metrics["stop_reason"],
            "relative_data_residual": result.metrics["data_relative_residual"],
            "rmse_mps": float(
                np.sqrt(
                    np.mean(
                        (result.sound_speed_mps - case.ground_truth.sound_speed_mps)
                        ** 2
                    )
                )
            ),
            "runtime_s": result.runtime_s,
            "work": result.metrics["stopping"]["work"],
        }
        records.append(record)
        print(json.dumps(record), flush=True)
    (args.out / "summary.json").write_text(
        json.dumps(
            {
                "scope": "Same-model synthetic execution check; synthetic DPS prior, no experimental-performance claim.",
                "records": records,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
