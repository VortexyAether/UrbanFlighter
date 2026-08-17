from __future__ import annotations

import argparse
import json
from pathlib import Path

from .cfd_adapter import StructuredCFDDataset
from .evaluation import DEFAULT_ARTIFACT_PATH, evaluation_summary, run_baseline_evaluation
from .scenario import DEFAULT_HELD_OUT_SEEDS


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the legacy synthetic rectangular UrbanFlow fixture baselines. "
            "Primary live-world evaluation is POST /urbanflow-gym/live/evaluate."
        )
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=list(DEFAULT_HELD_OUT_SEEDS),
        help="One to five deterministic synthetic-fixture seeds.",
    )
    parser.add_argument("--max-steps", type=int, default=360)
    parser.add_argument("--output", type=Path, default=DEFAULT_ARTIFACT_PATH)
    parser.add_argument(
        "--cfd-npz",
        type=Path,
        default=None,
        help=(
            "Optional structured 3D snapshot/temporal field. Without this flag the run is "
            "synthetic proxy evaluation and makes no real-CFD validation claim."
        ),
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    dataset = StructuredCFDDataset.from_npz(args.cfd_npz) if args.cfd_npz else None
    payload = run_baseline_evaluation(
        seeds=args.seeds,
        max_steps=args.max_steps,
        artifact_path=args.output,
        cfd_dataset=dataset,
    )
    print(json.dumps(evaluation_summary(payload), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
