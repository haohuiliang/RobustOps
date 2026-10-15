from __future__ import annotations

import argparse
import sys

from robustops.launch import inference, train
from robustops.launch.config import LaunchConfigError, load_recipe, project_root


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run.sh",
        description="Configuration-driven RobustOps training and inference.",
    )
    subparsers = parser.add_subparsers(dest="interface", required=True)

    train_parser = subparsers.add_parser("train", help="route SFT or RL from a training recipe")
    train_parser.add_argument("--config", required=True)
    train_parser.add_argument("--gpus", type=int, default=None, help="optional count; default: all visible NVIDIA GPUs")
    train_parser.add_argument("--dry-run", action="store_true")
    train_parser.add_argument(
        "hydra_overrides",
        nargs=argparse.REMAINDER,
        help="optional backend overrides after --",
    )

    inference_parser = subparsers.add_parser(
        "inference", help="route an inference/evaluation pipeline from a recipe"
    )
    inference_parser.add_argument("--config", required=True)
    inference_parser.add_argument("--gpus", type=int, default=None, help="optional count; fixed evaluations default to recipe hardware.gpus")
    inference_parser.add_argument(
        "--partition", choices=("all", "train", "validation", "test"), default=None
    )
    inference_parser.add_argument("--dry-run", action="store_true")
    inference_parser.add_argument(
        "--base-fixed-mechanisms", help="override the fixed fault manifest for local or teacher evaluation"
    )
    inference_parser.add_argument("--output", help="override the evaluation or Base assignment output directory")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = project_root()
    try:
        recipe_path, recipe = load_recipe(root, args.config)
        if args.interface == "train":
            overrides = args.hydra_overrides
            if overrides[:1] == ["--"]:
                overrides = overrides[1:]
            return train.launch(
                root,
                recipe_path,
                recipe,
                gpu_override=args.gpus,
                dry_run=args.dry_run,
                hydra_overrides=overrides,
            )
        if args.base_fixed_mechanisms is not None:
            if recipe.get("pipeline") not in ("fixed_checkpoint_test", "teacher_fixed_test"):
                raise LaunchConfigError("--base-fixed-mechanisms requires a fixed evaluation recipe")
            recipe["base_fixed_mechanisms"] = args.base_fixed_mechanisms
        if args.output is not None:
            if (recipe.get("pipeline") not in ("fixed_checkpoint_test", "teacher_fixed_test", "fault_assignment_evaluation")
                    or recipe.get("samples_per_task", 1) != 1):
                raise LaunchConfigError("--output requires fixed evaluation or a single-sample Base assignment recipe")
            recipe["output"] = args.output
        return inference.launch(
            root,
            recipe_path,
            recipe,
            gpu_override=args.gpus,
            partition_override=args.partition,
            dry_run=args.dry_run,
        )
    except (LaunchConfigError, KeyError, TypeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


def train_main() -> int:
    return main(["train", *sys.argv[1:]])


def inference_main() -> int:
    return main(["inference", *sys.argv[1:]])


if __name__ == "__main__":
    raise SystemExit(main())
