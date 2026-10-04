"""CLI entrypoint for running architectural ablation sweeps."""

import argparse
from re_reason.utils.seeding import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RE:Reason Ablation Runner")
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic ablation execution.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Execution device.",
    )
    parser.add_argument(
        "--ablation",
        type=str,
        default="none",
        help="Ablation identifier.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    print(f"[RE:Reason Ablation Runner] Seed: {args.seed}, Device: {args.device}, Ablation: {args.ablation}")
    print("Phase 0 scaffolding: Ready for Phase 2+ ablation sweeps.")


if __name__ == "__main__":
    main()
