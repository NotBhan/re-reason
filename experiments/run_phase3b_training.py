"""Milestone 3B behavioral-cloning training runner (Stage A infrastructure; execution gated).

This runner implements the authorized training infrastructure for the paired
behavioral-cloning objective in ``docs/research/pre_3b_experimental_protocol.md``.
It is **inert during Milestone 3B Stage A**: execution requires the explicit
``--stage-b-authorized`` flag, which is only to be supplied after separate PI
authorization of Stage B.

Engineering properties:

* Trains on the BFS shortest-path demonstrations from the ``train`` partition
  only; the loader is hard-restricted to ``train`` and never materializes the
  validation or held-out ``test`` partitions.
* Refuses to overwrite existing checkpoint or telemetry artifacts by default.
  Replacement requires ``--allow-overwrite-authorized`` and emits an explicit
  ``WARNING`` on stdout plus ``overwrite_authorized`` fields in telemetry.
* Writes artifacts only after training completes, using unique timestamped
  run-specific paths and atomic temp-file replacement, so setup or training
  failures never leave partial or misleading result files behind.
* Contains no evaluation path: validation and held-out evaluation will be a
  separate entry point under separate authorization. ``HISTORICAL_BASELINE.json``
  is never touched here.
* Training hyperparameters (epochs, batch size, learning rate, weight decay,
  gradient clipping) are supplied explicitly on the command line; the protocol
  does not ratify numeric values, so no defaults are baked in. AdamW follows the
  repository's established optimizer convention.
"""

import argparse
import json
import os
import sys
import tempfile
import time
from typing import Any, Dict, List, Optional, Sequence

import torch

from re_reason.benchmarks.pathfinding_benchmark import PathfindingGeneratorConfig
from re_reason.hypothesis.pathfinding_branching import ActionConditionedBrancher
from re_reason.training.behavioral_cloning import (
    iter_training_batches,
    load_behavioral_cloning_demonstrations,
    teacher_forced_batch_loss,
)
from re_reason.utils.rng_namespaces import configure_training_rng


DEFAULT_OUTPUT_DIR = "experiments/phase3b_runs"
DEFAULT_TRAIN_PARTITION_SIZE = 200
DEFAULT_GRAPH_BASE_SEED = 42


def resolve_run_artifacts(
    seed: int,
    run_label: str,
    timestamp: str,
    output_dir: str = DEFAULT_OUTPUT_DIR,
) -> Dict[str, str]:
    """Returns unique, run-specific checkpoint and telemetry paths."""
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 1:
        raise ValueError(f"seed must be an integer >= 1, got {seed!r}")
    for name, value in (("run_label", run_label), ("timestamp", timestamp), ("output_dir", output_dir)):
        if not isinstance(value, str) or not value:
            raise ValueError(f"{name} must be a non-empty string, got {value!r}")
    stem = f"phase3b_bc_seed{seed}_{run_label}_{timestamp}"
    return {
        "checkpoint": os.path.join(output_dir, f"{stem}.pt"),
        "telemetry": os.path.join(output_dir, f"{stem}.json"),
    }


def check_artifact_targets(paths: Sequence[str], allow_overwrite: bool) -> List[str]:
    """Validates artifact destinations before any training starts.

    Returns the list of existing paths that will be overwritten (empty when all
    destinations are new). Raises :class:`FileExistsError` for an existing path
    unless explicit overwrite authorization is supplied; an authorized overwrite
    emits a clear warning on stdout.
    """
    overwritten: List[str] = []
    for path in paths:
        if not isinstance(path, str) or not path:
            raise TypeError(f"artifact paths must be non-empty strings, got {path!r}")
        if os.path.exists(path):
            if not allow_overwrite:
                raise FileExistsError(
                    f"refusing to overwrite existing artifact: {path}. "
                    "Re-run with --allow-overwrite-authorized to replace it explicitly."
                )
            overwritten.append(path)
            print(
                f"WARNING: overwriting existing artifact (explicitly authorized): {path}"
            )
    return overwritten


def _write_atomically(path: str, write_payload) -> None:
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    descriptor, temporary_path = tempfile.mkstemp(prefix=".tmp-", dir=directory)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            write_payload(handle)
        os.replace(temporary_path, path)
    except BaseException:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)
        raise


def write_json_atomic(path: str, payload: Any) -> None:
    """Writes JSON to ``path`` atomically; failures leave no partial target file."""

    def write(handle) -> None:
        handle.write(json.dumps(payload, indent=2).encode("utf-8"))

    _write_atomically(path, write)


def write_checkpoint_atomic(path: str, state_dict: Dict[str, Any], metadata: Dict[str, Any]) -> None:
    """Writes a checkpoint atomically; failures leave no partial target file."""

    def write(handle) -> None:
        torch.save({"state_dict": state_dict, "metadata": metadata}, handle)

    _write_atomically(path, write)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Milestone 3B behavioral-cloning training runner (execution requires --stage-b-authorized)"
    )
    parser.add_argument(
        "--stage-b-authorized",
        action="store_true",
        help="Required explicit Milestone 3B Stage B authorization gate.",
    )
    parser.add_argument("--seed", type=int, required=True, help="Paired model training seed s.")
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--weight-decay", type=float, required=True)
    parser.add_argument("--gradient-clip-norm", type=float, required=True)
    parser.add_argument(
        "--train-size",
        type=int,
        default=DEFAULT_TRAIN_PARTITION_SIZE,
        help="Number of training episodes (protocol: 200).",
    )
    parser.add_argument(
        "--base-seed",
        type=int,
        default=DEFAULT_GRAPH_BASE_SEED,
        help="Graph-generation base seed (protocol: 42).",
    )
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--run-label", default="bc")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument(
        "--allow-overwrite-authorized",
        action="store_true",
        help="Explicitly authorize replacing an existing artifact.",
    )
    return parser


def _validate_arguments(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    positive_ints = (
        ("--seed", args.seed),
        ("--epochs", args.epochs),
        ("--batch-size", args.batch_size),
        ("--train-size", args.train_size),
    )
    for name, value in positive_ints:
        if value < 1:
            parser.error(f"{name} must be a positive integer, got {value}")
    if args.base_seed < 0:
        parser.error(f"--base-seed must be non-negative, got {args.base_seed}")
    if args.learning_rate <= 0.0:
        parser.error(f"--learning-rate must be positive, got {args.learning_rate}")
    if args.weight_decay < 0.0:
        parser.error(f"--weight-decay must be non-negative, got {args.weight_decay}")
    if args.gradient_clip_norm <= 0.0:
        parser.error(f"--gradient-clip-norm must be positive, got {args.gradient_clip_norm}")


def _resolve_device(requested: str) -> torch.device:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("--device cuda requested but no CUDA/ROCm device is available")
        return torch.device("cuda")
    return torch.device("cpu")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)

    if not args.stage_b_authorized:
        parser.error(
            "Milestone 3B Stage B training is not authorized. This runner is inert during "
            "Stage A; re-run with --stage-b-authorized only after separate PI authorization."
        )
    _validate_arguments(parser, args)

    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    artifacts = resolve_run_artifacts(args.seed, args.run_label, timestamp, args.output_dir)
    overwritten = check_artifact_targets(
        list(artifacts.values()), args.allow_overwrite_authorized
    )

    device = _resolve_device(args.device)
    training_rng = configure_training_rng(args.seed)

    demonstrations = load_behavioral_cloning_demonstrations(
        PathfindingGeneratorConfig(), size=args.train_size, base_seed=args.base_seed
    )
    if not demonstrations:
        raise RuntimeError("no solvable demonstrations were produced for the training partition")

    brancher = ActionConditionedBrancher().to(device)
    optimizer = torch.optim.AdamW(
        brancher.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )

    telemetry: Dict[str, Any] = {
        "runner": "run_phase3b_training",
        "status": "completed",
        "seed": args.seed,
        "partition": "train",
        "train_episodes": args.train_size,
        "demonstration_sequences": len(demonstrations),
        "demonstration_transitions": sum(demo.length for demo in demonstrations),
        "graph_base_seed": args.base_seed,
        "parameter_count": brancher.count_parameters(),
        "device": str(device),
        "hyperparameters": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "gradient_clip_norm": args.gradient_clip_norm,
            "optimizer": "AdamW",
        },
        "rng": {
            "training_python_seed": training_rng.python_seed,
            "training_numpy_seed": training_rng.python_seed,
            "training_model_seed": training_rng.model_seed,
        },
        "overwrite_authorized": bool(args.allow_overwrite_authorized),
        "overwritten_artifacts": overwritten,
        "timestamp_utc": timestamp,
        "epochs": [],
    }

    start_time = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        batch_losses: List[float] = []
        for batch in iter_training_batches(
            demonstrations, args.batch_size, training_rng.batch_generator
        ):
            report = teacher_forced_batch_loss(brancher, batch)
            optimizer.zero_grad()
            report.loss.backward()
            torch.nn.utils.clip_grad_norm_(brancher.parameters(), args.gradient_clip_norm)
            optimizer.step()
            batch_losses.append(float(report.loss.detach().item()))
        telemetry["epochs"].append(
            {
                "epoch": epoch,
                "batches": len(batch_losses),
                "mean_loss": sum(batch_losses) / len(batch_losses) if batch_losses else None,
            }
        )
        print(f"[seed {args.seed}] epoch {epoch}/{args.epochs} mean loss: {telemetry['epochs'][-1]['mean_loss']}")
    telemetry["runtime_seconds"] = time.perf_counter() - start_time

    checkpoint_metadata = {
        "runner": "run_phase3b_training",
        "seed": args.seed,
        "parameter_count": brancher.count_parameters(),
        "hyperparameters": telemetry["hyperparameters"],
        "timestamp_utc": timestamp,
        "training_partition": "train",
    }
    write_checkpoint_atomic(artifacts["checkpoint"], brancher.state_dict(), checkpoint_metadata)
    write_json_atomic(artifacts["telemetry"], telemetry)
    print(f"Wrote checkpoint: {artifacts['checkpoint']}")
    print(f"Wrote telemetry: {artifacts['telemetry']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
