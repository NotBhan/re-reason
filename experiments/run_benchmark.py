"""Phase 1 Benchmark and Comparison Runner.

Supports:
    --config: Path to YAML config file.
    --seed: Random seed (default: 42).
    --device: Execution device (auto, cpu, cuda).
    --model: Model architecture (rereason, gru, both).
    --num-seeds: Number of independent random seeds to evaluate (default: 1).
"""

import argparse
import os
import sys
from typing import Any, Dict, List
import yaml
import torch

from re_reason.benchmarks.register_swap import RegisterTrackingTask
from re_reason.core.baseline_gru import GRUPhase1Model
from re_reason.core.model import ReReasonPhase1Model
from re_reason.training.trainer import EvalMetrics, train_phase1_model
from re_reason.utils.seeding import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RE:Reason Phase 1 Benchmark Runner")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/phase1_core.yaml",
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Target execution device.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="both",
        choices=["rereason", "gru", "both"],
        help="Model architecture to run.",
    )
    parser.add_argument(
        "--num-seeds",
        type=int,
        default=1,
        help="Number of consecutive seeds to evaluate (e.g. 5 for full Phase 1 comparison).",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Override total training steps.",
    )
    return parser.parse_args()


def resolve_device(device_str: str) -> torch.device:
    if device_str == "cpu":
        return torch.device("cpu")
    elif device_str == "cuda":
        if not torch.cuda.is_available():
            print("[Warning] CUDA/ROCm requested but unavailable. Falling back to CPU.", file=sys.stderr)
            return torch.device("cpu")
        return torch.device("cuda")
    else:  # auto
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_single_experiment(
    model_name: str,
    config: Dict[str, Any],
    seed: int,
    device: torch.device,
    steps_override: int | None = None,
) -> Dict[str, Any]:
    """Runs a single seed training and evaluation run."""
    set_seed(seed)

    task_cfg = config.get("task", {})
    task = RegisterTrackingTask(
        num_registers=task_cfg.get("num_registers", 4),
        sequence_length=task_cfg.get("sequence_length", 10),
        num_values=task_cfg.get("num_values", 10),
        seed=seed,
    )

    train_cfg = config.get("training", {})
    total_steps = steps_override or train_cfg.get("total_steps", 1200)
    batch_size = train_cfg.get("batch_size", 64)
    lr = train_cfg.get("learning_rate", 1e-3)
    wd = train_cfg.get("weight_decay", 0.01)
    clip_norm = train_cfg.get("gradient_clip_norm", 1.0)

    if model_name == "rereason":
        model_cfg = config.get("model", {})
        model = ReReasonPhase1Model(
            input_dim=task.input_dim,
            state_dim=model_cfg.get("state_dim", 256),
            encoder_hidden_dim=model_cfg.get("encoder_hidden_dim", 64),
            core_hidden_dim=model_cfg.get("core_hidden_dim", 256),
            num_registers=task.num_registers,
            num_values=task.num_values,
        )
    elif model_name == "gru":
        # Load GRU baseline config if available
        gru_cfg_path = "configs/baselines/gru_baseline.yaml"
        if os.path.exists(gru_cfg_path):
            with open(gru_cfg_path, "r") as f:
                gru_raw_cfg = yaml.safe_load(f)
            b_cfg = gru_raw_cfg.get("baseline", {})
        else:
            b_cfg = {}

        model = GRUPhase1Model(
            input_dim=task.input_dim,
            encoder_hidden_dim=b_cfg.get("encoder_hidden_dim", 64),
            encoder_out_dim=b_cfg.get("encoder_out_dim", 256),
            gru_hidden_dim=b_cfg.get("gru_hidden_dim", 162),
            num_registers=task.num_registers,
            num_values=task.num_values,
        )
    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    params = model.count_parameters()

    eval_metrics, wall_clock = train_phase1_model(
        model=model,
        task=task,
        total_steps=total_steps,
        batch_size=batch_size,
        learning_rate=lr,
        weight_decay=wd,
        gradient_clip_norm=clip_norm,
        device=device,
        seed=seed,
    )

    return {
        "model": model_name,
        "seed": seed,
        "params": params,
        "wall_clock_s": wall_clock,
        "exact_match": eval_metrics.final_exact_match,
        "per_reg_acc": eval_metrics.final_per_reg_acc,
        "step_exact": eval_metrics.step_exact_matches,
        "mean_loss": eval_metrics.mean_loss,
        "latency_ms": eval_metrics.inference_latency_ms,
        "peak_mem_mb": eval_metrics.peak_memory_mb,
    }


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    models_to_run = ["rereason", "gru"] if args.model == "both" else [args.model]
    base_seed = args.seed
    num_seeds = args.num_seeds

    print("=" * 80)
    print(f"RE:Reason Phase 1 Experiment Runner")
    print(f"Device: {device} | Base Seed: {base_seed} | Num Seeds: {num_seeds}")
    print(f"Config: {args.config}")
    print("=" * 80)

    all_results: Dict[str, List[Dict[str, Any]]] = {m: [] for m in models_to_run}

    for m in models_to_run:
        print(f"\n--- Running Model: {m.upper()} ---")
        for s_idx in range(num_seeds):
            current_seed = base_seed + s_idx
            res = run_single_experiment(
                model_name=m,
                config=config,
                seed=current_seed,
                device=device,
                steps_override=args.steps,
            )
            all_results[m].append(res)
            print(
                f"Seed {current_seed:2d} | Params: {res['params']:,} | "
                f"Exact Match: {res['exact_match']*100:.2f}% | "
                f"Per-Reg Acc: {res['per_reg_acc']*100:.2f}% | "
                f"Time: {res['wall_clock_s']:.2f}s | "
                f"Loss: {res['mean_loss']:.4f}"
            )

    # Print summary statistics
    print("\n" + "=" * 80)
    print("PHASE 1 EXPERIMENT SUMMARY & DECISION CRITERIA")
    print("=" * 80)
    summary_data = {}
    for m in models_to_run:
        runs = all_results[m]
        exact_scores = [r["exact_match"] for r in runs]
        per_reg_scores = [r["per_reg_acc"] for r in runs]
        times = [r["wall_clock_s"] for r in runs]
        params = runs[0]["params"]

        mean_exact = sum(exact_scores) / len(exact_scores)
        std_exact = (sum((x - mean_exact) ** 2 for x in exact_scores) / len(exact_scores)) ** 0.5 if len(exact_scores) > 1 else 0.0

        mean_per_reg = sum(per_reg_scores) / len(per_reg_scores)
        std_per_reg = (sum((x - mean_per_reg) ** 2 for x in per_reg_scores) / len(per_reg_scores)) ** 0.5 if len(per_reg_scores) > 1 else 0.0

        mean_time = sum(times) / len(times)

        summary_data[m] = {
            "params": params,
            "mean_exact": mean_exact,
            "std_exact": std_exact,
            "mean_per_reg": mean_per_reg,
            "std_per_reg": std_per_reg,
            "mean_time": mean_time,
            "step_exact": [
                sum(runs[i]["step_exact"][t] for i in range(len(runs))) / len(runs)
                for t in range(len(runs[0]["step_exact"]))
            ],
        }

        print(f"\nModel: {m.upper()}")
        print(f"  Parameters:          {params:,}")
        print(f"  Exact Match at t=10: {mean_exact*100:.2f}% ± {std_exact*100:.2f}%")
        print(f"  Per-Register Acc:    {mean_per_reg*100:.2f}% ± {std_per_reg*100:.2f}%")
        print(f"  Avg Training Time:   {mean_time:.2f}s")
        print(f"  Step-wise Exact Acc: {[round(s*100, 1) for s in summary_data[m]['step_exact']]}")

    if "rereason" in summary_data and "gru" in summary_data:
        diff = summary_data["rereason"]["mean_exact"] - summary_data["gru"]["mean_exact"]
        param_diff = abs(summary_data["rereason"]["params"] - summary_data["gru"]["params"]) / summary_data["rereason"]["params"]
        print("\n" + "-" * 80)
        print(f"Comparison: RE:Reason vs GRU Baseline")
        print(f"  Parameter Parity:  {param_diff*100:.2f}% delta (within ±5% rule: {'PASS' if param_diff <= 0.05 else 'FAIL'})")
        print(f"  Exact Match Delta: {diff*100:+.2f}%")
        decision = abs(diff) <= 0.02 or diff > 0.02
        print(f"  Phase 1 Decision Rule (Within ±2% of GRU): {'GO (PASS)' if decision else 'NO-GO (FAIL)'}")
        print("-" * 80)


if __name__ == "__main__":
    main()
