"""Model E held-out generalization campaign runner (SCLM four-cell benchmark; execution gated).

This runner implements the registered protocol of the 2026-10-18 Phase 1.6 amendment
(`docs/reports/phase_1_6_architectural_reformulation_proposal.md`): four candidates
(A SCLM 8,255; B Slot-MLP 10,249; C Cyclic-MGU 227,062; D Monolithic-MLP 229,056),
seeds 42-46, AdamW (lr 3e-3, weight decay 1e-2), batch size 64, 3,000 updates,
validation every 50 updates on VAL-SEEN, and minimum-validation-loss checkpoint
selection with earliest-update tie-break.

Engineering properties:

* Inert by default: training requires the explicit ``--profile-preflight`` mode plus
  ``--stage-authorized``; without both, nothing is executed and no artifact is created.
* The protected ``b in {0, 5}`` split is reachable only through ``--evaluate-tier2``
  combined with ``--tier2-authorized`` AND a recorded ``tier1_pass`` verdict; the
  loader refuses before any generation occurs otherwise.
* The profiling stop-gate measures the first 100 updates of all 20 registered runs
  (including validations at updates 50 and 100); the projected campaign duration is
  ``sum(30 * t_100)`` and the campaign stops after profiling at >= 12.0 hours without
  altering the registered schedule. A hard cumulative 12.0-hour stop is enforced
  during continuation.
* Refuses to overwrite existing artifacts by default (``--allow-overwrite-authorized``
  emits an explicit WARNING); the Tier 2 protected result is never overwritable.
* Writes artifacts atomically (temp file + rename) with unique timestamped run
  directories, so setup or training failures never leave partial result files.
* Training requires a protected hash manifest supplied via ``--protected-hash-manifest``
  and verifies metadata-only zero sequence overlap between it and the generated
  TRAIN-MASKED / VAL-SEEN / PROXY-DEV pools before any training starts.
"""

import argparse
import hashlib
import json
import os
import tempfile
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch

from re_reason.benchmarks.dataset_split import (
    SplitDataset,
    assert_manifests_disjoint,
    build_hash_manifest,
)
from re_reason.benchmarks.register_machine_benchmark import (
    Phase16RegisterTask,
    Phase16SplitManager,
)
from re_reason.core.sclm_models import build_candidate
from re_reason.training.heldout_objective import (
    classify_tier1_gate,
    collect_eligible_add_transitions,
    heldout_add_balanced_accuracy,
    terminal_sequence_exact_match,
    unified_heldout_loss,
)
from re_reason.utils.rng_namespaces import shuffled_batch_indices
from re_reason.utils.seeding import set_seed


DEFAULT_OUTPUT_DIR = "experiments/model_e_heldout_runs"
CANDIDATE_IDS: Tuple[str, ...] = ("A", "B", "C", "D")
REGISTERED_SEEDS: Tuple[int, ...] = (42, 43, 44, 45, 46)
TOTAL_UPDATES = 3000
BATCH_SIZE = 64
VALIDATION_INTERVAL = 50
PROFILE_UPDATES = 100
RUNTIME_CAP_HOURS = 12.0
LEARNING_RATE = 3e-3
WEIGHT_DECAY = 1e-2
SPLIT_BASE_SEED = 42
TRAIN_SIZE = 6000
VAL_SIZE = 1200
PROXY_PER_CLASS = 120
PROTECTED_ELIGIBLE_SOURCE_VALUES: Tuple[int, ...] = (0, 5)
PROXY_ELIGIBLE_SOURCE_VALUES: Tuple[int, ...] = (3,)


class ProtocolGuardError(RuntimeError):
    """Raised when an execution precondition of the registered protocol is not met."""


class CampaignRuntimeCapExceeded(RuntimeError):
    """Raised when the cumulative campaign wall clock reaches the registered 12.0-hour cap."""


def projected_campaign_seconds(profile_seconds: Mapping[str, float]) -> float:
    """Registered projection: ``sum over 20 runs of 30 * t_100`` (3,000 / 100)."""
    expected = {f"{model_id}:{seed}" for model_id in CANDIDATE_IDS for seed in REGISTERED_SEEDS}
    if set(profile_seconds) != expected:
        raise ValueError(
            "profile_seconds must cover exactly the 20 registered model-seed runs; "
            f"got {sorted(profile_seconds)}"
        )
    factor = TOTAL_UPDATES // PROFILE_UPDATES
    total = 0.0
    for key, value in profile_seconds.items():
        seconds = float(value)
        if not (seconds > 0.0) or seconds != seconds:
            raise ValueError(f"profile timing for {key!r} must be a positive finite number")
        total += factor * seconds
    return total


def require_stage_authorized(stage_authorized: bool) -> None:
    """Fails closed unless the explicit PI-authorization flag is present."""
    if not stage_authorized:
        raise ProtocolGuardError(
            "execution is gated: re-run with --stage-authorized after explicit PI authorization"
        )


def require_tier2_authorized(tier2_authorized: bool, tier1_verdict: Optional[str]) -> None:
    """Fails closed unless the Tier 2 flag is present AND Tier 1 passed."""
    if not tier2_authorized:
        raise ProtocolGuardError(
            "protected evaluation requires --tier2-authorized in addition to --stage-authorized"
        )
    if tier1_verdict != "tier1_pass":
        raise ProtocolGuardError(
            "Tier 1 PROXY-DEV gate has not passed; the protected split remains sealed"
        )


def _generate_protected_split() -> SplitDataset:
    """Materializes the existing sealed TEST-PROTECTED (``b in {0, 5}``) derivation.

    Called only after ``require_tier2_authorized`` has passed; never invoked by any
    other code path. Kept as a module-level indirection so tests can assert that the
    unlicensed path raises before this generator is ever reached.
    """
    task = Phase16RegisterTask()
    manager = Phase16SplitManager(task)
    split, _hashes = manager.generate_disjoint_split(
        VAL_SIZE,
        seed=SPLIT_BASE_SEED + 90000,
        generation_mode="regime_b_heldout_test",
    )
    return split


def load_protected_split(tier2_authorized: bool, tier1_verdict: Optional[str]) -> SplitDataset:
    """Guard-first protected-split loader (authorization is checked before generation)."""
    require_tier2_authorized(tier2_authorized, tier1_verdict)
    return _generate_protected_split()


def check_artifact_targets(paths: Sequence[str], allow_overwrite: bool) -> List[str]:
    """Validates artifact destinations before any training starts.

    Raises ``FileExistsError`` for an existing target unless explicit overwrite
    authorization is supplied; an authorized overwrite emits an explicit WARNING and
    returns the list of replaced paths.
    """
    existing = [path for path in paths if os.path.exists(path)]
    if existing and not allow_overwrite:
        raise FileExistsError(
            f"refusing to overwrite existing artifact: {existing[0]}. "
            "Re-run with --allow-overwrite-authorized to replace it explicitly."
        )
    if existing:
        print(f"WARNING: authorized overwrite of {len(existing)} existing artifact(s): {existing}")
    return existing


def _write_atomically(path: str, write_payload) -> None:
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    suffix = os.path.splitext(path)[1]
    descriptor, temporary_path = tempfile.mkstemp(prefix=".tmp-", suffix=suffix, dir=directory)
    os.close(descriptor)
    try:
        write_payload(temporary_path)
        os.replace(temporary_path, path)
    except BaseException:
        if os.path.exists(temporary_path):
            os.remove(temporary_path)
        raise


def write_json_atomic(path: str, payload: Any) -> None:
    """Writes JSON to ``path`` atomically; failures leave no partial target file."""

    def write(temporary_path: str) -> None:
        with open(temporary_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)

    _write_atomically(path, write)


def write_checkpoint_atomic(path: str, state_dict: Dict[str, Any], metadata: Dict[str, Any]) -> None:
    """Writes a checkpoint atomically; failures leave no partial target file."""

    def write(temporary_path: str) -> None:
        torch.save({"state_dict": state_dict, "metadata": metadata}, temporary_path)

    _write_atomically(path, write)


def sha256_file(path: str) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def resolve_campaign_paths(output_dir: str, timestamp: str) -> Dict[str, str]:
    """Returns the unique timestamped campaign directory and its artifact paths."""
    run_dir = os.path.join(output_dir, f"model_e_heldout_{timestamp}")
    return {
        "run_dir": run_dir,
        "profile": os.path.join(run_dir, "profile_preflight.json"),
        "summary": os.path.join(run_dir, "campaign_summary.json"),
        "tier1": os.path.join(run_dir, "tier1_results.json"),
        "tier2": os.path.join(run_dir, "tier2_results.json"),
    }


def resolve_run_artifacts(run_dir: str, model_id: str, seed: int) -> Dict[str, str]:
    """Returns the per-run selected-checkpoint and telemetry paths."""
    if model_id not in CANDIDATE_IDS:
        raise ValueError(f"unknown candidate id {model_id!r}; expected {sorted(CANDIDATE_IDS)}")
    if seed not in REGISTERED_SEEDS:
        raise ValueError(f"seed {seed!r} is outside the registered set {REGISTERED_SEEDS}")
    stem = f"model_e_heldout_{model_id}_seed{seed}"
    return {
        "checkpoint": os.path.join(run_dir, f"{stem}_selected.pt"),
        "telemetry": os.path.join(run_dir, f"{stem}_telemetry.json"),
    }


def resolve_device(device_option: str) -> torch.device:
    if device_option == "cpu":
        return torch.device("cpu")
    if device_option == "cuda":
        if not torch.cuda.is_available():
            raise ProtocolGuardError("--device cuda requested but no CUDA/ROCm device is available")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate_split_loss_and_em(
    model: torch.nn.Module,
    split: SplitDataset,
    device: torch.device,
) -> Tuple[float, float]:
    """Mean unified validation loss and terminal sequence EM over a split."""
    model.eval()
    total_loss = 0.0
    total_em = 0.0
    num_items = len(split)
    with torch.no_grad():
        for start in range(0, num_items, BATCH_SIZE):
            indices = list(range(start, min(start + BATCH_SIZE, num_items)))
            inputs = split.inputs[indices].to(device)
            step_targets = split.step_targets[indices].to(device)
            output = model(inputs)
            loss, _ = unified_heldout_loss(output.step_logits, step_targets, inputs)
            terminal_predictions = output.step_logits.argmax(dim=-1)[:, -1]
            em = terminal_sequence_exact_match(terminal_predictions.cpu(), step_targets[:, -1].cpu())
            total_loss += float(loss.detach()) * len(indices)
            total_em += em * len(indices)
    return total_loss / num_items, total_em / num_items


class CampaignRun:
    """In-process state for one registered model-seed training run."""

    def __init__(self, model_id: str, seed: int, device: torch.device) -> None:
        self.model_id = model_id
        self.seed = seed
        self.device = device
        set_seed(seed)
        self.model = build_candidate(model_id).to(device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
        )
        self.batch_generator = torch.Generator().manual_seed(seed)
        self.update = 0
        self.best_val_loss = float("inf")
        self.best_update = 0
        self.best_state: Optional[Dict[str, Any]] = None
        self.telemetry: List[Dict[str, Any]] = []
        self.batch_losses: List[float] = []
        self._epoch_batches: List[Tuple[int, ...]] = []
        self._epoch_index = 0
        self.start_time = time.perf_counter()

    def next_batch_indices(self, train_size: int) -> Tuple[int, ...]:
        if self._epoch_index >= len(self._epoch_batches):
            self._epoch_batches = shuffled_batch_indices(
                train_size, BATCH_SIZE, self.batch_generator, drop_last=True
            )
            self._epoch_index = 0
            if not self._epoch_batches:
                raise ProtocolGuardError("training pool does not contain one full registered batch")
        batch = self._epoch_batches[self._epoch_index]
        self._epoch_index += 1
        return batch


def advance_run(
    run: CampaignRun,
    train_set: SplitDataset,
    val_set: SplitDataset,
    target_update: int,
    deadline: Optional[float] = None,
) -> None:
    """Advances a run to ``target_update``, validating every 50 updates (fail-closed)."""
    while run.update < target_update:
        indices = run.next_batch_indices(len(train_set))
        inputs = train_set.inputs[list(indices)].to(run.device)
        step_targets = train_set.step_targets[list(indices)].to(run.device)

        run.model.train()
        output = run.model(inputs)
        loss, _components = unified_heldout_loss(output.step_logits, step_targets, inputs)
        if not torch.isfinite(loss):
            raise RuntimeError(
                f"non-finite training loss at update {run.update + 1} "
                f"(model {run.model_id}, seed {run.seed}); run halted fail-closed"
            )
        run.optimizer.zero_grad()
        loss.backward()
        run.optimizer.step()
        run.update += 1
        run.batch_losses.append(float(loss.detach()))

        if run.update % VALIDATION_INTERVAL == 0:
            if deadline is not None and time.perf_counter() >= deadline:
                raise CampaignRuntimeCapExceeded(
                    f"cumulative campaign wall clock reached the {RUNTIME_CAP_HOURS}-hour cap "
                    f"before update {run.update} of model {run.model_id} seed {run.seed}"
                )
            val_loss, val_em = evaluate_split_loss_and_em(run.model, val_set, run.device)
            run.telemetry.append(
                {
                    "update": run.update,
                    "train_loss_mean_interval": sum(run.batch_losses[-VALIDATION_INTERVAL:])
                    / min(VALIDATION_INTERVAL, len(run.batch_losses)),
                    "val_loss": val_loss,
                    "val_em_terminal": val_em,
                    "elapsed_seconds": time.perf_counter() - run.start_time,
                }
            )
            if val_loss < run.best_val_loss:
                run.best_val_loss = val_loss
                run.best_update = run.update
                run.best_state = {
                    key: value.detach().cpu().clone() for key, value in run.model.state_dict().items()
                }


def load_manifest_file(path: str, default_label: str) -> Dict[str, Any]:
    """Loads an external hash manifest (metadata only: label + count + hashes)."""
    if not os.path.exists(path):
        raise ProtocolGuardError(
            f"required hash manifest not found: {path}. The protected-set manifest "
            "must be established and approved before training (Review Gate 2 item)."
        )
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict) or not isinstance(payload.get("hashes"), list):
        raise ProtocolGuardError(f"malformed hash manifest: {path} (expected a 'hashes' list)")
    label = payload.get("label") or default_label
    return {"label": label, "count": len(payload["hashes"]), "hashes": list(payload["hashes"])}


def generate_campaign_splits(
    protected_manifest: Dict[str, Any],
) -> Tuple[SplitDataset, SplitDataset, SplitDataset]:
    """Generates and verifies the registered TRAIN-MASKED / VAL-SEEN / PROXY-DEV pools.

    Cross-split and protected-set disjointness are verified via metadata-only SHA-256
    hash-manifest comparisons (zero overlap required); protected examples are never
    streamed, browsed, or inspected.
    """
    manager = Phase16SplitManager(Phase16RegisterTask())
    train_set, val_set = manager.create_masked_campaign_splits(
        train_size=TRAIN_SIZE, val_size=VAL_SIZE
    )
    proxy_set = manager.create_proxy_dev_split(per_class=PROXY_PER_CLASS)

    manifests = {
        "TRAIN-MASKED": build_hash_manifest(train_set.hashes, "TRAIN-MASKED"),
        "VAL-SEEN": build_hash_manifest(val_set.hashes, "VAL-SEEN"),
        "PROXY-DEV": build_hash_manifest(proxy_set.hashes, "PROXY-DEV"),
        "TEST-PROTECTED": protected_manifest,
    }
    for left, right in (
        ("TRAIN-MASKED", "VAL-SEEN"),
        ("TRAIN-MASKED", "PROXY-DEV"),
        ("VAL-SEEN", "PROXY-DEV"),
        ("TRAIN-MASKED", "TEST-PROTECTED"),
        ("VAL-SEEN", "TEST-PROTECTED"),
        ("PROXY-DEV", "TEST-PROTECTED"),
    ):
        assert_manifests_disjoint(manifests[left], manifests[right])
    return train_set, val_set, proxy_set


def run_profile_preflight(args: argparse.Namespace) -> int:
    """Registered 20-run profiling stop-gate; continues to 3,000 updates under the cap."""
    require_stage_authorized(args.stage_authorized)
    if not args.protected_hash_manifest:
        raise ProtocolGuardError(
            "training requires --protected-hash-manifest (metadata-only protected-set "
            "disjointness check; Review Gate 2 item before training)"
        )
    protected_manifest = load_manifest_file(args.protected_hash_manifest, "TEST-PROTECTED")

    device = resolve_device(args.device)
    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    paths = resolve_campaign_paths(args.output_dir, timestamp)
    run_artifacts = {
        (model_id, seed): resolve_run_artifacts(paths["run_dir"], model_id, seed)
        for model_id in CANDIDATE_IDS
        for seed in REGISTERED_SEEDS
    }
    check_artifact_targets(
        [paths["profile"], paths["summary"]]
        + [
            artifact[key]
            for artifact in run_artifacts.values()
            for key in ("checkpoint", "telemetry")
        ],
        args.allow_overwrite_authorized,
    )
    os.makedirs(paths["run_dir"], exist_ok=True)

    train_set, val_set, _proxy_set = generate_campaign_splits(protected_manifest)
    campaign_start = time.perf_counter()

    runs: Dict[Tuple[str, int], CampaignRun] = {}
    for model_id in CANDIDATE_IDS:
        for seed in REGISTERED_SEEDS:
            runs[(model_id, seed)] = CampaignRun(model_id, seed, device)

    profile_seconds: Dict[str, float] = {}
    for (model_id, seed), run in runs.items():
        run_start = time.perf_counter()
        advance_run(run, train_set, val_set, PROFILE_UPDATES)
        profile_seconds[f"{model_id}:{seed}"] = time.perf_counter() - run_start

    projected_seconds = projected_campaign_seconds(profile_seconds)
    projected_hours = projected_seconds / 3600.0
    stop_after_profiling = projected_hours >= RUNTIME_CAP_HOURS
    write_json_atomic(
        paths["profile"],
        {
            "protocol": "model_e_heldout_campaign",
            "mode": "profile_preflight",
            "profile_updates_per_run": PROFILE_UPDATES,
            "registered_runs": len(profile_seconds),
            "per_run_seconds": profile_seconds,
            "projection_factor": TOTAL_UPDATES // PROFILE_UPDATES,
            "projected_seconds": projected_seconds,
            "projected_hours": projected_hours,
            "runtime_cap_hours": RUNTIME_CAP_HOURS,
            "verdict": "stop_after_profiling" if stop_after_profiling else "continue",
        },
    )

    if stop_after_profiling:
        write_json_atomic(
            paths["summary"],
            {
                "status": "stopped_after_profiling",
                "reason": (
                    f"projected campaign duration {projected_hours:.3f} h is >= "
                    f"{RUNTIME_CAP_HOURS} h; escalate to the PI without altering the "
                    "registered schedule"
                ),
                "registered_runs": len(profile_seconds),
            },
        )
        print(f"STOP after profiling: projected {projected_hours:.3f} h >= {RUNTIME_CAP_HOURS} h")
        return 2

    deadline = campaign_start + RUNTIME_CAP_HOURS * 3600.0
    status = "completed"
    try:
        for run in runs.values():
            advance_run(run, train_set, val_set, TOTAL_UPDATES, deadline=deadline)
    except CampaignRuntimeCapExceeded as exc:
        status = "hard_cap_stopped"
        print(f"STOP: {exc}")

    summary_runs: Dict[str, Any] = {}
    for (model_id, seed), run in runs.items():
        artifacts = run_artifacts[(model_id, seed)]
        if run.best_state is not None:
            write_checkpoint_atomic(
                artifacts["checkpoint"],
                run.best_state,
                {
                    "runner": "run_model_e_heldout_campaign",
                    "model_id": model_id,
                    "seed": seed,
                    "protocol": "phase_1_6_amendment_2026-10-18",
                    "parameter_count": run.model.count_parameters(),
                    "optimizer": "AdamW",
                    "learning_rate": LEARNING_RATE,
                    "weight_decay": WEIGHT_DECAY,
                    "batch_size": BATCH_SIZE,
                    "updates": run.update,
                    "selected_update": run.best_update,
                    "selected_val_loss": run.best_val_loss,
                },
            )
        telemetry = {
            "protocol": "model_e_heldout_campaign",
            "model_id": model_id,
            "seed": seed,
            "parameter_count": run.model.count_parameters(),
            "hyperparameters": {
                "optimizer": "AdamW",
                "learning_rate": LEARNING_RATE,
                "weight_decay": WEIGHT_DECAY,
                "batch_size": BATCH_SIZE,
                "total_updates": TOTAL_UPDATES,
                "validation_interval": VALIDATION_INTERVAL,
                "checkpoint_selection": "min_val_loss_earliest_update_tie_break",
            },
            "updates_completed": run.update,
            "run_status": "completed" if run.update == TOTAL_UPDATES else "incomplete_hard_cap",
            "selection": {"selected_update": run.best_update, "selected_val_loss": run.best_val_loss},
            "checkpoints": run.telemetry,
            "runtime_seconds": time.perf_counter() - run.start_time,
            "checkpoint_sha256": (
                sha256_file(artifacts["checkpoint"]) if os.path.exists(artifacts["checkpoint"]) else None
            ),
        }
        write_json_atomic(artifacts["telemetry"], telemetry)
        summary_runs[f"{model_id}:{seed}"] = {
            "updates_completed": run.update,
            "selected_update": run.best_update,
            "selected_val_loss": run.best_val_loss,
            "runtime_seconds": telemetry["runtime_seconds"],
            "status": telemetry["run_status"],
        }

    write_json_atomic(
        paths["summary"],
        {
            "status": status,
            "projected_hours": projected_hours,
            "runtime_cap_hours": RUNTIME_CAP_HOURS,
            "runs": summary_runs,
        },
    )
    print(f"campaign {status}: {len(runs)} registered runs; profile artifact at {paths['profile']}")
    return 0 if status == "completed" else 3


def evaluate_selected_checkpoints(
    run_dir: str,
    split: SplitDataset,
    task: Phase16RegisterTask,
    eligible_source_values: Tuple[int, ...],
    device: torch.device,
) -> Dict[str, Any]:
    """Evaluates the four models' selected checkpoints on one split (single pass)."""
    eligible = collect_eligible_add_transitions(task, split.inputs, eligible_source_values)
    results: Dict[str, Any] = {}
    for model_id in CANDIDATE_IDS:
        per_seed: List[Dict[str, Any]] = []
        for seed in REGISTERED_SEEDS:
            artifacts = resolve_run_artifacts(run_dir, model_id, seed)
            if not os.path.exists(artifacts["checkpoint"]):
                raise ProtocolGuardError(
                    f"selected checkpoint missing for model {model_id} seed {seed}: "
                    f"{artifacts['checkpoint']}"
                )
            checkpoint = torch.load(artifacts["checkpoint"], map_location="cpu", weights_only=True)
            model = build_candidate(model_id)
            model.load_state_dict(checkpoint["state_dict"])
            model.to(device).eval()

            predicted: List[int] = []
            targets: List[int] = []
            em_total = 0.0
            with torch.no_grad():
                for start in range(0, len(split), BATCH_SIZE):
                    indices = list(range(start, min(start + BATCH_SIZE, len(split))))
                    inputs = split.inputs[indices].to(device)
                    output = model(inputs)
                    terminal_predictions = output.step_logits.argmax(dim=-1)[:, -1]
                    em_total += (
                        terminal_sequence_exact_match(
                            terminal_predictions.cpu(), split.step_targets[indices, -1]
                        )
                        * len(indices)
                    )
                    for local, global_index in enumerate(indices):
                        for step, reg_dest in eligible[global_index]:
                            predicted.append(int(output.step_logits[local, step, reg_dest].argmax()))
                            targets.append(int(split.step_targets[global_index, step, reg_dest]))

            accuracy = heldout_add_balanced_accuracy(predicted, targets)
            per_seed.append(
                {
                    "seed": seed,
                    "acc_heldout": accuracy["macro_recall"],
                    "em_terminal": em_total / len(split),
                    "eligible_transitions": accuracy,
                    "checkpoint_sha256": sha256_file(artifacts["checkpoint"]),
                }
            )
        results[model_id] = {"seeds": per_seed}
    return results


def run_tier1(args: argparse.Namespace) -> int:
    """Tier 1: evaluate all four frozen models on PROXY-DEV; gate on Model A."""
    require_stage_authorized(args.stage_authorized)
    if not args.run_dir:
        raise ProtocolGuardError("--run-dir is required for evaluation modes")
    if not os.path.isdir(args.run_dir):
        raise ProtocolGuardError(f"run directory not found: {args.run_dir}")

    tier1_path = os.path.join(args.run_dir, "tier1_results.json")
    check_artifact_targets([tier1_path], args.allow_overwrite_authorized)

    device = resolve_device(args.device)
    task = Phase16RegisterTask()
    proxy_set = Phase16SplitManager(task).create_proxy_dev_split()
    results = evaluate_selected_checkpoints(
        args.run_dir, proxy_set, task, PROXY_ELIGIBLE_SOURCE_VALUES, device
    )
    gate = classify_tier1_gate(
        [
            {"seed": entry["seed"], "acc": entry["acc_heldout"], "em": entry["em_terminal"]}
            for entry in results["A"]["seeds"]
        ]
    )
    payload = {
        "protocol": "model_e_heldout_campaign",
        "mode": "tier1_proxy_dev",
        "split": "PROXY-DEV",
        "eligible_source_values": list(PROXY_ELIGIBLE_SOURCE_VALUES),
        "models_evaluated": list(CANDIDATE_IDS),
        "results": results,
        "tier1_gate": gate,
    }
    write_json_atomic(tier1_path, payload)
    print(f"Tier 1 verdict: {gate['verdict']} (n_pass={gate['n_pass']}, n_reject={gate['n_reject']})")
    return 0 if gate["verdict"] == "tier1_pass" else 1


def run_tier2(args: argparse.Namespace) -> int:
    """Tier 2: single pass over TEST-PROTECTED for all four frozen models."""
    require_stage_authorized(args.stage_authorized)
    if not args.run_dir:
        raise ProtocolGuardError("--run-dir is required for evaluation modes")
    tier1_path = os.path.join(args.run_dir, "tier1_results.json")
    if not os.path.exists(tier1_path):
        raise ProtocolGuardError(
            "Tier 1 results not found; the protected split cannot be unlocked without a "
            "recorded tier1_pass verdict"
        )
    with open(tier1_path, encoding="utf-8") as handle:
        tier1_payload = json.load(handle)
    tier1_verdict = tier1_payload.get("tier1_gate", {}).get("verdict")

    tier2_path = os.path.join(args.run_dir, "tier2_results.json")
    if os.path.exists(tier2_path):
        raise ProtocolGuardError(
            "protected evaluation artifact already exists; the single authorized pass has "
            "been consumed and Tier 2 is never re-run"
        )

    device = resolve_device(args.device)
    task = Phase16RegisterTask()
    protected_set = load_protected_split(args.tier2_authorized, tier1_verdict)
    results = evaluate_selected_checkpoints(
        args.run_dir, protected_set, task, PROTECTED_ELIGIBLE_SOURCE_VALUES, device
    )
    payload = {
        "protocol": "model_e_heldout_campaign",
        "mode": "tier2_protected",
        "split": "TEST-PROTECTED",
        "eligible_source_values": list(PROTECTED_ELIGIBLE_SOURCE_VALUES),
        "models_evaluated": list(CANDIDATE_IDS),
        "historical_model_e_evaluated": False,
        "tier1_verdict": tier1_verdict,
        "single_pass": True,
        "results": results,
    }
    write_json_atomic(tier2_path, payload)
    print(f"Tier 2 protected evaluation recorded at {tier2_path}")
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Model E held-out generalization campaign runner (registered Phase 1.6 "
            "amendment 2026-10-18). Execution gated: training requires "
            "--profile-preflight --stage-authorized --protected-hash-manifest."
        )
    )
    parser.add_argument("--profile-preflight", action="store_true", help="run the registered 20-run profiling stop-gate and continue under the 12.0-hour cap")
    parser.add_argument("--evaluate-tier1", action="store_true", help="evaluate the four frozen models on PROXY-DEV (Tier 1)")
    parser.add_argument("--evaluate-tier2", action="store_true", help="single protected pass over TEST-PROTECTED (requires --tier2-authorized)")
    parser.add_argument("--stage-authorized", action="store_true", help="explicit PI authorization for execution")
    parser.add_argument("--tier2-authorized", action="store_true", help="explicit PI authorization for the single protected pass")
    parser.add_argument("--run-dir", default=None, help="timestamped campaign directory for evaluation modes")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="root directory for timestamped campaign runs")
    parser.add_argument("--protected-hash-manifest", default=None, help="metadata-only SHA-256 manifest of the sealed protected split (required before training)")
    parser.add_argument("--allow-overwrite-authorized", action="store_true", help="authorize replacing existing artifacts (never for Tier 2)")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    modes = [args.profile_preflight, args.evaluate_tier1, args.evaluate_tier2]
    if sum(bool(mode) for mode in modes) != 1:
        print(
            "Registered protocol (Phase 1.6 amendment 2026-10-18): 4 candidates x seeds 42-46, "
            "3,000 updates, batch 64, min-VAL-SEEN checkpoint selection, Tier 1 PROXY-DEV gate, "
            "Tier 2 single protected pass, 12.0-hour runtime cap. "
            "No mode selected; nothing executed."
        )
        return 1
    try:
        if args.profile_preflight:
            return run_profile_preflight(args)
        if args.evaluate_tier1:
            return run_tier1(args)
        return run_tier2(args)
    except (ProtocolGuardError, FileExistsError) as exc:
        print(f"ERROR: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
