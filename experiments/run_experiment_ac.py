#!/usr/bin/env python3
"""Experiment AC — TCN re-qualification on the ADD `y_true` target (collector + mandatory preflight).

AB (AB-A) showed that the existing causal TCN fits the directly observable `op_dest` control under
AdamW (decoupled weight decay) or a xavier-uniform initialization under Adam, while the original
Adam/default recipe and RMSprop fail. AC asks whether either AB-qualified recipe lets the TCN fit the
**actual ADD target** `y_true = (operand_a + operand_b) mod 10` on Y's fixed 256-transition subset, and
whether any improved fitting transfers to untouched VAL/TEST populations.

This collector does NOT train the grid. It:
  * reuses Y's exact fitting subset and canonical VAL/TEST (carried verbatim in Z's artifact), copied
    byte-for-byte, with source hashes, subset provenance, 10-class coverage, the `y_true`
    derivation, held-out-pair exclusion, split disjointness and frozen checkpoint hashes verified; it
    fails explicitly on any invariant.
  * runs the **mandatory trainable-parameter preflight** (TCN subject and GRU control, both histories):
    parameter-group/`requires_grad` registration, loss/gradient finiteness, per-stage gradient norms and
    zero-grad fractions, and a fixed small number of diagnostic optimizer steps on a disposable model
    measuring per-stage parameter-delta norms, relative update magnitudes and logit/loss changes.

Measurement-only; frozen E/F checkpoints unchanged; no RE:Reason modification; no Phase 2. The readers
are separate analysis-only models. The disposable preflight models are discarded.

Outputs
-------
    experiments/experiment_ac_ytrue_requalification.npz        Y's subset + VAL + E/F TEST (verbatim)
    experiments/experiment_ac_ytrue_meta.json                 identities, invariants, preflight evidence
Reproduce
---------
    PYTHONPATH=src python experiments/run_experiment_ac.py \
        --z-npz experiments/experiment_z_op_dest.npz \
        --output experiments/experiment_ac_ytrue_requalification.npz
"""

import argparse
import hashlib
import json
import math
import os
import sys
import time
from typing import Any, Dict, List

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import analyze_experiment_s as as_  # noqa: E402
import analyze_experiment_u as au  # noqa: E402
import analyze_experiment_v as av  # noqa: E402
import run_experiment_u as ru  # noqa: E402
import run_experiment_z as rz  # noqa: E402

MODEL_KEYS = ["E_candidate", "F_combined"]
NCLASS = 10
SUBSET_PREFIX = "sub256"
T_MAX = ru.T_MAX
OBS_DIM = ru.OBS_DIM
TARGET = "y_true"
HELD_OUT_PAIRS = au.HELD_OUT_PAIRS


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_array(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def _seq_hashes(arr: np.ndarray) -> set:
    return {rz.hash_sequence(arr[i].reshape(T_MAX, OBS_DIM)) for i in range(arr.shape[0])}


def class_counts(y: np.ndarray, nclass: int = NCLASS) -> List[int]:
    return [int((y == c).sum()) for c in range(nclass)]


def assert_all_classes(counts: List[int], where: str) -> None:
    if len(counts) != NCLASS or any(c <= 0 for c in counts):
        raise SystemExit(f"class coverage failure: {where} = {counts}")


def verify_target_derivation(oa: np.ndarray, ob: np.ndarray, y: np.ndarray, where: str) -> Dict[str, Any]:
    """y_true == (operand_a + operand_b) mod 10, exactly."""
    expected = (oa.astype(np.int64) + ob.astype(np.int64)) % NCLASS
    ok = bool(np.array_equal(expected, y.astype(np.int64)))
    if not ok:
        n_bad = int((expected != y.astype(np.int64)).sum())
        raise SystemExit(f"y_true derivation failure: {where} ({n_bad} mismatches)")
    return {"n": int(len(y)), "exact": ok, "rule": "(operand_a + operand_b) mod 10"}


def verify_heldout_exclusion(ob: np.ndarray, where: str) -> Dict[str, Any]:
    """The fitting subset must exclude held-out operand pairs (b in {0,5})."""
    bad = int(np.isin(ob.astype(np.int64), [0, 5]).sum())
    if bad != 0:
        raise SystemExit(f"held-out pair exclusion failure: {where} has {bad} rows with b in {{0,5}}")
    return {"n": int(len(ob)), "n_with_b_in_{0,5}": bad, "excluded_pairs_b": [0, 5]}


# ---------------------------------------------------------------------------
# mandatory trainable-parameter preflight
# ---------------------------------------------------------------------------

def iter_stages(model, reader: str) -> Dict[str, List]:
    out: Dict[str, List] = {}
    if reader == "TCN":
        for i, c in enumerate(model.convs):
            out[f"conv{i}"] = [(f"convs.{i}.weight", c.weight), (f"convs.{i}.bias", c.bias)]
        out["head"] = [("head.weight", model.head.weight), ("head.bias", model.head.bias)]
    else:                                                     # GRU
        out["gru"] = [(f"gru.{n}", p) for n, p in model.gru.named_parameters()]
        out["head"] = [("head.weight", model.head.weight), ("head.bias", model.head.bias)]
    return out


def _grad_stats(stages) -> Dict[str, Any]:
    out = {}
    for st, plist in stages.items():
        gsq, n, zero, norms = 0.0, 0, 0, []
        for _, p in plist:
            if p.grad is None:
                continue
            g = p.grad.detach()
            gsq += float((g * g).sum())
            n += int(g.numel())
            zero += int((g == 0).sum())
            norms.append(float(g.norm()))
        out[st] = {"grad_norm": math.sqrt(gsq), "n_elements": n,
                   "zero_grad_fraction": (zero / n) if n else 1.0, "dead": bool(gsq == 0.0)}
    return out


def preflight_reader(reader: str, arch: Dict[str, Any], X, y, name: str,
                     lr: float = 3e-3, wd: float = 1e-2, steps: int = 20, seed: int = 0) -> Dict[str, Any]:
    """Verify the loss produces trainable-parameter gradients and optimizer updates (disposable model)."""
    import torch
    import torch.nn.functional as F
    torch.manual_seed(seed)
    model = av.build_reader(reader, NCLASS, arch)
    model.train()
    params = list(model.parameters())
    n_params = sum(p.numel() for p in params)
    expected = av.reader_param_count(reader, NCLASS, arch)

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    opt_ids = {id(p) for g in opt.param_groups for p in g["params"]}
    in_opt = all(id(p) in opt_ids for p in params)

    Xn = torch.tensor(np.asarray(X, dtype=np.float32))
    yn = torch.tensor(y, dtype=torch.long)
    wc = torch.tensor(as_.class_weights(y, NCLASS), dtype=torch.float32)

    loss = F.cross_entropy(model(Xn), yn, weight=wc)
    loss.backward()
    loss_finite = bool(torch.isfinite(loss))
    stages = iter_stages(model, reader)
    grad = _grad_stats(stages)
    agg_grad = math.sqrt(sum(v["grad_norm"] ** 2 for v in grad.values()))
    grad_finite = bool(all(v["grad_norm"] == v["grad_norm"] for v in grad.values()))

    flat = [(n, p) for plist in stages.values() for n, p in plist]
    before = {n: p.detach().clone() for n, p in flat}
    pnorm_before = {n: float(p.detach().norm()) for n, p in flat}
    with torch.no_grad():
        logits_before = model(Xn).detach().clone()
    loss_before = float(loss)
    opt.zero_grad()
    for _ in range(steps):
        opt.zero_grad()
        l = F.cross_entropy(model(Xn), yn, weight=wc)
        l.backward()
        opt.step()
    upd: Dict[str, Any] = {}
    any_update = False
    for st, plist in stages.items():
        d2, rel = 0.0, []
        for n, p in plist:
            d = (p.detach() - before[n])
            dn = float(d.norm())
            d2 += dn * dn
            rel.append(dn / (pnorm_before[n] + 1e-12))
            if dn > 0:
                any_update = True
        upd[st] = {"delta_norm": math.sqrt(d2), "max_relative_update": max(rel) if rel else 0.0,
                   "effectively_unchanged": bool(math.sqrt(d2) == 0.0)}
    with torch.no_grad():
        logits_after = model(Xn)
        loss_after = float(F.cross_entropy(logits_after, yn, weight=wc))
        logit_change = float((logits_after - logits_before).norm())

    dead_stages = [st for st, v in grad.items() if v["dead"]]
    passed = bool((n_params == expected) and in_opt and all(p.requires_grad for p in params)
                  and loss_finite and grad_finite and agg_grad > 0.0 and any_update)
    return {
        "name": name, "reader": reader, "n_trainable_params": int(n_params),
        "expected_params": int(expected), "n_param_tensors": int(len(params)),
        "all_in_optimizer": bool(in_opt), "all_requires_grad": bool(all(p.requires_grad for p in params)),
        "loss_before": loss_before, "loss_after": loss_after,
        "loss_decreased": bool(loss_after < loss_before), "loss_finite": loss_finite,
        "aggregate_grad_norm": agg_grad, "grad_finite": grad_finite,
        "per_stage_grad": grad, "per_stage_update": upd, "dead_stages": dead_stages,
        "any_parameter_updated": bool(any_update), "logit_change": logit_change,
        "diagnostic_steps": steps, "lr": lr, "weight_decay": wd, "n": int(len(y)),
        "passed": passed,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Experiment AC collector + trainable-parameter preflight")
    ap.add_argument("--z-npz", default="experiments/experiment_z_op_dest.npz")
    ap.add_argument("--y-npz", default="experiments/experiment_y_fitting_control.npz")
    ap.add_argument("--w-npz", default="experiments/experiment_w_data_scale.npz")
    ap.add_argument("--checkpoints-dir", default="experiments/checkpoints")
    ap.add_argument("--output", default="experiments/experiment_ac_ytrue_requalification.npz")
    ap.add_argument("--meta-output", default="experiments/experiment_ac_ytrue_meta.json")
    args = ap.parse_args()
    t0 = time.perf_counter()

    if not os.path.exists(args.z_npz):
        raise SystemExit(f"Z artifact not found: {args.z_npz}. Run run_experiment_z.py first.")
    zz = np.load(args.z_npz, allow_pickle=False)
    z_meta = json.loads(str(zz["meta_json"][0]))
    top = int(z_meta["source"]["top_size"])

    y_sha = sha256_file(args.y_npz)
    w_sha = sha256_file(args.w_npz)
    if y_sha != z_meta["source"]["y_sha256"]:
        raise SystemExit(f"Y source SHA mismatch: {z_meta['source']['y_sha256']} vs {y_sha}")
    if w_sha != z_meta["source"]["w_sha256"]:
        raise SystemExit(f"W source SHA mismatch: {z_meta['source']['w_sha256']} vs {w_sha}")

    meta: Dict[str, Any] = {
        "experiment": "AC — TCN re-qualification on the ADD y_true target",
        "target": TARGET, "n_classes": NCLASS,
        "readers": ["TCN", "GRU"], "histories": ["L4", "Lfull"],
        "subset_size": int(z_meta["subset_size"]), "subset_seed": int(z_meta["subset_seed"]),
        "held_out_pairs": sorted([list(p) for p in HELD_OUT_PAIRS]),
        "source": {"z_npz": args.z_npz, "z_sha256": sha256_file(args.z_npz),
                   "y_npz": args.y_npz, "y_sha256": y_sha, "w_npz": args.w_npz, "w_sha256": w_sha,
                   "train_sizes": z_meta["source"]["train_sizes"], "top_size": top},
        "identity": {}, "class_counts": {}, "target_derivation": {}, "heldout_exclusion": {},
        "disjointness": {}, "preflight": {}, "models": {},
    }

    payload: Dict[str, np.ndarray] = {}
    for k in zz.files:
        if k == "meta_json":
            continue
        payload[k] = zz[k]
        assert np.array_equal(payload[k], zz[k]), f"copy mismatch {k}"
        meta["identity"][k] = {"sha256": sha256_array(payload[k]), "shape": list(payload[k].shape),
                               "dtype": str(payload[k].dtype)}

    seeds = sorted(int(s) for s in np.unique(payload[f"{SUBSET_PREFIX}__seed"]))
    # ---- class coverage + target derivation + held-out exclusion ----
    meta["class_counts"] = {"train": {"per_seed": {}, "pooled": None},
                            "val": {"per_seed": {}, "pooled": None}, "test": {}}
    meta["target_derivation"] = {}
    meta["heldout_exclusion"] = {}
    for s in seeds:
        rows = np.nonzero(payload[f"{SUBSET_PREFIX}__seed"] == s)[0]
        c = class_counts(payload[f"{SUBSET_PREFIX}__{TARGET}"][rows])
        assert_all_classes(c, f"train seed {s}")
        meta["class_counts"]["train"]["per_seed"][str(s)] = c
        rows_v = np.nonzero(payload["val__seed"] == s)[0]
        assert_all_classes(class_counts(payload["val__y_true"][rows_v]), f"val seed {s}")
    meta["class_counts"]["train"]["pooled"] = class_counts(payload[f"{SUBSET_PREFIX}__{TARGET}"])
    meta["class_counts"]["val"]["pooled"] = class_counts(payload["val__y_true"])
    for mk in MODEL_KEYS:
        assert_all_classes(class_counts(payload[f"{mk}__test__{TARGET}"]), f"{mk} test")
        meta["class_counts"]["test"][mk] = class_counts(payload[f"{mk}__test__{TARGET}"])

    meta["target_derivation"]["train"] = verify_target_derivation(
        payload[f"{SUBSET_PREFIX}__operand_a"], payload[f"{SUBSET_PREFIX}__operand_b"],
        payload[f"{SUBSET_PREFIX}__{TARGET}"], "train")
    meta["target_derivation"]["val"] = verify_target_derivation(
        payload["val__operand_a"], payload["val__operand_b"], payload["val__y_true"], "val")
    for mk in MODEL_KEYS:
        meta["target_derivation"][f"{mk}_test"] = verify_target_derivation(
            payload[f"{mk}__test__operand_a"], payload[f"{mk}__test__operand_b"],
            payload[f"{mk}__test__{TARGET}"], f"{mk} test")
    meta["heldout_exclusion"]["train"] = verify_heldout_exclusion(
        payload[f"{SUBSET_PREFIX}__operand_b"], "train")

    sub_h = _seq_hashes(payload[f"{SUBSET_PREFIX}__seq_obs"])
    val_h = _seq_hashes(payload["val__seq_obs"])
    meta["disjointness"] = {"subset_vs_val": len(sub_h & val_h)}
    for mk in MODEL_KEYS:
        test_h = _seq_hashes(payload[f"{mk}__test__seq_obs"])
        meta["disjointness"][f"subset_vs_{mk}_test"] = len(sub_h & test_h)
        meta["disjointness"][f"val_vs_{mk}_test"] = len(val_h & test_h)
    if any(v != 0 for v in meta["disjointness"].values()):
        raise SystemExit(f"split disjointness failure: {meta['disjointness']}")

    # ---- mandatory trainable-parameter preflight (disposable models) ----
    arch = av.architecture()
    for reader in ("TCN", "GRU"):
        for rep, L in (("L4", 4), ("Lfull", T_MAX)):
            rows = np.nonzero(payload[f"{SUBSET_PREFIX}__seed"] == 42)[0]
            X = av.reader_input(payload[f"{SUBSET_PREFIX}__seq_obs"][rows],
                                payload[f"{SUBSET_PREFIX}__t"][rows], L)
            y = payload[f"{SUBSET_PREFIX}__{TARGET}"][rows].astype(int)
            meta["preflight"][f"{reader}_{rep}"] = preflight_reader(
                reader, arch, X, y, name=f"{reader}_{rep}")
    meta["preflight_pass"] = bool(all(p["passed"] for p in meta["preflight"].values()))
    if not meta["preflight_pass"]:
        failed = [k for k, p in meta["preflight"].items() if not p["passed"]]
        print(f"WARNING: trainable-parameter preflight FAILED for {failed}; AC will be AC-INVALID")

    for mk in MODEL_KEYS:
        meta["models"][mk] = {"checkpoints": {}}
        for s in seeds:
            p = os.path.join(args.checkpoints_dir, f"{mk}_seed{s}_3000steps.pt")
            meta["models"][mk]["checkpoints"][str(s)] = {"path": p, "sha256": sha256_file(p)}

    meta["n"] = {"subset": int(payload[f"{SUBSET_PREFIX}__y_true"].shape[0]),
                 "subset_per_seed": int(z_meta["n"]["subset_per_seed"]),
                 "val": int(payload["val__y_true"].shape[0])}
    for mk in MODEL_KEYS:
        meta["n"][f"{mk}__test"] = int(payload[f"{mk}__test__y_true"].shape[0])
    meta["date"] = time.strftime("%Y-%m-%d %H:%M:%S")
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    payload["meta_json"] = np.array([json.dumps(meta)])
    np.savez_compressed(args.output, **payload)
    with open(args.meta_output, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Copied {len(meta['identity'])} arrays verbatim from {args.z_npz}")
    print(f"preflight_pass={meta['preflight_pass']}")
    for k, p in meta["preflight"].items():
        print(f"  preflight {k}: passed={p['passed']} params={p['n_trainable_params']} "
              f"agg_grad={p['aggregate_grad_norm']:.3e} loss {p['loss_before']:.4f}->{p['loss_after']:.4f} "
              f"dead_stages={p['dead_stages']}")
    print(f"class_counts train(pooled)={meta['class_counts']['train']['pooled']} "
          f"val(pooled)={meta['class_counts']['val']['pooled']} "
          f"test(E)={meta['class_counts']['test']['E_candidate']}")
    print(f"disjointness={meta['disjointness']}")
    print(f"Wrote {args.output} and {args.meta_output} in {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
