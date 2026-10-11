#!/usr/bin/env python3
"""Experiment AB — TCN Optimizer and Initialization Diagnostic (collector + mandatory preflight).

Z and AA establish that the existing causal TCN cannot fit the directly observable four-class
`op_dest` target under the tested Adam learning-rate and update-budget grid, while the GRU control
fits it perfectly. AB asks whether a predeclared change in **optimizer family** or **initialization**
lets the existing TCN fit `op_dest`.

This collector does NOT train the grid. It:
  * reuses Z's exact `op_dest` dataset (Y's fixed 256-example subset + canonical VAL/TEST), copied
    verbatim and byte-verified, with source hashes, subset provenance, class coverage, split
    disjointness and frozen checkpoint hashes verified; it fails explicitly on any invariant.
  * runs the **mandatory trainable-parameter preflight** (§3): parameter-group/`requires_grad`
    registration, loss/gradient finiteness, per-stage gradient norms and zero-grad fractions, and a
    fixed small number of diagnostic optimizer steps on a disposable TCN measuring per-stage
    parameter-delta norms, relative update magnitudes, and logit/loss changes — reporting dead or
    effectively-unchanged stages.

Measurement-only; frozen E/F checkpoints unchanged; no RE:Reason modification; no Phase 2. The
readers are separate analysis-only models. The disposable preflight model is discarded and never
used in the optimizer grid.

Outputs
-------
    experiments/experiment_ab_optimizer_init.npz        Z's op_dest arrays (verbatim) + preflight
    experiments/experiment_ab_optimizer_init_meta.json  identities, preflight evidence
Reproduce
---------
    PYTHONPATH=src python experiments/run_experiment_ab.py \
        --z-npz experiments/experiment_z_op_dest.npz \
        --output experiments/experiment_ab_optimizer_init.npz
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
import analyze_experiment_v as av  # noqa: E402
import run_experiment_u as ru  # noqa: E402
import run_experiment_y as ry  # noqa: E402
import run_experiment_z as rz  # noqa: E402

MODEL_KEYS = ["E_candidate", "F_combined"]
NCLASS = 4
SUBSET_PREFIX = "sub256"
T_MAX = ru.T_MAX
OBS_DIM = ru.OBS_DIM
DEST_SLICE = slice(3, 7)


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


# ---------------------------------------------------------------------------
# mandatory trainable-parameter preflight
# ---------------------------------------------------------------------------

def iter_stages(model) -> Dict[str, List]:
    """Group a reader's parameterized stages (named (name, Parameter) pairs)."""
    out: Dict[str, List] = {}
    if hasattr(model, "convs"):                       # TCN
        for i, c in enumerate(model.convs):
            out[f"conv{i}"] = [(f"convs.{i}.weight", c.weight), (f"convs.{i}.bias", c.bias)]
        out["head"] = [("head.weight", model.head.weight), ("head.bias", model.head.bias)]
    else:                                             # GRU
        out["gru"] = [(f"gru.{n}", p) for n, p in model.gru.named_parameters()]
        out["head"] = [("head.weight", model.head.weight), ("head.bias", model.head.bias)]
    return out


def _grad_stats(stages, ) -> Dict[str, Any]:
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
                   "zero_grad_fraction": (zero / n) if n else 1.0,
                   "dead": bool(gsq == 0.0)}
    return out


def preflight_tcn(arch: Dict[str, Any], Xtr, ytr, Xva, yva, name: str,
                  lr: float = 3e-3, wd: float = 1e-2, steps: int = 20, seed: int = 0) -> Dict[str, Any]:
    """Verify the loss produces trainable-parameter gradients and optimizer updates (disposable model)."""
    import torch
    import torch.nn.functional as F
    torch.manual_seed(seed)
    model = av.build_reader("TCN", NCLASS, arch)      # existing default initialization
    model.train()
    params = list(model.parameters())
    n_params = sum(p.numel() for p in params)
    expected = av.reader_param_count("TCN", NCLASS, arch)

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    opt_ids = {id(p) for g in opt.param_groups for p in g["params"]}
    in_opt = all(id(p) in opt_ids for p in params)

    Xn = torch.tensor(np.asarray(Xtr, dtype=np.float32))
    yn = torch.tensor(ytr, dtype=torch.long)
    wc = torch.tensor(as_.class_weights(ytr, NCLASS), dtype=torch.float32)

    logits = model(Xn)
    loss = F.cross_entropy(logits, yn, weight=wc)
    loss.backward()
    loss_finite = bool(torch.isfinite(loss))
    stages = iter_stages(model)
    grad = _grad_stats(stages)
    agg_grad = math.sqrt(sum(v["grad_norm"] ** 2 for v in grad.values()))
    grad_finite = bool(all(v["grad_norm"] == v["grad_norm"] for v in grad.values()))  # no NaN

    # diagnostic optimizer steps on a disposable copy: snapshot params/logits/loss, step, re-measure
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
        "name": name, "n_trainable_params": int(n_params), "expected_params": int(expected),
        "n_param_tensors": int(len(params)), "all_in_optimizer": bool(in_opt),
        "all_requires_grad": bool(all(p.requires_grad for p in params)),
        "loss_before": loss_before, "loss_after": loss_after,
        "loss_decreased": bool(loss_after < loss_before), "loss_finite": loss_finite,
        "aggregate_grad_norm": agg_grad, "grad_finite": grad_finite,
        "per_stage_grad": grad, "per_stage_update": upd, "dead_stages": dead_stages,
        "any_parameter_updated": bool(any_update), "logit_change": logit_change,
        "diagnostic_steps": steps, "lr": lr, "weight_decay": wd, "n": int(len(ytr)),
        "passed": passed,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Experiment AB collector + trainable-parameter preflight")
    ap.add_argument("--z-npz", default="experiments/experiment_z_op_dest.npz")
    ap.add_argument("--y-npz", default="experiments/experiment_y_fitting_control.npz")
    ap.add_argument("--w-npz", default="experiments/experiment_w_data_scale.npz")
    ap.add_argument("--checkpoints-dir", default="experiments/checkpoints")
    ap.add_argument("--output", default="experiments/experiment_ab_optimizer_init.npz")
    ap.add_argument("--meta-output", default="experiments/experiment_ab_optimizer_init_meta.json")
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
        "experiment": "AB — TCN optimizer and initialization diagnostic on the op_dest control",
        "target": "op_dest", "n_classes": NCLASS,
        "readers": ["TCN", "GRU"], "histories": ["L4", "Lfull"],
        "subset_size": int(z_meta["subset_size"]), "subset_seed": int(z_meta["subset_seed"]),
        "source": {"z_npz": args.z_npz, "z_sha256": sha256_file(args.z_npz),
                   "y_npz": args.y_npz, "y_sha256": y_sha, "w_npz": args.w_npz, "w_sha256": w_sha,
                   "train_sizes": z_meta["source"]["train_sizes"], "top_size": top},
        "identity": {}, "class_counts": {}, "recoverability": {}, "disjointness": {},
        "preflight": {}, "models": {},
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
    meta["class_counts"] = {"train": {}, "val": {}, "test": {}}
    for s in seeds:
        rows = np.nonzero(payload[f"{SUBSET_PREFIX}__seed"] == s)[0]
        rz.assert_all_classes(rz.op_dest_counts(payload[f"{SUBSET_PREFIX}__op_dest"][rows]),
                              f"train seed {s}")
        rows_v = np.nonzero(payload["val__seed"] == s)[0]
        rz.assert_all_classes(rz.op_dest_counts(payload["val__op_dest"][rows_v]), f"val seed {s}")
    meta["class_counts"]["train"]["pooled"] = rz.op_dest_counts(payload[f"{SUBSET_PREFIX}__op_dest"])
    meta["class_counts"]["val"]["pooled"] = rz.op_dest_counts(payload["val__op_dest"])
    for mk in MODEL_KEYS:
        rz.assert_all_classes(rz.op_dest_counts(payload[f"{mk}__test__op_dest"]), f"{mk} test")
        meta["class_counts"]["test"][mk] = rz.op_dest_counts(payload[f"{mk}__test__op_dest"])

    meta["recoverability"]["train"] = rz.verify_direct_recoverability(
        payload[f"{SUBSET_PREFIX}__seq_obs"], payload[f"{SUBSET_PREFIX}__t"],
        payload[f"{SUBSET_PREFIX}__op_dest"], "train")
    meta["recoverability"]["val"] = rz.verify_direct_recoverability(
        payload["val__seq_obs"], payload["val__t"], payload["val__op_dest"], "val")
    for mk in MODEL_KEYS:
        meta["recoverability"][f"{mk}_test"] = rz.verify_direct_recoverability(
            payload[f"{mk}__test__seq_obs"], payload[f"{mk}__test__t"],
            payload[f"{mk}__test__op_dest"], f"{mk} test")
    sub_h = _seq_hashes(payload[f"{SUBSET_PREFIX}__seq_obs"])
    val_h = _seq_hashes(payload["val__seq_obs"])
    meta["disjointness"] = {"subset_vs_val": len(sub_h & val_h)}
    for mk in MODEL_KEYS:
        test_h = _seq_hashes(payload[f"{mk}__test__seq_obs"])
        meta["disjointness"][f"subset_vs_{mk}_test"] = len(sub_h & test_h)
        meta["disjointness"][f"val_vs_{mk}_test"] = len(val_h & test_h)
    if any(v != 0 for v in meta["disjointness"].values()):
        raise SystemExit(f"split disjointness failure: {meta['disjointness']}")

    # ---- mandatory trainable-parameter preflight (disposable model) ----
    arch = av.architecture()
    for rep, L in (("L4", 4), ("Lfull", T_MAX)):
        rows = np.nonzero(payload[f"{SUBSET_PREFIX}__seed"] == 42)[0]
        X = av.reader_input(payload[f"{SUBSET_PREFIX}__seq_obs"][rows],
                            payload[f"{SUBSET_PREFIX}__t"][rows], L)
        y = payload[f"{SUBSET_PREFIX}__op_dest"][rows].astype(int)
        meta["preflight"][rep] = preflight_tcn(arch, X, y, X, y, name=f"TCN_{rep}")
    meta["preflight_pass"] = bool(all(p["passed"] for p in meta["preflight"].values()))
    if not meta["preflight_pass"]:
        failed = [k for k, p in meta["preflight"].items() if not p["passed"]]
        print(f"WARNING: trainable-parameter preflight FAILED for {failed}; AB will be AB-INVALID")

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
    for rep, p in meta["preflight"].items():
        print(f"  preflight {rep}: passed={p['passed']} params={p['n_trainable_params']} "
              f"agg_grad={p['aggregate_grad_norm']:.3e} loss {p['loss_before']:.4f}->{p['loss_after']:.4f} "
              f"dead_stages={p['dead_stages']}")
        for st, v in p["per_stage_grad"].items():
            print(f"      {st}: grad_norm={v['grad_norm']:.3e} zero_frac={v['zero_grad_fraction']:.3f} "
                  f"delta_norm={p['per_stage_update'][st]['delta_norm']:.3e}")
    print(f"Wrote {args.output} and {args.meta_output} in {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
