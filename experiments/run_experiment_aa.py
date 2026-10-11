#!/usr/bin/env python3
"""Experiment AA — TCN Optimization-Recipe Diagnostic (collector + mandatory implementation audit).

Z (Z-B) found the causal TCN does not fit the directly observable `op_dest` target under the fixed
protocol while the GRU control fits it to 1.000. AA asks whether a **predeclared optimizer grid**
(learning rate x update budget) lets the existing TCN implementation fit `op_dest`, or whether the
TCN remains unqualified under the tested conditions.

This collector does NOT train. It:
  * reuses Z's exact `op_dest` dataset (which reuses Y's fixed 256-example subset per seed, canonical
    VAL/TEST, and provenance) — copied verbatim, byte-verified;
  * verifies artifact hashes, split identity/disjointness, `op_dest` class coverage and checkpoint
    hashes, failing explicitly on any invariant violation;
  * runs the **mandatory non-training TCN implementation audit** (§3): output-at-final-position
    derivation, causal padding/trimming alignment through every layer, a computational path from the
    current timestep's `x_t[3:7]` to the final logits, a finite nontrivial autograd input gradient
    at those coordinates, and a controlled change to those coordinates affecting the output.

Measurement-only; frozen E/F checkpoints unchanged; no RE:Reason modification; no Phase 2. The
readers are separate analysis-only models. `op_dest` is a LABEL only.

Outputs
-------
    experiments/experiment_aa_optimizer_grid.npz        Z's op_dest arrays (verbatim) + audit
    experiments/experiment_aa_optimizer_grid_meta.json  identities, audit results, class counts
Reproduce
---------
    PYTHONPATH=src python experiments/run_experiment_aa.py \
        --z-npz experiments/experiment_z_op_dest.npz \
        --output experiments/experiment_aa_optimizer_grid.npz
"""

import argparse
import hashlib
import json
import os
import sys
import time
from typing import Any, Dict, List

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import analyze_experiment_v as av  # noqa: E402
import run_experiment_u as ru  # noqa: E402
import run_experiment_y as ry  # noqa: E402
import run_experiment_z as rz  # noqa: E402  (op_dest helpers: class counts, recoverability)

MODEL_KEYS = ["E_candidate", "F_combined"]
NCLASS = 4
SUBSET_PREFIX = "sub256"
T_MAX = ru.T_MAX
OBS_DIM = ru.OBS_DIM
DEST_SLICE = slice(3, 7)   # reg_a block of the 21-dim observation


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
# mandatory non-training TCN implementation audit
# ---------------------------------------------------------------------------

def _tcn_prehead(model, x):
    """Run the causal conv stack; return the pre-head representation (n, channels, L)."""
    import torch
    z = x.transpose(1, 2)
    with torch.no_grad():
        for conv in model.convs:
            z = conv(z)
            trim = conv.dilation[0] * (conv.kernel_size[0] - 1)
            if trim > 0:
                z = z[:, :, :-trim]
            z = model.act(z)
    return z


def audit_tcn(arch: Dict[str, Any], X: np.ndarray, name: str, seed: int = 0) -> Dict[str, Any]:
    """Non-training validity audit of the existing TCN forward/backward path.

    Distinguishes: (a) an absent computational path, (b) numerical/gradient failure, (c) a merely
    untrained model. Does NOT require every class/sample to have a nonzero gradient.
    """
    import torch
    torch.manual_seed(seed)
    model = av.build_reader("TCN", NCLASS, arch)
    model.eval()
    Xt = torch.tensor(np.asarray(X, dtype=np.float32))
    n, L, F = Xt.shape
    out: Dict[str, Any] = {"name": name, "n": int(n), "L": int(L), "n_features": int(F)}

    # (1) every causal conv+trim preserves the temporal length (alignment through the stack)
    lengths = []
    z = Xt.transpose(1, 2)
    for conv in model.convs:
        z = conv(z)
        trim = conv.dilation[0] * (conv.kernel_size[0] - 1)
        if trim > 0:
            z = z[:, :, :-trim]
        lengths.append(int(z.shape[-1]))
    out["layer_lengths"] = lengths
    out["layer_alignment"] = bool(all(v == L for v in lengths))

    # (2) output at position i depends only on inputs at positions <= i (no future leakage)
    pre = _tcn_prehead(model, Xt)
    causal_ok = True
    rng = np.random.default_rng(0)
    for i in range(L - 1):
        mod = Xt.clone()
        if i + 1 < L:
            mod[:, i + 1:, :] = torch.tensor(
                rng.standard_normal((n, L - i - 1, F)), dtype=torch.float32)
        pre_mod = _tcn_prehead(model, mod)
        if not torch.allclose(pre[:, :, i], pre_mod[:, :, i], atol=1e-5):
            causal_ok = False
            break
    out["causal_alignment"] = bool(causal_ok)

    # (3) the final logits depend on the final input timestep
    with torch.no_grad():
        logits = model(Xt)
        mod = Xt.clone()
        mod[:, -1, :] = torch.tensor(rng.standard_normal((n, F)), dtype=torch.float32)
        logits_mod = model(mod)
    out["final_step_effect"] = bool(not torch.allclose(logits, logits_mod, atol=1e-6))
    out["logits_finite"] = bool(torch.isfinite(logits).all())

    # (4) finite nontrivial autograd input gradient at the current step's destination coords x_t[3:7]
    Xg = Xt.clone().requires_grad_(True)
    logits_g = model(Xg)
    # scalar objective independent of class priors: sum of per-row max logit
    obj = logits_g.max(dim=1).values.sum()
    obj.backward()
    g = Xg.grad
    gd = g[:, -1, DEST_SLICE]
    out["grad_finite"] = bool(torch.isfinite(g).all())
    out["dest_grad_abs_max"] = float(gd.abs().max().item())
    out["dest_grad_abs_sum"] = float(gd.abs().sum().item())
    out["dest_grad_nonzero_rows"] = int((gd.abs().sum(dim=1) > 0).sum().item())
    out["dest_grad_path"] = bool(out["grad_finite"] and out["dest_grad_abs_sum"] > 0)

    # (5) a controlled change to the destination coords of the current step affects the output
    with torch.no_grad():
        Xc = Xt.clone()
        # move the reg_a one-hot to a different register (deterministic non-identity remap)
        reg = Xc[:, -1, DEST_SLICE].argmax(dim=1)
        Xc[:, -1, DEST_SLICE] = 0.0
        newreg = (reg + 1) % NCLASS
        Xc[torch.arange(n), -1, 3 + newreg] = 1.0
        logits_c = model(Xc)
    out["dest_change_effect"] = bool(not torch.allclose(logits, logits_c, atol=1e-6))

    out["passed"] = bool(out["layer_alignment"] and out["causal_alignment"]
                         and out["final_step_effect"] and out["logits_finite"]
                         and out["dest_grad_path"] and out["dest_change_effect"])
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Experiment AA collector + TCN implementation audit")
    ap.add_argument("--z-npz", default="experiments/experiment_z_op_dest.npz")
    ap.add_argument("--y-npz", default="experiments/experiment_y_fitting_control.npz")
    ap.add_argument("--w-npz", default="experiments/experiment_w_data_scale.npz")
    ap.add_argument("--checkpoints-dir", default="experiments/checkpoints")
    ap.add_argument("--output", default="experiments/experiment_aa_optimizer_grid.npz")
    ap.add_argument("--meta-output", default="experiments/experiment_aa_optimizer_grid_meta.json")
    args = ap.parse_args()
    t0 = time.perf_counter()

    if not os.path.exists(args.z_npz):
        raise SystemExit(f"Z artifact not found: {args.z_npz}. Run run_experiment_z.py first.")
    zz = np.load(args.z_npz, allow_pickle=False)
    z_meta = json.loads(str(zz["meta_json"][0]))
    top = int(z_meta["source"]["top_size"])

    # Z must itself be pinned to verified Y/W sources.
    y_sha = sha256_file(args.y_npz)
    w_sha = sha256_file(args.w_npz)
    if y_sha != z_meta["source"]["y_sha256"]:
        raise SystemExit(f"Y source SHA mismatch: Z recorded {z_meta['source']['y_sha256']} vs {y_sha}")
    if w_sha != z_meta["source"]["w_sha256"]:
        raise SystemExit(f"W source SHA mismatch: Z recorded {z_meta['source']['w_sha256']} vs {w_sha}")

    meta: Dict[str, Any] = {
        "experiment": "AA — TCN optimization-recipe diagnostic on the op_dest control",
        "target": "op_dest", "n_classes": NCLASS,
        "readers": ["TCN", "GRU"], "histories": ["L4", "Lfull"],
        "subset_size": int(z_meta["subset_size"]), "subset_seed": int(z_meta["subset_seed"]),
        "source": {
            "z_npz": args.z_npz, "z_sha256": sha256_file(args.z_npz),
            "y_npz": args.y_npz, "y_sha256": y_sha,
            "w_npz": args.w_npz, "w_sha256": w_sha,
            "train_sizes": z_meta["source"]["train_sizes"], "top_size": top,
        },
        "identity": {}, "class_counts": {}, "recoverability": {}, "disjointness": {},
        "audit": {}, "models": {},
    }

    # ---- copy Z's arrays verbatim (byte identity) ----
    payload: Dict[str, np.ndarray] = {}
    for k in zz.files:
        if k == "meta_json":
            continue
        payload[k] = zz[k]
        assert np.array_equal(payload[k], zz[k]), f"copy mismatch {k}"
        meta["identity"][k] = {"sha256": sha256_array(payload[k]), "shape": list(payload[k].shape),
                               "dtype": str(payload[k].dtype)}

    # ---- class coverage (op_dest) ----
    meta["class_counts"] = {"train": {}, "val": {}, "test": {}}
    seeds = sorted(int(s) for s in np.unique(payload[f"{SUBSET_PREFIX}__seed"]))
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

    # ---- recoverability / disjointness ----
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

    # ---- mandatory TCN implementation audit (non-training) ----
    arch = av.architecture()
    for rep, L in (("L4", 4), ("Lfull", T_MAX)):
        rows = np.nonzero(payload[f"{SUBSET_PREFIX}__seed"] == 42)[0][:64]
        X = av.reader_input(payload[f"{SUBSET_PREFIX}__seq_obs"][rows],
                            payload[f"{SUBSET_PREFIX}__t"][rows], L)
        meta["audit"][rep] = audit_tcn(arch, X, name=f"TCN_subset{SUBSET_PREFIX}_{rep}")
    meta["audit_pass"] = bool(all(a["passed"] for a in meta["audit"].values()))
    if not meta["audit_pass"]:
        failed = [k for k, a in meta["audit"].items() if not a["passed"]]
        print(f"WARNING: TCN implementation audit FAILED for {failed}; AA will be AA-INVALID")

    # ---- checkpoint hashes ----
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
    print(f"audit_pass={meta['audit_pass']}")
    for rep, a in meta["audit"].items():
        print(f"  audit {rep}: passed={a['passed']} layer_align={a['layer_alignment']} "
              f"causal={a['causal_alignment']} final_step={a['final_step_effect']} "
              f"dest_grad_path={a['dest_grad_path']} (|g|sum={a['dest_grad_abs_sum']:.3e}) "
              f"dest_change={a['dest_change_effect']}")
    print(f"class_counts train pooled={meta['class_counts']['train']['pooled']}")
    print(f"disjointness={meta['disjointness']}")
    print(f"Wrote {args.output} and {args.meta_output} in {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
