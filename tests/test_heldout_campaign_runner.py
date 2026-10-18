"""Guarded campaign-runner unit tests (no training, no data generation, no checkpoint loading).

Covers the registered runner safety contract: inert default mode, fail-closed
authorization gates for training and the protected single pass, the guard that
must precede any protected-split generation, artifact overwrite refusal with
authorized-warning behavior, atomic failure-safe writes, the profiling projection
formula, and the metadata-only manifest collision checker (including a deliberate
leakage-failure path demonstrating that the guard fires).
"""

import json

import pytest
import torch

import experiments.run_model_e_heldout_campaign as runner
from re_reason.benchmarks.dataset_split import (
    assert_manifests_disjoint,
    build_hash_manifest,
    check_manifest_disjointness,
)
from re_reason.training.heldout_objective import (
    TIER1_ACC_MIN,
    TIER1_EM_MIN,
    TIER1_REJECTION_ACC,
    TIER1_REJECTION_EM,
)


def test_no_mode_is_inert(capsys) -> None:
    assert runner.main([]) == 1
    output = capsys.readouterr().out
    assert "No mode selected; nothing executed." in output
    assert "12.0-hour runtime cap" in output


def test_training_requires_stage_authorization(tmp_path, capsys) -> None:
    assert runner.main(["--profile-preflight", "--output-dir", str(tmp_path)]) == 1
    assert "--stage-authorized" in capsys.readouterr().out
    assert list(tmp_path.iterdir()) == []


def test_training_requires_protected_hash_manifest(tmp_path, capsys) -> None:
    result = runner.main(
        ["--profile-preflight", "--stage-authorized", "--output-dir", str(tmp_path)]
    )
    assert result == 1
    assert "--protected-hash-manifest" in capsys.readouterr().out
    assert list(tmp_path.iterdir()) == []


def test_missing_manifest_file_fails_closed_before_any_artifact(tmp_path, capsys) -> None:
    result = runner.main(
        [
            "--profile-preflight",
            "--stage-authorized",
            "--output-dir",
            str(tmp_path / "runs"),
            "--protected-hash-manifest",
            str(tmp_path / "missing_manifest.json"),
        ]
    )
    assert result == 1
    assert "required hash manifest not found" in capsys.readouterr().out
    assert not (tmp_path / "runs").exists()


def test_tier1_requires_stage_authorization(capsys) -> None:
    assert runner.main(["--evaluate-tier1", "--run-dir", "/nonexistent"]) == 1
    assert "--stage-authorized" in capsys.readouterr().out


def test_tier2_requires_dedicated_flag_and_writes_nothing(tmp_path, capsys) -> None:
    (tmp_path / "tier1_results.json").write_text(
        json.dumps({"tier1_gate": {"verdict": "tier1_pass"}}), encoding="utf-8"
    )
    result = runner.main(
        ["--evaluate-tier2", "--stage-authorized", "--run-dir", str(tmp_path)]
    )
    assert result == 1
    assert "--tier2-authorized" in capsys.readouterr().out
    assert not (tmp_path / "tier2_results.json").exists()


def test_protected_loader_guard_precedes_generation(monkeypatch) -> None:
    calls = []

    def canary():
        calls.append("generated")
        raise AssertionError("protected generation must never be reached without authorization")

    monkeypatch.setattr(runner, "_generate_protected_split", canary)

    with pytest.raises(runner.ProtocolGuardError, match="tier2-authorized"):
        runner.load_protected_split(False, "tier1_pass")
    with pytest.raises(runner.ProtocolGuardError, match="remains sealed"):
        runner.load_protected_split(True, "tier1_halt")
    with pytest.raises(runner.ProtocolGuardError, match="tier2-authorized"):
        runner.load_protected_split(False, None)
    assert calls == []


def test_tier2_refuses_consumed_protected_pass(tmp_path, capsys) -> None:
    (tmp_path / "tier1_results.json").write_text(
        json.dumps({"tier1_gate": {"verdict": "tier1_pass"}}), encoding="utf-8"
    )
    (tmp_path / "tier2_results.json").write_text("{}", encoding="utf-8")

    result = runner.main(
        [
            "--evaluate-tier2",
            "--stage-authorized",
            "--tier2-authorized",
            "--run-dir",
            str(tmp_path),
        ]
    )
    assert result == 1
    assert "already exists" in capsys.readouterr().out
    assert (tmp_path / "tier2_results.json").read_text(encoding="utf-8") == "{}"


def test_artifact_targets_refuse_overwrite_and_authorized_overwrite_warns(tmp_path, capsys) -> None:
    target = tmp_path / "artifact.json"
    target.write_text("original", encoding="utf-8")

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        runner.check_artifact_targets([str(target)], allow_overwrite=False)

    overwritten = runner.check_artifact_targets([str(target)], allow_overwrite=True)
    assert overwritten == [str(target)]
    assert "WARNING" in capsys.readouterr().out
    assert target.read_text(encoding="utf-8") == "original"


def test_atomic_writes_are_failure_safe(tmp_path) -> None:
    target = tmp_path / "artifact.json"
    runner.write_json_atomic(str(target), {"value": 3})
    assert json.loads(target.read_text(encoding="utf-8")) == {"value": 3}

    failing = tmp_path / "failing.json"
    with pytest.raises(TypeError):
        runner.write_json_atomic(str(failing), {"bad": object()})
    assert not failing.exists()
    assert not list(tmp_path.glob(".tmp-*"))

    checkpoint = tmp_path / "checkpoint.pt"
    runner.write_checkpoint_atomic(str(checkpoint), {"weight": torch.zeros(2, 2)}, {"seed": 42})
    loaded = torch.load(checkpoint, weights_only=True)
    assert loaded["metadata"]["seed"] == 42
    assert torch.equal(loaded["state_dict"]["weight"], torch.zeros(2, 2))


def test_projected_campaign_seconds_formula() -> None:
    profile = {
        f"{model_id}:{seed}": 1.0
        for model_id in runner.CANDIDATE_IDS
        for seed in runner.REGISTERED_SEEDS
    }
    assert len(profile) == 20
    assert runner.projected_campaign_seconds(profile) == pytest.approx(600.0)

    with pytest.raises(ValueError, match="exactly the 20 registered"):
        runner.projected_campaign_seconds({"A:42": 1.0})

    invalid = dict(profile)
    invalid["A:42"] = float("nan")
    with pytest.raises(ValueError, match="positive finite"):
        runner.projected_campaign_seconds(invalid)


def test_registered_protocol_constants() -> None:
    assert runner.CANDIDATE_IDS == ("A", "B", "C", "D")
    assert runner.REGISTERED_SEEDS == (42, 43, 44, 45, 46)
    assert runner.TOTAL_UPDATES == 3000
    assert runner.PROFILE_UPDATES == 100
    assert runner.BATCH_SIZE == 64
    assert runner.VALIDATION_INTERVAL == 50
    assert runner.RUNTIME_CAP_HOURS == 12.0
    assert runner.LEARNING_RATE == 3e-3
    assert runner.WEIGHT_DECAY == 1e-2
    assert runner.PROXY_ELIGIBLE_SOURCE_VALUES == (3,)
    assert runner.PROTECTED_ELIGIBLE_SOURCE_VALUES == (0, 5)
    assert (TIER1_ACC_MIN, TIER1_EM_MIN) == (0.800, 0.700)
    assert (TIER1_REJECTION_ACC, TIER1_REJECTION_EM) == (0.250, 0.150)


def test_manifest_disjointness_pass_and_leakage_guard() -> None:
    manifest_a = build_hash_manifest(["h1", "h2", "h3"], "TRAIN-MASKED")
    manifest_b = build_hash_manifest(["h4", "h5"], "VAL-SEEN")
    result = check_manifest_disjointness(manifest_a, manifest_b)
    assert result == {
        "pass": True,
        "overlap_count": 0,
        "label_a": "TRAIN-MASKED",
        "label_b": "VAL-SEEN",
        "count_a": 3,
        "count_b": 2,
    }
    assert_manifests_disjoint(manifest_a, manifest_b)

    leaking = build_hash_manifest(["h2", "h9"], "LEAK")
    check = check_manifest_disjointness(manifest_a, leaking)
    assert check["pass"] is False
    assert check["overlap_count"] == 1
    with pytest.raises(RuntimeError, match="split collision detected"):
        assert_manifests_disjoint(manifest_a, leaking)

    with pytest.raises(ValueError, match="duplicate sequence hashes"):
        build_hash_manifest(["h1", "h1"], "DUP")
