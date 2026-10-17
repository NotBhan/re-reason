"""Unit tests for the guarded Milestone 3B training runner infrastructure (no training executed)."""

import inspect
import json
import os

import pytest
import torch

import experiments.run_phase3b_training as runner
from experiments.run_phase3b_training import (
    DEFAULT_OUTPUT_DIR,
    check_artifact_targets,
    resolve_run_artifacts,
    write_checkpoint_atomic,
    write_json_atomic,
)


def test_artifact_targets_refuse_overwrite_by_default(tmp_path) -> None:
    target = tmp_path / "telemetry.json"
    target.write_text("original", encoding="utf-8")

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        check_artifact_targets([str(target)], allow_overwrite=False)

    assert target.read_text(encoding="utf-8") == "original"


def test_authorized_overwrite_emits_warning(tmp_path, capsys) -> None:
    existing = tmp_path / "telemetry.json"
    existing.write_text("original", encoding="utf-8")
    new_path = tmp_path / "new.json"

    overwritten = check_artifact_targets(
        [str(existing), str(new_path)], allow_overwrite=True
    )

    assert overwritten == [str(existing)]
    assert "WARNING" in capsys.readouterr().out
    assert existing.read_text(encoding="utf-8") == "original"


def test_json_write_is_atomic_and_failure_safe(tmp_path) -> None:
    target = tmp_path / "artifact.json"
    write_json_atomic(str(target), {"value": 3})
    assert json.loads(target.read_text(encoding="utf-8")) == {"value": 3}

    failing = tmp_path / "failing.json"
    with pytest.raises(TypeError):
        write_json_atomic(str(failing), {"bad": object()})
    assert not failing.exists()
    assert not list(tmp_path.glob(".tmp-*"))


def test_checkpoint_write_is_atomic_and_failure_safe(tmp_path) -> None:
    target = tmp_path / "checkpoint.pt"
    state = {"weight": torch.zeros(2, 3)}
    write_checkpoint_atomic(str(target), state, {"seed": 1})

    payload = torch.load(target, weights_only=False)
    assert torch.equal(payload["state_dict"]["weight"], state["weight"])
    assert payload["metadata"] == {"seed": 1}

    failing = tmp_path / "failing.pt"
    with pytest.raises(Exception):
        write_checkpoint_atomic(str(failing), {"bad": lambda value: value}, {})
    assert not failing.exists()
    assert not list(tmp_path.glob(".tmp-*"))


def test_resolve_run_artifacts_are_unique_and_run_specific() -> None:
    first = resolve_run_artifacts(1, "bc", "20260101T000000Z", "runs")
    second = resolve_run_artifacts(1, "bc", "20260101T000001Z", "runs")
    third = resolve_run_artifacts(2, "bc", "20260101T000000Z", "runs")

    assert first["checkpoint"] != second["checkpoint"]
    assert first["telemetry"] != third["telemetry"]
    assert "seed1" in first["checkpoint"]
    assert "20260101T000000Z" in first["checkpoint"]
    assert first["checkpoint"].startswith("runs" + os.sep)
    assert first["checkpoint"].endswith(".pt")
    assert first["telemetry"].endswith(".json")

    default_artifacts = resolve_run_artifacts(1, "bc", "20260101T000000Z")
    assert default_artifacts["checkpoint"].startswith(DEFAULT_OUTPUT_DIR + os.sep)

    with pytest.raises(ValueError):
        resolve_run_artifacts(1, "", "20260101T000000Z")
    with pytest.raises(ValueError):
        resolve_run_artifacts(0, "bc", "20260101T000000Z")


def test_parser_exposes_authorization_and_overwrite_flags() -> None:
    parser = runner.build_argument_parser()
    options = parser.parse_args(
        [
            "--seed",
            "1",
            "--epochs",
            "1",
            "--batch-size",
            "2",
            "--learning-rate",
            "0.001",
            "--weight-decay",
            "0.01",
            "--gradient-clip-norm",
            "1.0",
            "--stage-b-authorized",
            "--allow-overwrite-authorized",
        ]
    )
    assert options.stage_b_authorized is True
    assert options.allow_overwrite_authorized is True


def test_main_requires_stage_b_authorization(tmp_path, capsys) -> None:
    with pytest.raises(SystemExit) as excinfo:
        runner.main(
            [
                "--seed",
                "1",
                "--epochs",
                "1",
                "--batch-size",
                "2",
                "--learning-rate",
                "0.001",
                "--weight-decay",
                "0.01",
                "--gradient-clip-norm",
                "1.0",
                "--output-dir",
                str(tmp_path),
            ]
        )

    assert excinfo.value.code == 2
    assert "not authorized" in capsys.readouterr().err
    assert list(tmp_path.iterdir()) == []


def test_training_loader_is_restricted_to_train(monkeypatch: pytest.MonkeyPatch) -> None:
    recorded = []

    def fake_generate_partition(config, partition, size, base_seed=42):
        recorded.append((partition, size))
        return ()

    monkeypatch.setattr(
        "re_reason.training.behavioral_cloning.generate_pathfinding_partition",
        fake_generate_partition,
    )
    from re_reason.training.behavioral_cloning import (
        load_behavioral_cloning_demonstrations,
    )

    load_behavioral_cloning_demonstrations(size=4)
    assert recorded == [("train", 4)]


def test_runner_source_has_no_evaluation_or_heldout_paths() -> None:
    source = inspect.getsource(runner)
    assert "generate_pathfinding_splits" not in source
    assert "rollout_two_trajectory" not in source
    assert "docs/reports" not in source
