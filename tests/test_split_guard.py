from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from src.orchestrator import (
    TUNING_STAGES,
    SplitGuardError,
    compute_config_hash,
    freeze_config,
    guard_stage,
    record_stage_result,
)


def _sample_config(frozen: bool = False) -> dict:
    return {
        "data": {"root": "data/nfcorpus"},
        "rrf": {"k": 60, "depth": 100},
        "frozen": frozen,
    }


def _write_validation_dev(path: Path, systems: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["system", "ndcg10_dev", "timestamp"])
        writer.writeheader()
        for system in systems:
            writer.writerow(
                {
                    "system": system,
                    "ndcg10_dev": "0.270000",
                    "timestamp": "2026-11-15T12:00:00+00:00",
                }
            )


def test_tuning_stages_reject_test_split(tmp_path: Path):
    cfg = _sample_config(frozen=False)
    for stage in sorted(TUNING_STAGES):
        with pytest.raises(SplitGuardError):
            guard_stage(stage, split="test", config=cfg, artifacts_dir=tmp_path)


def test_tuning_stage_allows_dev_split(tmp_path: Path):
    cfg = _sample_config(frozen=False)
    guard_stage("rrf", split="dev", config=cfg, artifacts_dir=tmp_path)


def test_test_stage_requires_frozen_and_validations(tmp_path: Path):
    val_csv = tmp_path / "dev" / "validation_dev.csv"
    _write_validation_dev(val_csv, ["semantic", "reranking"])

    with pytest.raises(SplitGuardError, match="frozen"):
        guard_stage("test", split="test", config=_sample_config(frozen=False), artifacts_dir=tmp_path)

    _write_validation_dev(val_csv, ["semantic"])
    with pytest.raises(SplitGuardError, match="reranking"):
        guard_stage("test", split="test", config=_sample_config(frozen=True), artifacts_dir=tmp_path)


def test_test_run_lock_blocks_different_config_hash_unless_forced(tmp_path: Path):
    val_csv = tmp_path / "dev" / "validation_dev.csv"
    _write_validation_dev(val_csv, ["semantic", "reranking"])

    cfg1 = _sample_config(frozen=True)
    guard_stage("test", split="test", config=cfg1, artifacts_dir=tmp_path)
    lock_path = tmp_path / "test_run.lock"
    assert lock_path.exists()

    cfg2 = _sample_config(frozen=True)
    cfg2["rrf"]["k"] = 100
    assert compute_config_hash(cfg1) != compute_config_hash(cfg2)

    with pytest.raises(SplitGuardError, match="test_run.lock"):
        guard_stage("test", split="test", config=cfg2, artifacts_dir=tmp_path)

    with pytest.raises(SplitGuardError, match="reason"):
        guard_stage("test", split="test", config=cfg2, artifacts_dir=tmp_path, force=True, reason="")

    guard_stage(
        "test",
        split="test",
        config=cfg2,
        artifacts_dir=tmp_path,
        force=True,
        reason="Correcao de bug documentada",
    )
    freeze_log = json.loads((tmp_path / "freeze_log.json").read_text(encoding="utf-8"))
    assert any(entry.get("force_reason") == "Correcao de bug documentada" for entry in freeze_log)


def test_freeze_config_and_record_stage_result(tmp_path: Path):
    cfg = _sample_config(frozen=False)
    out_file = tmp_path / "dev" / "rrf_k_dev.csv"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text("k,ndcg10_dev,n_queries\n60,0.3,324\n", encoding="utf-8")

    entry = freeze_config(cfg, artifacts_dir=tmp_path)
    assert len(entry["config_hash"]) == 64

    record_stage_result("rrf", out_file, config=cfg, artifacts_dir=tmp_path)
    log = json.loads((tmp_path / "freeze_log.json").read_text(encoding="utf-8"))
    assert log[-1]["stage_results"]["rrf"]["file"] == str(out_file)
