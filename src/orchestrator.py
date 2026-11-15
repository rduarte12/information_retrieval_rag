from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import yaml

TUNING_STAGES = frozenset({"preproc", "bm25", "dense_val", "rrf", "rerank_val", "spell"})
TEST_STAGES = frozenset({"test", "typo", "rag"})


class SplitGuardError(RuntimeError):
    """Raised when an operation violates dev/test split isolation or config freeze."""


def load_config(path: Path | str = Path("config/config.yaml")) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def save_config(config: Mapping[str, Any], path: Path | str = Path("config/config.yaml")) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(yaml.safe_dump(dict(config), sort_keys=False, allow_unicode=True), encoding="utf-8")


def compute_config_hash(config: Mapping[str, Any]) -> str:
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_freeze_log(log_path: Path) -> list[dict[str, Any]]:
    if not log_path.exists():
        return []
    return json.loads(log_path.read_text(encoding="utf-8"))


def _write_freeze_log(log_path: Path, entries: list[dict[str, Any]]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps(entries, indent=2), encoding="utf-8")


def freeze_config(
    config: Mapping[str, Any],
    artifacts_dir: Path | str = Path("artifacts"),
    force_reason: str | None = None,
) -> dict[str, Any]:
    """Append a config snapshot entry to artifacts/freeze_log.json."""
    art_dir = Path(artifacts_dir)
    log_path = art_dir / "freeze_log.json"
    entries = _read_freeze_log(log_path)
    entry: dict[str, Any] = {
        "config_hash": compute_config_hash(config),
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "stage_results": {},
    }
    if force_reason is not None:
        entry["force_reason"] = force_reason
    entries.append(entry)
    _write_freeze_log(log_path, entries)
    return entry


def record_stage_result(
    stage: str,
    file_path: Path | str,
    config: Mapping[str, Any],
    artifacts_dir: Path | str = Path("artifacts"),
) -> None:
    """Record a stage output file hash and timestamp in freeze_log.json."""
    art_dir = Path(artifacts_dir)
    log_path = art_dir / "freeze_log.json"
    entries = _read_freeze_log(log_path)
    cfg_hash = compute_config_hash(config)

    if not entries or entries[-1].get("config_hash") != cfg_hash:
        entries.append(
            {
                "config_hash": cfg_hash,
                "frozen_at": datetime.now(timezone.utc).isoformat(),
                "stage_results": {},
            }
        )

    target = Path(file_path)
    entries[-1].setdefault("stage_results", {})[stage] = {
        "file": str(target),
        "sha256": _sha256_file(target) if target.exists() else None,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    _write_freeze_log(log_path, entries)


def _check_validation_dev(art_dir: Path) -> None:
    val_csv = art_dir / "dev" / "validation_dev.csv"
    if not val_csv.exists():
        raise SplitGuardError(
            "Estágio 'test' exige artifacts/dev/validation_dev.csv com 'semantic' e 'reranking'."
        )
    with val_csv.open("r", encoding="utf-8", newline="") as f:
        systems = {row.get("system", "") for row in csv.DictReader(f)}
    missing = [s for s in ("semantic", "reranking") if s not in systems]
    if missing:
        raise SplitGuardError(
            f"Estágio 'test' exige validações no dev antes de rodar: faltando {missing}."
        )


def guard_stage(
    stage: str,
    split: str,
    config: Mapping[str, Any],
    artifacts_dir: Path | str = Path("artifacts"),
    force: bool = False,
    reason: str | None = None,
) -> None:
    """Enforce split isolation and test-run lock rules (TDS-14)."""
    if stage in TUNING_STAGES and split == "test":
        raise SplitGuardError(
            f"Estágio de ajuste '{stage}' não pode ser executado com split='test'."
        )

    if stage == "test":
        if not config.get("frozen", False):
            raise SplitGuardError("Estágio 'test' exige frozen: true no config.yaml.")

        art_dir = Path(artifacts_dir)
        _check_validation_dev(art_dir)

        cfg_hash = compute_config_hash(config)
        lock_path = art_dir / "test_run.lock"
        if lock_path.exists():
            lock_data = json.loads(lock_path.read_text(encoding="utf-8"))
            if lock_data.get("config_hash") != cfg_hash:
                if not force:
                    raise SplitGuardError(
                        "test_run.lock já existe com outro config_hash; use --force --reason para sobrescrever."
                    )
                if not reason or not reason.strip():
                    raise SplitGuardError("Uso de --force exige --reason não vazio.")
                freeze_config(config, artifacts_dir=art_dir, force_reason=reason.strip())

        lock_path.parent.mkdir(parents=True, exist_ok=True)
        lock_path.write_text(
            json.dumps(
                {
                    "config_hash": cfg_hash,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
