from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import pandas as pd
import pytest
import yaml

from src.fusion import read_trec_run
from src.rerank import (
    DEFAULT_RERANK_DEPTH,
    DEFAULT_RERANK_MAX_LENGTH,
    DEFAULT_RERANK_MODEL_ID,
    build_rerank_rows,
    rerank_single_query,
    score_source_for_rank,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = REPO_ROOT / "artifacts"
CONFIG_PATH = REPO_ROOT / "config" / "config.yaml"


def test_rerank_single_query_permutates_top50_and_preserves_tail():
    hybrid = [(f"MED-{i:03d}", 1.0 / (10 + i + 1)) for i in range(70)]
    # Reverse scores for top-50, with an explicit tie between MED-001 and MED-002
    ce_scores = {f"MED-{i:03d}": float(i) for i in range(50)}
    ce_scores["MED-001"] = 100.0
    ce_scores["MED-002"] = 100.0

    reranked = rerank_single_query(hybrid, ce_scores=ce_scores, depth=50)

    assert len(reranked) == 70
    assert {d for d, _ in reranked[:50]} == {d for d, _ in hybrid[:50]}
    assert reranked[0][0] == "MED-001"
    assert reranked[1][0] == "MED-002"
    assert reranked[50:] == hybrid[50:]


def test_rerank_handles_fewer_than_depth_docs():
    hybrid = [("D1", 0.5), ("D2", 0.4), ("D3", 0.3)]
    ce_scores = {"D1": -1.0, "D2": 2.5, "D3": 0.0}

    reranked = rerank_single_query(hybrid, ce_scores=ce_scores, depth=50)
    assert [d for d, _ in reranked] == ["D2", "D3", "D1"]


def test_score_source_coherence():
    assert score_source_for_rank(1, depth=50) == "cross_encoder"
    assert score_source_for_rank(50, depth=50) == "cross_encoder"
    assert score_source_for_rank(51, depth=50) == "hybrid_rrf"
    assert score_source_for_rank(100, depth=50) == "hybrid_rrf"

    ranking = [(f"D{i}", float(100 - i)) for i in range(1, 61)]
    rows = build_rerank_rows("Q1", ranking, depth=50)
    assert len(rows) == 60
    for row in rows:
        expected_source = "cross_encoder" if row["rank"] <= 50 else "hybrid_rrf"
        assert row["score_source"] == expected_source


@pytest.mark.data
def test_rerank_dev_artifacts_and_invariants():
    rerank_path = ARTIFACTS_DIR / "runs" / "reranking_dev.trec"
    hybrid_path = ARTIFACTS_DIR / "runs" / "hybrid_dev.trec"
    parquet_path = ARTIFACTS_DIR / "rerank_scores_dev.parquet"
    manifest_path = ARTIFACTS_DIR / "rerank_manifest.json"
    val_csv = ARTIFACTS_DIR / "dev" / "validation_dev.csv"

    assert rerank_path.exists()
    assert parquet_path.exists()
    assert manifest_path.exists()
    assert val_csv.exists()

    rerank_run = read_trec_run(rerank_path)
    hybrid_run = read_trec_run(hybrid_path)
    assert len(rerank_run) == 324

    for qid in sorted(hybrid_run.keys()):
        h_list = hybrid_run[qid]
        r_list = rerank_run[qid]
        assert len(r_list) == len(h_list)
        assert {d for d, _ in r_list[:50]} == {d for d, _ in h_list[:50]}
        assert r_list[50:100] == h_list[50:100]

    df = pd.read_parquet(parquet_path)
    assert set(df.columns) == {
        "query_id",
        "doc_id",
        "hybrid_rank",
        "ce_score",
        "n_tokens",
        "truncated",
        "model_rev",
    }
    assert len(df) == sum(min(50, len(hybrid_run[q])) for q in hybrid_run)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["model_id"] == DEFAULT_RERANK_MODEL_ID
    assert re.fullmatch(r"[0-9a-f]{40}", manifest["revision"])
    assert manifest["max_length"] == DEFAULT_RERANK_MAX_LENGTH
    assert manifest["depth"] == DEFAULT_RERANK_DEPTH
    assert manifest["activation"] == "identity"
    assert manifest["n_pairs"] == len(df)
    assert manifest["n_truncated"] >= 0
    assert manifest["pct_truncated"] >= 0.0
    assert manifest["rerank_time_ms"] > 0.0

    with val_csv.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    rerank_rows = [r for r in rows if r["system"] == "reranking"]
    assert len(rerank_rows) >= 1
    assert 0.0 <= float(rerank_rows[-1]["ndcg10_dev"]) <= 1.0

    cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    assert isinstance(cfg["frozen"], bool)
