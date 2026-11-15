from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd
import pytest

from src.data_loader import load_qrels
from src.results_writer import (
    validate_results,
    write_metrics_per_query_csv,
    write_ranking_csv,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = REPO_ROOT / "data" / "nfcorpus"
RESULTS_DIR = REPO_ROOT / "results"
ARTIFACTS_DIR = REPO_ROOT / "artifacts"


def test_write_ranking_csv_and_validate_synthetic(tmp_path: Path):
    qrels = {"Q1": {"D1": 2, "D3": 1}, "Q2": {"D2": 1}}
    qids = ["Q1", "Q2"]

    lex = {"Q1": [("D1", 5.0), ("D2", 3.0)], "Q2": [("D2", 4.0)]}
    sem = {"Q1": [("D1", 0.9), ("D2", 0.5)], "Q2": [("D2", 0.8), ("D3", 0.2)]}
    hyb = {"Q1": [("D1", 0.03), ("D2", 0.02)], "Q2": [("D2", 0.03), ("D3", 0.01)]}
    rer = {"Q1": [("D1", 4.0), ("D2", 1.0)], "Q2": [("D2", 3.0), ("D3", -1.0)]}

    write_ranking_csv(lex, qrels, tmp_path / "results_lexical.csv", model="lexical")
    write_ranking_csv(sem, qrels, tmp_path / "results_semantic.csv", model="semantic")
    write_ranking_csv(hyb, qrels, tmp_path / "results_hybrid.csv", model="hybrid")
    write_ranking_csv(rer, qrels, tmp_path / "results_reranking.csv", model="reranking")

    write_metrics_per_query_csv(
        {"lexical": lex, "semantic": sem, "hybrid": hyb, "reranking": rer},
        qrels,
        qids,
        tmp_path / "metrics_per_query.csv",
    )

    df_lex = pd.read_csv(tmp_path / "results_lexical.csv")
    assert list(df_lex.columns) == ["query_id", "doc_id", "rank", "score", "relevance"]
    assert list(df_lex["relevance"]) == [2, 0, 1]

    df_rer = pd.read_csv(tmp_path / "results_reranking.csv")
    assert list(df_rer.columns) == [
        "query_id",
        "doc_id",
        "rank",
        "score",
        "relevance",
        "score_source",
    ]
    assert set(df_rer["score_source"]) == {"cross_encoder"}

    df_m = pd.read_csv(tmp_path / "metrics_per_query.csv")
    assert len(df_m) == 8


def test_validate_results_catches_bad_rank_gap(tmp_path: Path):
    bad = tmp_path / "results_semantic.csv"
    with bad.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["query_id", "doc_id", "rank", "score", "relevance"])
        w.writerow(["Q1", "D1", 1, 0.9, 1])
        w.writerow(["Q1", "D2", 3, 0.8, 0])

    with pytest.raises(ValueError, match="contiguous"):
        validate_results(
            results_dir=tmp_path,
            qrels_test={"Q1": {"D1": 1}},
            query_ids_test=["Q1"],
            check_models=("semantic",),
            require_semantic_exact_100=False,
        )


@pytest.mark.data
def test_s8_five_results_csvs_and_invariants():
    assert (ARTIFACTS_DIR / "test_run.lock").exists()

    qrels_test = load_qrels(DATA_ROOT / "qrels" / "test.tsv")
    query_ids_test = sorted(qrels_test.keys())
    assert len(query_ids_test) == 323

    summary = validate_results(
        results_dir=RESULTS_DIR,
        qrels_test=qrels_test,
        query_ids_test=query_ids_test,
        artifacts_dir=ARTIFACTS_DIR,
    )
    assert summary["semantic_rows"] == 32300
    assert summary["metrics_rows"] == 1292
