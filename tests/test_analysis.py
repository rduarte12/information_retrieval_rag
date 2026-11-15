from __future__ import annotations

import csv
import json
from pathlib import Path

import pandas as pd
import pytest

from src.analysis import (
    select_rf7_cases,
    select_rf10_cases,
    select_rf14_queries,
    build_rf14_trace,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
ANALYSIS_DIR = REPO_ROOT / "artifacts" / "analysis"


def test_select_rf7_cases_deterministic_tiebreak():
    df = pd.DataFrame(
        [
            {"query_id": "Q1", "model": "lexical", "NDCG10": 0.8},
            {"query_id": "Q1", "model": "semantic", "NDCG10": 0.2},
            {"query_id": "Q2", "model": "lexical", "NDCG10": 0.1},
            {"query_id": "Q2", "model": "semantic", "NDCG10": 0.9},
            {"query_id": "Q3", "model": "lexical", "NDCG10": 0.5},
            {"query_id": "Q3", "model": "semantic", "NDCG10": 0.5},
            {"query_id": "Q4", "model": "lexical", "NDCG10": 0.2},
            {"query_id": "Q4", "model": "semantic", "NDCG10": 0.6},
            {"query_id": "Q5", "model": "lexical", "NDCG10": 0.7},
            {"query_id": "Q5", "model": "semantic", "NDCG10": 0.3},
            {"query_id": "Q6", "model": "lexical", "NDCG10": 0.3},
            {"query_id": "Q6", "model": "semantic", "NDCG10": 0.5},
        ]
    )
    cases = select_rf7_cases(df)
    assert len(cases) == 6
    sem_wins = [c["query_id"] for c in cases if c["group"] == "sem_win"]
    lex_wins = [c["query_id"] for c in cases if c["group"] == "lex_win"]
    assert sem_wins == ["Q2", "Q4", "Q6"]
    assert lex_wins == ["Q1", "Q5", "Q3"]


def test_select_rf10_cases_and_nochange():
    df = pd.DataFrame(
        [
            {"query_id": "Q1", "model": "hybrid", "NDCG10": 0.2},
            {"query_id": "Q1", "model": "reranking", "NDCG10": 0.7},  # +0.5
            {"query_id": "Q2", "model": "hybrid", "NDCG10": 0.3},
            {"query_id": "Q2", "model": "reranking", "NDCG10": 0.6},  # +0.3
            {"query_id": "Q3", "model": "hybrid", "NDCG10": 0.8},
            {"query_id": "Q3", "model": "reranking", "NDCG10": 0.4},  # -0.4
            {"query_id": "Q4", "model": "hybrid", "NDCG10": 0.5},
            {"query_id": "Q4", "model": "reranking", "NDCG10": 0.5},  # 0.0
        ]
    )
    hybrid_top10_rel = {"Q1": 1, "Q2": 1, "Q3": 1, "Q4": 2}
    cases = select_rf10_cases(df, hybrid_top10_rel)
    by_case = {c["case"]: c["query_id"] for c in cases}
    assert by_case == {"gain1": "Q1", "gain2": "Q2", "nochange": "Q4", "loss": "Q3"}


def test_select_rf14_queries_and_trace():
    unblinded = [
        {"query_id": "Q1", "condition": "A", "fidelidade": "1", "completude": "1", "utilidade": "1"},
        {"query_id": "Q1", "condition": "B", "fidelidade": "2", "completude": "2", "utilidade": "2"},  # +1.0
        {"query_id": "Q2", "condition": "A", "fidelidade": "1", "completude": "1", "utilidade": "1"},
        {"query_id": "Q2", "condition": "B", "fidelidade": "1", "completude": "1", "utilidade": "1"},  # 0.0
        {"query_id": "Q3", "condition": "A", "fidelidade": "2", "completude": "2", "utilidade": "2"},
        {"query_id": "Q3", "condition": "B", "fidelidade": "1", "completude": "1", "utilidade": "1"},  # -1.0
    ]
    selected = select_rf14_queries(unblinded)
    assert selected == {"melhorou": "Q1", "pouca_diferenca": "Q2", "nao_ajudou": "Q3"}

    qrels = {"Q1": {"D1": 2, "D2": 1}, "Q2": {"D3": 1}, "Q3": {"D4": 2}}
    runs = {
        "lexical": {"Q1": [("D1", 5.0)], "Q2": [], "Q3": [("D4", 3.0)]},
        "semantic": {"Q1": [("D2", 0.9)], "Q2": [("D3", 0.8)], "Q3": []},
        "hybrid": {"Q1": [("D1", 0.03), ("D2", 0.02)], "Q2": [("D3", 0.02)], "Q3": [("D4", 0.01)]},
        "reranking": {"Q1": [("D1", 4.0)], "Q2": [("D3", 2.0)], "Q3": [("D4", 1.0)]},
    }
    contexts = {"Q1": [{"doc_id": "D1"}], "Q2": [{"doc_id": "D3"}], "Q3": []}
    rows = build_rf14_trace(["Q1", "Q2", "Q3"], qrels, runs, contexts)
    assert len(rows) == 4
    d1_row = [r for r in rows if r["doc_id"] == "D1"][0]
    assert d1_row["in_context"] is True
    assert d1_row["rank_rerank"] == 1
    d2_row = [r for r in rows if r["doc_id"] == "D2"][0]
    assert d2_row["in_context"] is False


@pytest.mark.data
def test_rf7_and_rf10_csvs_generated():
    rf7_cases = ANALYSIS_DIR / "rf7_cases.csv"
    rf7_top5 = ANALYSIS_DIR / "rf7_top5.csv"
    rf10_cases = ANALYSIS_DIR / "rf10_cases.csv"
    rf10_moves = ANALYSIS_DIR / "rf10_moves.csv"

    assert rf7_cases.exists()
    assert rf7_top5.exists()
    assert rf10_cases.exists()
    assert rf10_moves.exists()

    df7 = pd.read_csv(rf7_cases)
    assert len(df7) == 6
    assert set(df7["group"]) == {"sem_win", "lex_win"}

    df7_top5 = pd.read_csv(rf7_top5)
    assert 1 <= len(df7_top5) <= 60
    assert (df7_top5["rank"] <= 5).all()

    df10 = pd.read_csv(rf10_cases)
    assert len(df10) == 4
    assert set(df10["case"]) == {"gain1", "gain2", "nochange", "loss"}
