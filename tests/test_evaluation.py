# tests/test_evaluation.py
"""
Testes-contrato para C-6 (Avaliador, TDS-6 / Sprint S3).

Valida P@10, R@10, AP@10, NDCG@10, MAP@10, R@5, P@5 e Hit@5 contra
valores calculados à mão, cobrindo casos de borda (run vazia, R=0, <10 docs).
"""

import math

import pytest

from evaluation import evaluate_rag_retrieval, evaluate_run, score_query


def test_score_query_hand_calculated():
    # qrels: d1=2, d2=1, d3=0, d4=2 -> R = 3 (grau > 0)
    # ranking: [d2 (rel=1), d5 (unjudged=0), d1 (rel=2)] (apenas 3 recuperados)
    qrels_q = {"d1": 2, "d2": 1, "d3": 0, "d4": 2}
    ranking = [("d2", 0.9), ("d5", 0.8), ("d1", 0.7)]

    m = score_query(ranking, qrels_q)

    # P10 = 2 / 10 (denominador fixo 10 mesmo com < 10 recuperados)
    assert m["P10"] == pytest.approx(0.2)
    # R10 = 2 / 3
    assert m["R10"] == pytest.approx(2 / 3)
    # AP10: pos 1 -> P@1 = 1/1; pos 3 -> P@3 = 2/3; soma / R(3)
    expected_ap10 = (1.0 + 2.0 / 3.0) / 3.0
    assert m["AP10"] == pytest.approx(expected_ap10)
    # NDCG10: ganho linear, ideal = [2, 2, 1] de todos os julgamentos
    dcg = 1.0 / math.log2(1 + 1) + 0.0 / math.log2(2 + 1) + 2.0 / math.log2(3 + 1)
    idcg = 2.0 / math.log2(1 + 1) + 2.0 / math.log2(2 + 1) + 1.0 / math.log2(3 + 1)
    assert m["NDCG10"] == pytest.approx(dcg / idcg)


def test_score_query_empty_run():
    qrels_q = {"d1": 2, "d2": 1}
    m = score_query([], qrels_q)
    assert m == {"P10": 0.0, "R10": 0.0, "AP10": 0.0, "NDCG10": 0.0}


def test_score_query_zero_relevant():
    """TDS-DEC-5: consulta com R == 0 -> P10=0, R10=0, AP10=0, NDCG10=0."""
    qrels_q = {"d1": 0, "d2": 0}
    ranking = [("d1", 1.0), ("d3", 0.5)]
    m = score_query(ranking, qrels_q)
    assert m == {"P10": 0.0, "R10": 0.0, "AP10": 0.0, "NDCG10": 0.0}


def test_evaluate_run_aggregation_and_missing_queries():
    qrels = {
        "q1": {"d1": 2, "d2": 1},
        "q2": {"d3": 1, "d4": 1},
    }
    # q2 ausente na run -> deve contar como 0 e aparecer no per_query
    run = {"q1": [("d1", 1.0), ("d2", 0.5)]}

    per_query, agg = evaluate_run(run, qrels, ["q1", "q2"], model="lexical")

    assert list(per_query.columns) == ["query_id", "model", "P10", "R10", "AP10", "NDCG10"]
    assert len(per_query) == 2
    assert set(per_query["model"]) == {"lexical"}

    q2_row = per_query[per_query["query_id"] == "q2"].iloc[0]
    assert q2_row["P10"] == 0.0
    assert q2_row["R10"] == 0.0
    assert q2_row["AP10"] == 0.0
    assert q2_row["NDCG10"] == 0.0

    # MAP@10 == mean(AP10)
    assert agg["MAP@10"] == pytest.approx(per_query["AP10"].mean())
    assert agg["P@10"] == pytest.approx(per_query["P10"].mean())
    assert agg["R@10"] == pytest.approx(per_query["R10"].mean())
    assert agg["NDCG@10"] == pytest.approx(per_query["NDCG10"].mean())


def test_evaluate_run_rejects_duplicate_docs():
    qrels = {"q1": {"d1": 1}}
    run = {"q1": [("d1", 1.0), ("d1", 0.5)]}
    with pytest.raises(ValueError, match="duplicado"):
        evaluate_run(run, qrels, ["q1"], model="lexical")


def test_evaluate_rag_retrieval_hand_calculated():
    qrels = {
        "q1": {"d1": 2, "d2": 1, "d3": 1, "d4": 0},  # R = 3
        "q2": {"d5": 1, "d6": 1},  # R = 2
    }
    run = {
        # top-5 tem d1 e d3 (2 relevantes de 3) -> P5=2/5, R5=2/3, Hit5=1
        "q1": [("d4", 0.9), ("d1", 0.8), ("x1", 0.7), ("x2", 0.6), ("d3", 0.5), ("d2", 0.4)],
        # top-5 sem nenhum relevante -> P5=0, R5=0, Hit5=0
        "q2": [("x1", 0.9), ("x2", 0.8)],
    }

    df = evaluate_rag_retrieval(run, qrels, ["q1", "q2"])
    assert list(df.columns) == ["query_id", "R5", "P5", "Hit5"]
    assert len(df) == 2

    r1 = df[df["query_id"] == "q1"].iloc[0]
    assert r1["P5"] == pytest.approx(0.4)
    assert r1["R5"] == pytest.approx(2 / 3)
    assert r1["Hit5"] == 1

    r2 = df[df["query_id"] == "q2"].iloc[0]
    assert r2["P5"] == pytest.approx(0.0)
    assert r2["R5"] == pytest.approx(0.0)
    assert r2["Hit5"] == 0
