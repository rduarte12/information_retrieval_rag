# src/evaluation.py
"""
Avaliador de Recuperação de Informação (C-6 / TDS-6).

Implementa P@10, R@10, AP@10 (MAP@10), NDCG@10 (ganho linear, IDCG global
sobre todos os julgamentos da consulta), e R@5, P@5, Hit@5 para o RAG.
"""

from __future__ import annotations

import math
from typing import Mapping, Sequence

import pandas as pd


def _validate_unique_docs(ranking: Sequence[tuple[str, float]]) -> None:
    seen: set[str] = set()
    for doc_id, _ in ranking:
        if doc_id in seen:
            raise ValueError(f"Documento duplicado na run: {doc_id!r}")
        seen.add(doc_id)


def score_query(
    ranking: Sequence[tuple[str, float]],
    qrels_q: Mapping[str, int],
) -> dict[str, float]:
    """Calcula P10, R10, AP10 e NDCG10 para uma única consulta."""
    _validate_unique_docs(ranking)

    rel = {d for d, g in qrels_q.items() if g > 0}
    R = len(rel)
    if R == 0 or not ranking:
        return {"P10": 0.0, "R10": 0.0, "AP10": 0.0, "NDCG10": 0.0}

    top10 = [doc_id for doc_id, _ in ranking[:10]]

    rel_so_far = 0
    ap_sum = 0.0
    dcg = 0.0
    for rank_idx, doc_id in enumerate(top10, start=1):
        grade = qrels_q.get(doc_id, 0)
        if grade > 0:
            rel_so_far += 1
            ap_sum += rel_so_far / rank_idx
            dcg += grade / math.log2(rank_idx + 1)

    p10 = rel_so_far / 10.0
    r10 = rel_so_far / R
    ap10 = ap_sum / R

    ideal_grades = sorted((g for g in qrels_q.values() if g > 0), reverse=True)[:10]
    idcg = sum(g / math.log2(i + 1) for i, g in enumerate(ideal_grades, start=1))
    ndcg10 = (dcg / idcg) if idcg > 0.0 else 0.0

    return {
        "P10": float(p10),
        "R10": float(r10),
        "AP10": float(ap10),
        "NDCG10": float(ndcg10),
    }


def evaluate_run(
    run: Mapping[str, Sequence[tuple[str, float]]],
    qrels: Mapping[str, Mapping[str, int]],
    query_ids: Sequence[str],
    model: str,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Avalia uma Run completa sobre todas as consultas de query_ids."""
    rows: list[dict] = []
    for qid in query_ids:
        ranking = run.get(qid, [])
        qrels_q = qrels.get(qid, {})
        m = score_query(ranking, qrels_q)
        rows.append(
            {
                "query_id": qid,
                "model": model,
                "P10": m["P10"],
                "R10": m["R10"],
                "AP10": m["AP10"],
                "NDCG10": m["NDCG10"],
            }
        )

    per_query = pd.DataFrame(
        rows, columns=["query_id", "model", "P10", "R10", "AP10", "NDCG10"]
    )
    aggregate = {
        "P@10": float(per_query["P10"].mean()) if len(per_query) else 0.0,
        "R@10": float(per_query["R10"].mean()) if len(per_query) else 0.0,
        "MAP@10": float(per_query["AP10"].mean()) if len(per_query) else 0.0,
        "NDCG@10": float(per_query["NDCG10"].mean()) if len(per_query) else 0.0,
    }
    return per_query, aggregate


def evaluate_rag_retrieval(
    run: Mapping[str, Sequence[tuple[str, float]]],
    qrels: Mapping[str, Mapping[str, int]],
    query_ids: Sequence[str],
) -> pd.DataFrame:
    """Calcula R5, P5 e Hit5 sobre o Top-5 enviado ao contexto do RAG (Req. 13)."""
    rows: list[dict] = []
    for qid in query_ids:
        ranking = run.get(qid, [])
        _validate_unique_docs(ranking)
        qrels_q = qrels.get(qid, {})
        rel = {d for d, g in qrels_q.items() if g > 0}
        R = len(rel)

        top5 = [doc_id for doc_id, _ in ranking[:5]]
        hits = sum(1 for d in top5 if d in rel)

        rows.append(
            {
                "query_id": qid,
                "R5": float(hits / R) if R > 0 else 0.0,
                "P5": float(hits / 5.0),
                "Hit5": 1 if hits >= 1 else 0,
            }
        )

    return pd.DataFrame(rows, columns=["query_id", "R5", "P5", "Hit5"])
