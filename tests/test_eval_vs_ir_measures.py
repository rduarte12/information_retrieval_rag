# tests/test_eval_vs_ir_measures.py
"""
Validação cruzada da implementação própria de métricas (src/evaluation.py)
contra a biblioteca de referência ir_measures (diferença <= 1e-9).
"""

from pathlib import Path

import ir_measures
import pytest
from ir_measures import AP, P, R, nDCG

from data_loader import load_dataset
from evaluation import evaluate_run

ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = ROOT / "data" / "nfcorpus"
LEXICAL_DEV_TREC = ROOT / "artifacts" / "runs" / "lexical_dev.trec"


def _assert_matches_ir_measures(run, qrels, query_ids):
    per_query, agg = evaluate_run(run, qrels, query_ids, model="check")

    # Converte run para {qid: {doc_id: score_ordenado}} preservando a ordem exata do ranking
    # (evita que o desempate interno por doc_id decrescente do pytrec_eval altere empates).
    run_dict = {
        qid: {doc_id: float(len(run.get(qid, [])) - i) for i, (doc_id, _) in enumerate(run.get(qid, []))}
        for qid in query_ids
    }
    measures = [P(rel=1) @ 10, R(rel=1) @ 10, AP(rel=1) @ 10, nDCG @ 10]
    ir_per_q = {}
    for m in ir_measures.iter_calc(measures, qrels, run_dict):
        ir_per_q.setdefault(m.query_id, {})[str(m.measure)] = float(m.value)

    for _, row in per_query.iterrows():
        qid = row["query_id"]
        vals = ir_per_q.get(qid, {})
        assert abs(row["P10"] - vals.get("P@10", 0.0)) <= 1e-9
        assert abs(row["R10"] - vals.get("R@10", 0.0)) <= 1e-9
        assert abs(row["AP10"] - vals.get("AP@10", 0.0)) <= 1e-9
        assert abs(row["NDCG10"] - vals.get("nDCG@10", 0.0)) <= 1e-9

    assert abs(agg["P@10"] - per_query["P10"].mean()) <= 1e-9
    assert abs(agg["R@10"] - per_query["R10"].mean()) <= 1e-9
    assert abs(agg["MAP@10"] - per_query["AP10"].mean()) <= 1e-9
    assert abs(agg["NDCG@10"] - per_query["NDCG10"].mean()) <= 1e-9


def test_cross_validation_synthetic_with_empty_and_missing():
    qrels = {
        "q1": {"d1": 2, "d2": 1, "d3": 0, "d4": 2, "d5": 1},
        "q2": {"d10": 1, "d11": 2},
        "q3": {"d20": 0},  # R == 0
        "q4": {"d30": 1},  # ausente na run
    }
    run = {
        "q1": [("d2", 5.0), ("x1", 4.0), ("d1", 3.0), ("d4", 2.0), ("d3", 1.0)],
        "q2": [],  # run vazia
        "q3": [("d20", 1.0)],
    }
    _assert_matches_ir_measures(run, qrels, ["q1", "q2", "q3", "q4"])


@pytest.mark.data
def test_cross_validation_nfcorpus_dev_run():
    if not LEXICAL_DEV_TREC.exists():
        pytest.skip("artifacts/runs/lexical_dev.trec ainda não foi gerado.")
    dataset = load_dataset(DATA_ROOT)
    dev_qids = dataset.query_ids["dev"]
    dev_qrels = dataset.qrels["dev"]

    run: dict[str, list[tuple[str, float]]] = {qid: [] for qid in dev_qids}
    for sd in ir_measures.read_trec_run(str(LEXICAL_DEV_TREC)):
        if sd.query_id in run:
            run[sd.query_id].append((sd.doc_id, float(sd.score)))
    for qid in dev_qids:
        run[qid].sort(key=lambda x: (-x[1], x[0]))
    _assert_matches_ir_measures(run, dev_qrels, dev_qids)
