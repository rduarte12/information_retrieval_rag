from __future__ import annotations

import csv
import random
from pathlib import Path

import pytest
import yaml

from src.fusion import read_trec_run, rrf, rrf_single_query

REPO_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = REPO_ROOT / "artifacts"
CONFIG_PATH = REPO_ROOT / "config" / "config.yaml"


def test_rrf_hand_calculated():
    # k = 60, rank starts at 1
    # D1: lex rank 1, sem rank 2 -> 1/61 + 1/62
    # D2: lex rank 2, sem rank 1 -> 1/62 + 1/61 (ties with D1 -> D1 < D2)
    # D3: lex rank 3, sem absent -> 1/63
    # D4: lex absent, sem rank 3 -> 1/63 (ties with D3 -> D3 < D4)
    run_lex = {
        "Q1": [("D1", 10.0), ("D2", 9.0), ("D3", 8.0)],
    }
    run_sem = {
        "Q1": [("D2", 0.9), ("D1", 0.8), ("D4", 0.7)],
    }

    fused = rrf([run_lex, run_sem], k=60, depth=100, top_k=100)
    assert list(fused.keys()) == ["Q1"]

    ranking = fused["Q1"]
    assert [doc_id for doc_id, _ in ranking] == ["D1", "D2", "D3", "D4"]
    assert ranking[0][1] == pytest.approx(1.0 / 61 + 1.0 / 62, abs=1e-12)
    assert ranking[1][1] == pytest.approx(1.0 / 61 + 1.0 / 62, abs=1e-12)
    assert ranking[2][1] == pytest.approx(1.0 / 63, abs=1e-12)
    assert ranking[3][1] == pytest.approx(1.0 / 63, abs=1e-12)


def test_rrf_depth_and_top_k():
    r1 = [("A", 3.0), ("B", 2.0), ("C", 1.0)]
    r2 = [("C", 3.0), ("B", 2.0), ("A", 1.0)]

    # With depth=2, C is dropped from r1 and A is dropped from r2
    res = rrf_single_query([r1, r2], k=10, depth=2, top_k=2)
    # B: 1/12 + 1/12 = 2/12; A: 1/11; C: 1/11 -> B first, then A (tie with C)
    assert [d for d, _ in res] == ["B", "A"]


def test_read_trec_run(tmp_path: Path):
    p = tmp_path / "sample.trec"
    p.write_text(
        "Q2 Q0 D1 1 0.500000 tag\n"
        "Q1 Q0 D2 1 1.250000 tag\n"
        "Q1 Q0 D1 2 0.750000 tag\n",
        encoding="utf-8",
    )
    run = read_trec_run(p)
    assert run["Q1"] == [("D2", 1.25), ("D1", 0.75)]
    assert run["Q2"] == [("D1", 0.5)]


@pytest.mark.data
def test_rrf_k_dev_csv_and_config_argmax():
    csv_path = ARTIFACTS_DIR / "dev" / "rrf_k_dev.csv"
    assert csv_path.exists()

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) >= 3
    ks = {int(r["k"]) for r in rows}
    assert {10, 60, 100}.issubset(ks)
    for r in rows:
        assert int(r["n_queries"]) == 324

    best_row = min(rows, key=lambda r: (-float(r["ndcg10_dev"]), int(r["k"])))
    cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    assert int(cfg["rrf"]["k"]) == int(best_row["k"])


@pytest.mark.data
def test_hybrid_dev_trec_recomputed_scores():
    hybrid_path = ARTIFACTS_DIR / "runs" / "hybrid_dev.trec"
    lex_path = ARTIFACTS_DIR / "runs" / "lexical_dev.trec"
    sem_path = ARTIFACTS_DIR / "runs" / "semantic_dev.trec"

    assert hybrid_path.exists()
    cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    k = int(cfg["rrf"]["k"])
    depth = int(cfg["rrf"]["depth"])

    hybrid_run = read_trec_run(hybrid_path)
    lex_run = read_trec_run(lex_path)
    sem_run = read_trec_run(sem_path)

    assert len(hybrid_run) == 324
    for qid, ranking in hybrid_run.items():
        assert len(ranking) <= 100

    sampled_qids = random.Random(42).sample(sorted(hybrid_run.keys()), 5)
    for qid in sampled_qids:
        recomputed = rrf_single_query(
            [lex_run.get(qid, []), sem_run.get(qid, [])],
            k=k,
            depth=depth,
            top_k=100,
        )
        assert hybrid_run[qid][0][0] == recomputed[0][0]
        assert abs(hybrid_run[qid][0][1] - recomputed[0][1]) <= 1e-12
