# tests/test_data_loader.py
"""
Testes-contrato para C-0 (Carregador do NFCorpus, TDS-0 / Sprint S1).

Testes sem marcador rodam sobre dados sintéticos rápidos.
Testes marcados com `@pytest.mark.data` validam os arquivos reais em data/nfcorpus/.
"""

import json
from pathlib import Path

import pytest

from data_loader import (
    compute_data_stats,
    doc_text,
    load_corpus,
    load_dataset,
    load_qrels,
    load_queries,
    select_rag_queries,
)

ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = ROOT / "data" / "nfcorpus"

EXPECTED_RAG_QUERIES = [
    "PLAIN-68",
    "PLAIN-2450",
    "PLAIN-153",
    "PLAIN-2850",
    "PLAIN-2750",
    "PLAIN-270",
    "PLAIN-2470",
    "PLAIN-2430",
    "PLAIN-238",
    "PLAIN-3342",
]


# ---------------------------------------------------------------------------
# Testes unitários (dados sintéticos)
# ---------------------------------------------------------------------------


def test_doc_text_concatenation():
    """Regra TDS-DEC-2: (title.strip() + ' ' + text.strip()).strip()."""
    assert doc_text({"title": "  Cancer Diet ", "text": " Omega-3 study.  "}) == (
        "Cancer Diet Omega-3 study."
    )
    assert doc_text({"title": "", "text": "Only body text."}) == "Only body text."
    assert doc_text({"title": "   ", "text": "Only body text."}) == "Only body text."
    assert doc_text({"title": "Only Title", "text": ""}) == "Only Title"
    assert doc_text({"title": "", "text": ""}) == ""


def test_load_corpus_synthetic(tmp_path):
    corpus_file = tmp_path / "corpus.jsonl"
    rows = [
        {"_id": "MED-10", "title": "Title A", "text": "Body A", "metadata": {}},
        {"_id": "MED-2", "title": "", "text": "Body B", "metadata": {}},
    ]
    corpus_file.write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
    )

    docs = load_corpus(corpus_file)
    assert len(docs) == 2
    assert set(docs.keys()) == {"MED-10", "MED-2"}
    assert docs["MED-10"] == {"title": "Title A", "text": "Body A"}
    assert docs["MED-2"] == {"title": "", "text": "Body B"}


def test_load_queries_synthetic(tmp_path):
    queries_file = tmp_path / "queries.jsonl"
    rows = [
        {"_id": "PLAIN-2", "text": "Why coffee?", "metadata": {}},
        {"_id": "PLAIN-10", "text": "Vitamin D.", "metadata": {}},
    ]
    queries_file.write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
    )

    queries = load_queries(queries_file)
    assert queries == {"PLAIN-2": "Why coffee?", "PLAIN-10": "Vitamin D."}
    assert all(isinstance(k, str) for k in queries)


def test_load_qrels_synthetic_skips_header_and_preserves_str_ids(tmp_path):
    qrels_file = tmp_path / "test.tsv"
    qrels_file.write_text(
        "query-id\tcorpus-id\tscore\n"
        "PLAIN-10\tMED-01\t2\n"
        "PLAIN-10\tMED-02\t1\n"
        "PLAIN-2\tMED-01\t0\n",
        encoding="utf-8",
    )

    qrels = load_qrels(qrels_file)
    assert "query-id" not in qrels
    assert list(sorted(qrels.keys())) == ["PLAIN-10", "PLAIN-2"]  # ordem lexicográfica
    assert qrels["PLAIN-10"] == {"MED-01": 2, "MED-02": 1}
    assert qrels["PLAIN-2"] == {"MED-01": 0}
    assert isinstance(qrels["PLAIN-10"]["MED-01"], int)


def test_select_rag_queries_synthetic_deterministic():
    queries = {
        "Q-1": "First question?",
        "Q-2": "Not a question.",
        "Q-3": "Second question?  ",
        "Q-10": "Third question?",
    }
    test_qrels = {"Q-1": {"D1": 1}, "Q-2": {"D1": 1}, "Q-3": {"D1": 1}, "Q-10": {"D1": 1}}

    eligible, sampled = select_rag_queries(test_qrels, queries, n=2, seed=42)
    assert eligible == ["Q-1", "Q-10", "Q-3"]  # ordenado lexicograficamente
    assert len(sampled) == 2
    # Repetibilidade com mesma seed
    _, sampled_again = select_rag_queries(test_qrels, queries, n=2, seed=42)
    assert sampled == sampled_again


def test_compute_data_stats_synthetic():
    docs = {"D1": {"title": "T1", "text": "B1"}, "D2": {"title": "T2", "text": "B2"}}
    queries = {"Q1": "q1?", "Q2": "q2", "Q3": "q3"}
    qrels = {
        "dev": {"Q1": {"D1": 2, "D2": 1}},
        "test": {"Q2": {"D1": 2, "D2": 0}, "Q3": {"D1": 0}},
    }

    stats = compute_data_stats(docs, queries, qrels)
    assert stats["n_docs"] == 2
    assert stats["n_queries_total"] == 3
    assert stats["splits"]["dev"]["n_queries"] == 1
    assert stats["splits"]["dev"]["n_judgments"] == 2
    assert stats["splits"]["dev"]["n_queries_zero_rel"] == 0
    assert stats["splits"]["test"]["n_queries"] == 2
    assert stats["splits"]["test"]["n_queries_zero_rel"] == 1
    assert stats["splits"]["test"]["grade_distribution"]["2"] == 1
    assert stats["splits"]["test"]["grade_distribution"]["0"] == 2
    assert stats["splits"]["test"]["avg_rel_per_query"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Testes sobre os dados reais do NFCorpus (@pytest.mark.data)
# ---------------------------------------------------------------------------


@pytest.mark.data
def test_load_corpus_nfcorpus():
    docs = load_corpus(DATA_ROOT / "corpus.jsonl")
    assert len(docs) == 3633
    assert all(isinstance(k, str) for k in docs)
    assert all("title" in v and "text" in v for v in docs.values())


@pytest.mark.data
def test_load_queries_nfcorpus():
    queries = load_queries(DATA_ROOT / "queries.jsonl")
    assert len(queries) == 3237
    assert all(isinstance(k, str) for k in queries)


@pytest.mark.data
def test_load_qrels_and_splits_nfcorpus():
    dataset = load_dataset(DATA_ROOT)
    assert len(dataset.qrels["dev"]) == 324
    assert len(dataset.qrels["test"]) == 323
    assert len(dataset.query_ids["dev"]) == 324
    assert len(dataset.query_ids["test"]) == 323

    for split in ("dev", "test"):
        qids = dataset.query_ids[split]
        assert qids == sorted(qids)
        for qid in qids:
            assert qid in dataset.queries
            for doc_id, score in dataset.qrels[split][qid].items():
                assert isinstance(doc_id, str)
                assert isinstance(score, int)


@pytest.mark.data
def test_rag_queries_reproducibility_nfcorpus():
    dataset = load_dataset(DATA_ROOT)
    eligible, selected = select_rag_queries(
        dataset.qrels["test"], dataset.queries, n=10, seed=42
    )
    assert len(eligible) == 42
    assert selected == EXPECTED_RAG_QUERIES
