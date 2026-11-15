from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import numpy as np
import pytest

from src.data_loader import doc_text, load_corpus
from src.dense import (
    DEFAULT_MAX_SEQ_LENGTH,
    DEFAULT_MODEL_ID,
    DenseRetriever,
    model_slug,
    top_k_from_scores,
    validate_manifest,
    write_trec_run,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = REPO_ROOT / "data" / "nfcorpus"
ARTIFACTS_DIR = REPO_ROOT / "artifacts"


def test_top_k_tiebreak_by_doc_id_at_boundary():
    doc_ids = ["MED-30", "MED-10", "MED-20", "MED-05", "MED-40"]
    # MED-30, MED-10, MED-20, MED-05 all tie at 0.5; MED-40 is 0.9
    scores = np.array([0.5, 0.5, 0.5, 0.5, 0.9], dtype=np.float32)

    result = top_k_from_scores(scores, doc_ids, k=3)
    assert [doc_id for doc_id, _ in result] == ["MED-40", "MED-05", "MED-10"]
    assert result[0][1] == pytest.approx(0.9)
    assert result[1][1] == pytest.approx(0.5)
    assert result[2][1] == pytest.approx(0.5)


def test_top_k_edge_cases():
    doc_ids = ["B", "A"]
    scores = np.array([0.4, 0.4], dtype=np.float32)

    assert top_k_from_scores(scores, doc_ids, k=0) == []
    assert [d for d, _ in top_k_from_scores(scores, doc_ids, k=10)] == ["A", "B"]


def test_validate_manifest_rejects_mismatched_max_seq_length():
    manifest = {
        "model_id": DEFAULT_MODEL_ID,
        "max_seq_length": 256,
        "revision": "a" * 40,
    }
    with pytest.raises(ValueError, match="max_seq_length"):
        validate_manifest(manifest, model_id=DEFAULT_MODEL_ID, max_seq_length=512)


def test_write_trec_run_format(tmp_path: Path):
    run = {
        "Q1": [("MED-2", 0.9), ("MED-1", 0.5)],
    }
    out = tmp_path / "test.trec"
    write_trec_run(run, out, tag="semantic")
    lines = out.read_text(encoding="utf-8").strip().splitlines()
    assert lines == [
        "Q1 Q0 MED-2 1 0.900000 semantic",
        "Q1 Q0 MED-1 2 0.500000 semantic",
    ]


@pytest.mark.data
def test_dense_embeddings_shape_norms_and_manifest():
    emb_dir = ARTIFACTS_DIR / "embeddings" / model_slug(DEFAULT_MODEL_ID)
    emb_path = emb_dir / "doc_emb.npy"
    ids_path = emb_dir / "doc_ids.json"
    manifest_path = emb_dir / "manifest.json"
    ntokens_path = emb_dir / "doc_ntokens.csv"

    assert emb_path.exists()
    assert ids_path.exists()
    assert manifest_path.exists()
    assert ntokens_path.exists()

    emb = np.load(emb_path)
    assert emb.shape == (3633, 384)
    assert emb.dtype == np.float32

    norms = np.linalg.norm(emb, axis=1)
    assert np.all(np.abs(norms - 1.0) <= 1e-5)

    doc_ids = json.loads(ids_path.read_text(encoding="utf-8"))
    corpus = load_corpus(DATA_ROOT / "corpus.jsonl")
    assert doc_ids == sorted(corpus.keys())

    with ntokens_path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 3633
    assert set(rows[0].keys()) == {"doc_id", "n_tokens", "truncated"}

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["model_id"] == DEFAULT_MODEL_ID
    assert re.fullmatch(r"[0-9a-f]{40}", manifest["revision"])
    assert set(manifest["library_versions"].keys()) >= {
        "torch",
        "transformers",
        "sentence_transformers",
    }
    assert manifest["dim"] == 384
    assert manifest["max_seq_length"] == DEFAULT_MAX_SEQ_LENGTH
    assert manifest["pooling"]
    assert manifest["normalized"] is True
    assert manifest["similarity"] == "cosine"
    assert manifest["doc_representation"] == 'title + " " + text'
    assert manifest["n_docs"] == 3633
    assert manifest["n_truncated"] >= 0
    assert manifest["pct_truncated"] >= 0.0
    assert manifest["encode_time_s"] > 0.0
    assert manifest["hardware"]
    assert re.fullmatch(r"[0-9a-f]{64}", manifest["npy_sha256"])


@pytest.mark.data
def test_dense_self_retrieval_top1():
    corpus = load_corpus(DATA_ROOT / "corpus.jsonl")
    emb_dir = ARTIFACTS_DIR / "embeddings" / model_slug(DEFAULT_MODEL_ID)
    retriever = DenseRetriever.build_or_load(corpus, embeddings_dir=emb_dir)

    sample_id = sorted(corpus.keys())[0]
    query = doc_text(corpus[sample_id])
    results = retriever.search(query, k=10)
    assert len(results) == 10
    assert results[0][0] == sample_id


@pytest.mark.data
def test_dense_trec_runs_and_validation_dev():
    dev_run = ARTIFACTS_DIR / "runs" / "semantic_dev.trec"
    test_run = ARTIFACTS_DIR / "runs" / "semantic_test.trec"
    val_csv = ARTIFACTS_DIR / "dev" / "validation_dev.csv"

    assert dev_run.exists()
    assert test_run.exists()
    assert val_csv.exists()

    dev_lines = dev_run.read_text(encoding="utf-8").strip().splitlines()
    test_lines = test_run.read_text(encoding="utf-8").strip().splitlines()
    assert len(dev_lines) == 324 * 100
    assert len(test_lines) == 323 * 100

    with val_csv.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    semantic_rows = [r for r in rows if r["system"] == "semantic"]
    assert len(semantic_rows) >= 1
    score = float(semantic_rows[-1]["ndcg10_dev"])
    assert 0.0 <= score <= 1.0
