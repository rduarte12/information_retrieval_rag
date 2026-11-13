# tests/test_preprocessing.py
"""
Testes-contrato para C-1 (Pré-processador lexical, TDS-1 / Sprint S2)
e integração com C-2/C-3 sobre o NFCorpus real.
"""

import re
from pathlib import Path

import pytest

from bm25 import BM25
from data_loader import doc_text, load_dataset
from inverted_index import InvertedIndex
from preprocessing import (
    config_hash,
    preprocess,
    surface_tokens,
    surface_vocab,
    tokenize,
)

ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = ROOT / "data" / "nfcorpus"


# ---------------------------------------------------------------------------
# Testes unitários (TDS-1)
# ---------------------------------------------------------------------------


def test_tokenize_examples_from_tds1():
    """Exemplos obrigatórios da TDS-DEC-13 (sem pontuação, lowercase)."""
    assert tokenize("Cancer, diet. (Omega-3)!") == ["cancer", "diet", "omega", "3"]
    assert tokenize("vitamin B12; 1.5 mg") == ["vitamin", "b12", "1", "5", "mg"]
    assert tokenize("...") == []
    assert tokenize("don't / [anti-inflammatory]: 'test'") == [
        "don",
        "t",
        "anti",
        "inflammatory",
        "test",
    ]


def test_surface_tokens_uses_same_tokenize():
    text = "Vitamin B12, Omega-3 & Cancer!"
    assert surface_tokens(text) == tokenize(text)


def test_preprocess_configurations():
    text = "The patients are eating apples and running."
    cfg_raw = {"remove_stopwords": False, "stemmer": None}
    cfg_stop = {"remove_stopwords": True, "stemmer": None}
    cfg_stem = {"remove_stopwords": False, "stemmer": "porter"}
    cfg_both = {"remove_stopwords": True, "stemmer": "porter"}

    assert preprocess(text, cfg_raw) == [
        "the",
        "patients",
        "are",
        "eating",
        "apples",
        "and",
        "running",
    ]
    assert preprocess(text, cfg_stop) == ["patients", "eating", "apples", "running"]
    assert preprocess(text, cfg_stem) == [
        "the",
        "patient",
        "are",
        "eat",
        "appl",
        "and",
        "run",
    ]
    assert preprocess(text, cfg_both) == ["patient", "eat", "appl", "run"]


def test_surface_vocab_document_frequency():
    docs = {
        "D1": "Cancer diet. Cancer study!",
        "D2": "Diet and health.",
    }
    vocab = surface_vocab(docs)
    # Frequência em documentos (DF), não TF
    assert vocab["cancer"] == 1
    assert vocab["diet"] == 2
    assert vocab["health"] == 1


def test_config_hash_changes_with_config():
    c1 = {"lowercase": True, "remove_stopwords": True, "stemmer": "porter"}
    c2 = {"lowercase": True, "remove_stopwords": False, "stemmer": "porter"}
    c3 = {"lowercase": True, "remove_stopwords": True, "stemmer": "snowball"}
    c1_reordered = {"stemmer": "porter", "remove_stopwords": True, "lowercase": True}

    assert config_hash(c1) == config_hash(c1_reordered)
    assert config_hash(c1) != config_hash(c2)
    assert config_hash(c1) != config_hash(c3)
    assert len(config_hash(c1)) == 64  # SHA-256 hex


# ---------------------------------------------------------------------------
# Testes sobre os dados reais do NFCorpus (@pytest.mark.data)
# ---------------------------------------------------------------------------


@pytest.mark.data
def test_nfcorpus_index_terms_strictly_alphanumeric_and_no_full_scan():
    dataset = load_dataset(DATA_ROOT)
    cfg = {"lowercase": True, "remove_stopwords": True, "stemmer": "porter"}
    processed = {
        doc_id: preprocess(doc_text(doc), cfg) for doc_id, doc in dataset.docs.items()
    }
    index = InvertedIndex(processed)
    pattern = re.compile(r"[a-z0-9]+")

    for term in index.terms():
        assert pattern.fullmatch(term), f"Termo inválido no índice: {term!r}"

    # Verifica não-varredura no NFCorpus real
    bm25 = BM25(index)
    q_tokens = preprocess("omega 3 cancer", cfg)
    res = bm25.search(q_tokens, k=100)
    assert len(res) > 0
    assert bm25.last_n_candidates < index.N
