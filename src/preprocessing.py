# src/preprocessing.py
"""
Pré-processador lexical (C-1 / TDS-1).

Implementa tokenização determinística sem pontuação (D8 / TDS-DEC-13),
remoção de stopwords, stemming configurável (Porter / Snowball / nenhum),
extração de vocabulário de superfície (para C-4) e hash de configuração.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from functools import lru_cache
from typing import Mapping

import nltk
import numpy as np
from nltk.stem import PorterStemmer, SnowballStemmer

_TOKEN_RE = re.compile(r"[a-z0-9]+")


@lru_cache(maxsize=1)
def _english_stopwords() -> frozenset[str]:
    try:
        return frozenset(nltk.corpus.stopwords.words("english"))
    except LookupError:
        nltk.download("stopwords", quiet=True)
        return frozenset(nltk.corpus.stopwords.words("english"))


@lru_cache(maxsize=4)
def _get_stemmer(name: str):
    if name == "porter":
        return PorterStemmer()
    if name == "snowball":
        return SnowballStemmer("english")
    raise ValueError(f"Stemmer desconhecido: {name!r}")


def tokenize(text: str) -> list[str]:
    """Tokenização determinística em minúsculas, descartando toda pontuação (D8)."""
    return _TOKEN_RE.findall(text.lower())


def surface_tokens(text: str) -> list[str]:
    """Tokens de superfície (antes de stopwords e stemming), via mesmo tokenize."""
    return tokenize(text)


def preprocess(text: str, cfg: Mapping) -> list[str]:
    """Aplica tokenização, stopwords e stemming conforme o dicionário cfg."""
    tokens = tokenize(text)
    if cfg.get("remove_stopwords", False):
        sw = _english_stopwords()
        tokens = [t for t in tokens if t not in sw]

    stemmer_name = cfg.get("stemmer")
    if stemmer_name and str(stemmer_name).lower() != "none":
        stemmer = _get_stemmer(str(stemmer_name).lower())
        tokens = [stemmer.stem(t) for t in tokens]

    return tokens


def surface_vocab(docs: Mapping[str, str]) -> Counter[str]:
    """Calcula a frequência em documentos (DF) dos tokens de superfície."""
    vocab: Counter[str] = Counter()
    for text in docs.values():
        for term in set(surface_tokens(text)):
            vocab[term] += 1
    return vocab


def config_hash(cfg: Mapping) -> str:
    """SHA-256 determinístico da configuração e versões de bibliotecas."""
    payload = {
        "cfg": dict(cfg),
        "versions": {
            "nltk": nltk.__version__,
            "numpy": np.__version__,
        },
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()
