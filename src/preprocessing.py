from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from typing import Mapping

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, SnowballStemmer

from src.data_loader import doc_text

TOKEN_RE = re.compile(r"[a-z0-9]+")
STOPWORDS_EN: frozenset[str] = frozenset(stopwords.words("english"))

_PORTER = PorterStemmer()
_SNOWBALL = SnowballStemmer("english")


def tokenize(text: str, lowercase: bool = True) -> list[str]:
    """Tokenize text into alphanumeric tokens without punctuation."""
    s = text.lower() if lowercase else text
    return TOKEN_RE.findall(s)


def surface_tokens(text: str) -> list[str]:
    """Return surface tokens (lowercased, no stopwords/stemming)."""
    return tokenize(text, lowercase=True)


def preprocess(
    text: str,
    lowercase: bool = True,
    remove_stopwords: bool = True,
    stemmer: str | None = "snowball",
) -> list[str]:
    """Preprocess text according to configuration."""
    if not isinstance(lowercase, bool):
        raise TypeError(
            "preprocess(): 'lowercase' deve ser bool. Para passar um dicionário de "
            "configuração, use preprocess(texto, **cfg)."
        )
    tokens = tokenize(text, lowercase=lowercase)
    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS_EN]
    if stemmer == "porter":
        tokens = [_PORTER.stem(t) for t in tokens]
    elif stemmer == "snowball":
        tokens = [_SNOWBALL.stem(t) for t in tokens]
    elif stemmer not in (None, "none", ""):
        raise ValueError(f"Unsupported stemmer: {stemmer!r}")
    return tokens


def surface_vocab(docs: Mapping[str, Mapping[str, str]]) -> Counter[str]:
    """Build surface vocabulary Counter (term -> document frequency) over corpus."""
    df: Counter[str] = Counter()
    for doc_id in sorted(docs.keys()):
        terms = set(surface_tokens(doc_text(docs[doc_id])))
        df.update(terms)
    return df


def config_hash(config: Mapping[str, object]) -> str:
    """Return deterministic SHA-256 hash of a preprocessing config."""
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
