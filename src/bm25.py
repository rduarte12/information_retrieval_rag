# src/bm25.py
"""
BM25 sobre o índice invertido (Requisito 3 do T2).

    score(D,Q) = Σ_{t∈Q} qtf(t) · IDF(t) · f(t,D)·(k1+1) / ( f(t,D) + k1·(1 − b + b·|D|/avgdl) )

    IDF(t) = ln( (N − df + 0,5) / (df + 0,5) + 1 )     (variante do Lucene; nunca negativa)

Decisões:
  - qtf(t): termo repetido na consulta soma uma vez por ocorrência (igual ao T1).
  - Só são pontuados os documentos que aparecem nas posting lists dos termos da
    consulta. Não há varredura sobre os N documentos por consulta.
  - Desempate: score decrescente, depois doc_id crescente (via doc_idx).
  - k1 e b não afetam o índice: podem variar por chamada (grid no dev sem reconstruir).
"""

import heapq
import math
import time
from collections import Counter

import numpy as np

from inverted_index import InvertedIndex


class BM25:
    def __init__(self, index: InvertedIndex, k1: float = 1.2, b: float = 0.75):
        self.index = index
        self.k1, self.b = k1, b

        # IDF pré-calculado por termo (depende só de N e df).
        N = index.N
        self.idf = {
            t: math.log((N - index.df(t) + 0.5) / (index.df(t) + 0.5) + 1)
            for t in index.terms()
        }

        # Fator de normalização de comprimento por (k1, b): calculado uma vez por
        # combinação de parâmetros (não por consulta) e apenas indexado pelos candidatos.
        self._norm_cache = {}

        # Contadores de desempenho
        self.n_queries = 0
        self.total_query_time = 0.0
        self.last_n_candidates = 0

    def _norm(self, k1: float, b: float) -> np.ndarray:
        key = (k1, b)
        arr = self._norm_cache.get(key)
        if arr is None:
            arr = k1 * (1 - b + b * self.index.doc_lens / self.index.avgdl)
            self._norm_cache[key] = arr
        return arr

    def search(self, query_tokens: list, k: int = 100, k1: float = None, b: float = None) -> list:
        """
        query_tokens: tokens já processados, como os dos documentos indexados.
        k:            tamanho do ranking; None devolve todos os candidatos.
        retorna:      [(doc_id, score), ...] por score decrescente, desempate por doc_id.
        """
        t0 = time.perf_counter()
        k1 = self.k1 if k1 is None else k1
        b  = self.b  if b  is None else b
        norm = self._norm(k1, b)

        doc_parts, contrib_parts = [], []
        for term, qtf in Counter(query_tokens).items():
            p = self.index.get_postings(term)
            if p is None:                      # termo fora do vocabulário
                continue
            docs, tf = p
            tf = tf.astype(np.float64)
            contrib = qtf * self.idf[term] * tf * (k1 + 1) / (tf + norm[docs])
            doc_parts.append(docs)
            contrib_parts.append(contrib)

        if not doc_parts:
            self.last_n_candidates = 0
            self._tick(t0)
            return []

        all_docs    = np.concatenate(doc_parts)
        all_contrib = np.concatenate(contrib_parts)
        # Soma por documento candidato (sem alocar vetor de tamanho N)
        cand, inv = np.unique(all_docs, return_inverse=True)
        scores = np.bincount(inv, weights=all_contrib)
        self.last_n_candidates = len(cand)

        order = range(len(cand))
        key = lambda i: (scores[i], -cand[i])
        top = sorted(order, key=key, reverse=True) if k is None else heapq.nlargest(k, order, key=key)

        result = [(self.index.doc_ids[cand[i]], float(scores[i])) for i in top]
        self._tick(t0)
        return result

    def _tick(self, t0: float):
        self.n_queries += 1
        self.total_query_time += time.perf_counter() - t0

    # --- estatísticas exigidas no relatório ---

    @property
    def avg_query_time_ms(self) -> float:
        return 1000 * self.total_query_time / self.n_queries if self.n_queries else 0.0
