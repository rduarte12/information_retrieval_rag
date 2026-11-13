# src/inverted_index.py
"""
Índice invertido implementado do zero (Requisito 2 do T2).

Estrutura:
    postings[termo] = (doc_idx, tf)
        doc_idx : np.ndarray[int32], ordenado crescentemente
        tf      : np.ndarray[int32], tf[i] = frequência do termo em doc_idx[i]

O índice guarda apenas estatísticas independentes do modelo de ranking
(tf, df, tamanho dos documentos, avgdl). Nada aqui depende de k1/b do BM25.

doc_idx é a posição do documento em `doc_ids`, que é ordenado por doc_id.
Assim, ordenar por doc_idx equivale a ordenar por doc_id (desempate determinístico).
"""

import time
from collections import Counter

import numpy as np


class InvertedIndex:
    def __init__(self, processed_docs: dict):
        """
        processed_docs: {doc_id -> list[str]} com tokens já processados.
        """
        if not processed_docs:
            raise ValueError("processed_docs vazio")

        t0 = time.perf_counter()

        self.doc_ids = sorted(processed_docs)                    # doc_idx -> doc_id
        self.N       = len(self.doc_ids)
        self.doc_lens = np.array(
            [len(processed_docs[d]) for d in self.doc_ids], dtype=np.int64
        )
        total_len = int(self.doc_lens.sum())
        if total_len == 0:
            raise ValueError("todos os documentos estão vazios")
        self.avgdl = total_len / self.N

        # Passada única sobre a coleção: acumula (doc_idx, tf) por termo.
        # Como iteramos em ordem de doc_idx, as listas já saem ordenadas.
        tmp = {}
        for idx, doc_id in enumerate(self.doc_ids):
            for term, f in Counter(processed_docs[doc_id]).items():
                entry = tmp.get(term)
                if entry is None:
                    entry = tmp[term] = ([], [])
                entry[0].append(idx)
                entry[1].append(f)

        self.postings = {
            term: (np.array(docs, dtype=np.int32), np.array(tfs, dtype=np.int32))
            for term, (docs, tfs) in tmp.items()
        }
        self.build_time = time.perf_counter() - t0

    # --- acesso ---

    def get_postings(self, term: str):
        """Retorna (doc_idx[], tf[]) ou None se o termo não existe."""
        return self.postings.get(term)

    def df(self, term: str) -> int:
        p = self.postings.get(term)
        return 0 if p is None else len(p[0])

    def terms(self):
        """Vocabulário do índice (usado, por exemplo, pela correção ortográfica)."""
        return self.postings.keys()

    # --- estatísticas exigidas no relatório ---

    @property
    def vocab_size(self) -> int:
        return len(self.postings)

    @property
    def total_postings(self) -> int:
        return sum(len(p[0]) for p in self.postings.values())

    def stats(self) -> dict:
        return {
            "N_docs":         self.N,
            "avgdl":          self.avgdl,
            "vocab_size":     self.vocab_size,
            "total_postings": self.total_postings,
            "build_time_s":   self.build_time,
        }
