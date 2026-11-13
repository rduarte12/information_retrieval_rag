# Contexto — Busca lexical do T2 (BM25 reaproveitado do T1)

> Documento de contexto do Trabalho Prático 2 (SCC0282, RI 2026). Prazo: 19/11/2026 23:59.
> Cobre apenas o **Requisito 3 (busca lexical)** e o **Requisito 2 (índice invertido)** do T2.

## 1. Decisão

O baseline lexical do T2 é o **BM25**, reaproveitado do T1 (`src/bm25.py` do repositório `rduarte12/information-retrieval-1`).

Justificativa (para o relatório):
- A implementação manual da fórmula foi avaliada como correta no T1 (feedback do professor) e foi validada novamente
  contra o novo índice (seção 4).
- O score do BM25 é uma soma por termo, o que se encaixa naturalmente em posting lists.
- O mesmo modelo lexical é usado em todos os experimentos seguintes (correção ortográfica, comparação com o semântico,
  híbrido, reranking e RAG).
- Os parâmetros (`k1`, `b`) só podem ser ajustados no conjunto **dev**, nunca no test.

## 2. O índice invertido foi reaproveitado ou refeito?

**Foi refeito do zero.** O T1 não tinha índice invertido: o `BM25` guardava um índice **direto**
(`self.tf = {doc_id: Counter(termo -> freq)}`) e o `rank()` pontuava os N documentos a cada consulta.

| Item | Situação |
|---|---|
| Fórmula do BM25 (`score`) | **Reaproveitada** sem alteração matemática |
| IDF (variante Lucene: `ln((N-df+0,5)/(df+0,5)+1)`) | **Reaproveitado**, agora pré-calculado por termo |
| Padrões `k1=1,2`, `b=0,75` e o grid (k1 ∈ {0,5; 1,2; 2,0} × b ∈ {0; 0,75; 1}) | **Reaproveitados** |
| Soma por ocorrência de termo repetido na consulta | **Mantida** (igual ao T1) |
| Estrutura de dados (`tf` por documento) | **Substituída** por posting lists `termo -> (doc_idx[], tf[])` |
| Loop `rank()` sobre todos os documentos | **Substituído** por acumulação só nos candidatos |
| Classe única `BM25` | **Dividida** em `InvertedIndex` (índice) + `BM25` (score) |

## 3. Alterações em relação ao T1

1. **Índice separado do modelo.** `InvertedIndex` guarda só estatísticas independentes do ranking (tf, df, tamanho dos
   documentos, `avgdl`). `BM25` consome o índice. Motivo: o T2 pede índice e modelo lexical como componentes distintos,
   cada um com suas estatísticas.
2. **Posting lists** `termo -> (doc_idx: int32[], tf: int32[])`, ordenadas por `doc_idx`; `doc_idx` segue a ordem de
   `doc_id` ordenado (desempate determinístico). `df` é derivado do tamanho da lista.
3. **Sem varredura por consulta.** São pontuados apenas os documentos que aparecem nas posting lists dos termos da
   consulta (união das listas). A soma por documento usa `np.unique` + `np.bincount`, sem alocar vetor de tamanho N.
4. **Construtor sem consultas.** O T1 recebia `processed_queries`; agora o índice recebe só os documentos e a consulta
   chega já tokenizada. Saem `rank(query_id)` e `search(raw_text, flags)`; entra `search(query_tokens, k, k1, b)`.
5. **`k1` e `b` por chamada.** Nada armazenado no índice depende de `k1`/`b`; o grid no dev roda sem reconstruir o índice.
   O fator `k1*(1-b+b*|D|/avgdl)` é calculado **uma vez por combinação (k1, b)** (cache) e só indexado pelos candidatos.
6. **IDF pré-calculado** (antes era recalculado, com `log`, para cada par termo×documento).
7. **Top-k com `heapq.nlargest`** (`k=100` por padrão; `k=None` devolve todos os candidatos).
8. **Documentos sem nenhum termo da consulta não entram no ranking** (o T1 devolvia a coleção inteira, com zeros em
   ordem arbitrária). Importante para o top-100 dos CSVs e para o RRF.
9. **Desempate determinístico:** score decrescente, depois `doc_id` crescente.
10. **Estatísticas exigidas pelo T2:** `vocab_size`, `total_postings`, `build_time`, `avg_query_time_ms`.

## 4. Validação (arquivo `tests/test_bm25.py`)

- Regressão contra o `BM25` do T1, em 4 conjuntos de documentos e consultas já processados (4 × 225 consultas = 900
  execuções): mesmos candidatos (documentos com score > 0), diferença máxima de score 2,8e-14, ordem idêntica em todas
  as posições sem empate de score.
- Testes unitários: saturação de tf, fórmula calculada à mão, bordas (consulta vazia, termo fora do vocabulário,
  documento vazio com `b=1`), termo repetido, desempate, `k=None`, grid sem reconstruir, não-varredura, estatísticas.
- Desempenho (1.400 documentos, top-100): 7,71 ms/consulta (T1) → 0,73 ms/consulta (novo).

## 5. Pendências relacionadas

- **Correção ortográfica:** `InvertedIndex.terms()` expõe o vocabulário do índice.
- **Escolha de `k1`/`b`:** feita no **dev**, nunca no test.
- **Relatório:** registrar a variante do IDF, a regra de termos repetidos, o desempate por `doc_id` lexicográfico e as
  estatísticas do índice (vocabulário, postings, tempo de construção, tempo médio por consulta).

## 6. Uso

```python
from inverted_index import InvertedIndex
from bm25 import BM25

index = InvertedIndex(processed_docs)          # {doc_id: list[str]} (tokens)
bm25  = BM25(index)                            # k1=1.2, b=0.75
ranking = bm25.search(query_tokens, k=100)     # [(doc_id, score), ...]

# grid no dev, sem reconstruir o índice
for k1 in (0.5, 1.2, 2.0):
    for b in (0, 0.75, 1):
        ranking = bm25.search(query_tokens, k=100, k1=k1, b=b)

print(index.stats(), bm25.avg_query_time_ms)
```

## 7. Código

### 7.1 Original do T1 (`src/bm25.py`, referência — **não usar no T2**)

```python
from collections import Counter
import math

class BM25:
    def __init__(self, processed_docs: dict, processed_queries: dict, k1=1.2, b=0.75):
        self.doc_ids            = list(processed_docs.keys())
        self.query_ids          = list(processed_queries.keys())
        self.processed_queries  = processed_queries
        self.k1, self.b         = k1, b
        self.N                  = len(processed_docs)

        # TF por documento: {doc_id => Counter(termo => frequencia)}
        self.tf = {doc_id: Counter(tokens) for doc_id, tokens in processed_docs.items()}

        # tamanho de cada documento
        self.doc_lens = {doc_id: len(tokens) for doc_id, tokens in processed_docs.items()}
        self.avgdl    = sum(self.doc_lens.values()) / self.N

        # DF por termo: quantos documentos contem o termo
        self.df = Counter()
        for tokens in processed_docs.values():
            for term in set(tokens):
                self.df[term] += 1

    def _idf(self, term: str) -> float:
        df = self.df.get(term, 0)
        return math.log((self.N - df + 0.5) / (df + 0.5) + 1)

    def _score(self, query_tokens: list, doc_id: str) -> float:
        score  = 0.0
        tf     = self.tf[doc_id]
        doc_len = self.doc_lens[doc_id]
        for term in query_tokens:
            f = tf.get(term, 0)
            idf = self._idf(term)
            num = f * (self.k1 + 1)
            den = f + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)
            score += idf * (num / den) if den > 0 else 0.0
        return score

    def rank(self, query_id: str) -> list[tuple]:
        tokens = self.processed_queries[query_id]
        scores = [(doc_id, self._score(tokens, doc_id)) for doc_id in self.doc_ids]
        return sorted(scores, key=lambda x: x[1], reverse=True)

    def search(self, raw_text: str, remove_stopwords: bool, stem: bool) -> list[tuple]:
        from preprocessing import preprocess
        tokens = preprocess(raw_text, remove_stopwords=remove_stopwords, stem=stem)
        scores = [(doc_id, self._score(tokens, doc_id)) for doc_id in self.doc_ids]
        return sorted(scores, key=lambda x: x[1], reverse=True)
```

### 7.2 Novo — `src/inverted_index.py`

```python
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
```

### 7.3 Novo — `src/bm25.py`

```python
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
```