# Implementação ORIGINAL do BM25 do T1 (copiada sem alterações de docs/t2_busca_lexical_bm25_contexto.md §7.1).
# Uso exclusivo como ORÁCULO de regressão em tests/test_bm25.py. Não usar no pipeline do T2.

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
