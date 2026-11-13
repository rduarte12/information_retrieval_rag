"""
Testes-contrato do índice invertido (C-2 / TDS-2) e do BM25 (C-3 / TDS-3).

Recriados em 13/11/2026 (o arquivo original se perdeu). Usam corpus sintético, sem
depender do NFCorpus. A checagem de não-varredura sobre o NFCorpus real fica
para quando o data_loader (TDS-0) existir.

Fórmula (docs/t2_busca_lexical_bm25_contexto.md):
    score(D,Q) = Σ_t qtf(t) · IDF(t) · f·(k1+1) / (f + k1·(1 − b + b·|D|/avgdl))
    IDF(t)     = ln((N − df + 0,5) / (df + 0,5) + 1)
"""
import math
import random
import re
from pathlib import Path

import pytest

from bm25 import BM25
from bm25_t1 import BM25 as BM25_T1
from inverted_index import InvertedIndex

SRC = Path(__file__).resolve().parent.parent / "src"

DOCS = {
    "d1": ["cancer", "diet", "cancer"],
    "d2": ["diet", "fiber"],
    "d3": ["fiber", "fiber", "fiber", "fiber", "vitamin"],
    "d4": ["vitamin"],
}


def bm25_manual(docs, query, k1=1.2, b=0.75):
    """Implementação independente e literal da fórmula (oráculo)."""
    N = len(docs)
    avgdl = sum(len(t) for t in docs.values()) / N
    df = {}
    for toks in docs.values():
        for t in set(toks):
            df[t] = df.get(t, 0) + 1
    out = {}
    for d, toks in docs.items():
        s = 0.0
        for t in query:  # soma por ocorrência (qtf)
            f = toks.count(t)
            if f == 0:
                continue
            idf = math.log((N - df[t] + 0.5) / (df[t] + 0.5) + 1)
            s += idf * f * (k1 + 1) / (f + k1 * (1 - b + b * len(toks) / avgdl))
        if s > 0:
            out[d] = s
    return out


# ---------------------------------------------------------------- índice (TDS-2)

def test_postings_ordenadas_e_tf():
    idx = InvertedIndex(DOCS)
    docs, tf = idx.get_postings("fiber")
    assert [idx.doc_ids[i] for i in docs] == ["d2", "d3"]
    assert list(tf) == [1, 4]
    assert list(docs) == sorted(docs)


def test_df_e_termo_inexistente():
    idx = InvertedIndex(DOCS)
    assert idx.df("diet") == 2
    assert idx.df("inexistente") == 0
    assert idx.get_postings("inexistente") is None


def test_estatisticas():
    idx = InvertedIndex(DOCS)
    st = idx.stats()
    assert st["N_docs"] == 4
    assert st["avgdl"] == pytest.approx(11 / 4)
    assert st["vocab_size"] == len({t for v in DOCS.values() for t in v})
    assert st["total_postings"] == sum(len(set(v)) for v in DOCS.values())
    assert st["build_time_s"] >= 0


def test_entradas_invalidas():
    with pytest.raises(ValueError):
        InvertedIndex({})
    with pytest.raises(ValueError):
        InvertedIndex({"a": [], "b": []})


def test_sem_bibliotecas_de_indexacao_prontas():
    """Restrições C1/C2: nada pronto que construa ou consulte índice textual."""
    proibidas = r"rank_bm25|sklearn|whoosh|pyserini|lucene|elasticsearch|tantivy|bm25s|gensim|faiss"
    for nome in ("inverted_index.py", "bm25.py"):
        imports = [l for l in (SRC / nome).read_text(encoding="utf-8").splitlines()
                   if re.match(r"\s*(import|from)\s", l)]
        assert not any(re.search(proibidas, l) for l in imports), (nome, imports)


# ------------------------------------------------------------------ BM25 (TDS-3)

@pytest.mark.parametrize("query", [["cancer"], ["diet", "fiber"], ["fiber", "fiber"], ["vitamin", "cancer", "diet"]])
@pytest.mark.parametrize("k1,b", [(1.2, 0.75), (0.5, 0.0), (2.0, 1.0)])
def test_formula_contra_oraculo_manual(query, k1, b):
    got = dict(BM25(InvertedIndex(DOCS)).search(query, k=None, k1=k1, b=b))
    exp = bm25_manual(DOCS, query, k1, b)
    assert got.keys() == exp.keys()
    for d in exp:
        assert got[d] == pytest.approx(exp[d], rel=1e-12)


def test_saturacao_de_tf():
    docs = {f"d{f}": ["x"] * f + ["y"] * (10 - f) for f in range(1, 10)}
    docs["z"] = ["w"] * 10
    bm = BM25(InvertedIndex(docs), k1=1.2, b=0.0)
    s = dict(bm.search(["x"], k=None))
    vals = [s[f"d{f}"] for f in range(1, 10)]
    assert all(a < c for a, c in zip(vals, vals[1:]))                          # cresce com tf
    assert vals[-1] < bm.idf["x"] * (1.2 + 1)                                    # limitado por IDF·(k1+1)
    assert (vals[1] - vals[0]) > (vals[-1] - vals[-2])                           # ganhos decrescentes


def test_consulta_vazia_e_fora_do_vocabulario():
    bm = BM25(InvertedIndex(DOCS))
    assert bm.search([]) == []
    assert bm.search(["nada", "aqui"]) == []
    assert bm.last_n_candidates == 0


def test_documento_vazio_com_b_igual_a_1():
    docs = dict(DOCS, d5=[])
    res = BM25(InvertedIndex(docs)).search(["diet"], k=None, b=1.0)
    assert "d5" not in dict(res)
    assert all(math.isfinite(s) for _, s in res)


def test_termo_repetido_soma_por_ocorrencia():
    bm = BM25(InvertedIndex(DOCS))
    um = dict(bm.search(["diet"], k=None))
    dois = dict(bm.search(["diet", "diet"], k=None))
    for d in um:
        assert dois[d] == pytest.approx(2 * um[d], rel=1e-12)


def test_desempate_por_doc_id_crescente():
    docs = {"b": ["t", "u"], "a": ["t", "u"], "c": ["t", "u"], "z": ["v", "v"]}
    res = BM25(InvertedIndex(docs)).search(["t"], k=None)
    assert [d for d, _ in res] == ["a", "b", "c"]
    assert len({s for _, s in res}) == 1


def test_k_limita_e_k_none_devolve_todos_os_candidatos():
    bm = BM25(InvertedIndex(DOCS))
    assert len(bm.search(["diet", "fiber", "vitamin"], k=None)) == 4
    top2 = bm.search(["diet", "fiber", "vitamin"], k=2)
    assert len(top2) == 2
    assert top2 == bm.search(["diet", "fiber", "vitamin"], k=None)[:2]
    assert all(s > 0 for _, s in bm.search(["diet"], k=None))


def test_ordem_decrescente_de_score():
    res = BM25(InvertedIndex(DOCS)).search(["diet", "fiber", "vitamin", "cancer"], k=None)
    scores = [s for _, s in res]
    assert scores == sorted(scores, reverse=True)


def test_grade_k1_b_sem_reconstruir_o_indice():
    idx = InvertedIndex(DOCS)
    bm = BM25(idx)
    for k1 in (0.5, 1.2, 2.0):
        for b in (0, 0.75, 1):
            got = dict(bm.search(["diet", "fiber"], k=None, k1=k1, b=b))
            exp = bm25_manual(DOCS, ["diet", "fiber"], k1, b)
            assert got == pytest.approx(exp, rel=1e-12)
    assert bm.index is idx
    assert len(bm._norm_cache) == 9


def test_nao_varredura_so_pontua_candidatos():
    rng = random.Random(42)
    vocab = [f"t{i}" for i in range(300)]
    docs = {f"doc{i:04d}": rng.choices(vocab, k=rng.randint(5, 40)) for i in range(500)}
    idx = InvertedIndex(docs)
    bm = BM25(idx)
    for _ in range(50):
        q = rng.sample(vocab, 3)
        res = bm.search(q, k=None)
        union = set()
        for t in q:
            p = idx.get_postings(t)
            if p is not None:
                union |= {idx.doc_ids[i] for i in p[0]}
        assert bm.last_n_candidates == len(union)
        assert {d for d, _ in res} == union


def test_tempo_medio_por_consulta():
    bm = BM25(InvertedIndex(DOCS))
    assert bm.avg_query_time_ms == 0.0
    bm.search(["diet"])
    bm.search(["nada"])
    assert bm.n_queries == 2
    assert bm.avg_query_time_ms > 0


# ------------------------------------------------------ regressão contra o T1

def test_regressao_contra_bm25_do_t1():
    """Mesmos candidatos (score > 0), mesmos scores e mesma ordem fora de empates."""
    rng = random.Random(42)
    vocab = [f"w{i}" for i in range(400)]
    for rep in range(4):
        docs = {f"D{i:04d}": rng.choices(vocab, k=rng.randint(1, 60)) for i in range(400)}
        queries = {f"Q{j}": rng.choices(vocab, k=rng.randint(1, 6)) for j in range(50)}
        novo = BM25(InvertedIndex(docs))
        antigo = BM25_T1(docs, queries)
        for qid, q in queries.items():
            ref = [(d, s) for d, s in antigo.rank(qid) if s > 0]
            got = novo.search(q, k=None)
            assert {d for d, _ in got} == {d for d, _ in ref}
            ref_map = dict(ref)
            for d, s in got:
                assert s == pytest.approx(ref_map[d], abs=1e-12)
            # ordem idêntica onde não há empate de score
            ref_sorted = sorted(ref, key=lambda x: (-x[1], x[0]))
            assert [d for d, _ in got] == [d for d, _ in ref_sorted]
