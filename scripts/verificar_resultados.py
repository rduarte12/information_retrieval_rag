"""
Verificação independente dos 7 CSVs do RF-15 (não depende do código do projeto nem do NFCorpus).

Uso (na raiz do repositório):
    python scripts/verificar_resultados.py            # lê results/
    python scripts/verificar_resultados.py caminho/para/results

Confere schemas, contagens, ranks, invariantes do reranking, consistência das métricas com os
rankings (P@10 recalculado; IDCG implícito idêntico entre modelos), fórmula da recuperação no
typo_experiment e o protocolo do rag_experiment. Sai com código 1 se algo falhar.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RES = Path(sys.argv[1] if len(sys.argv) > 1 else "results")
SYS = ["lexical", "semantic", "hybrid", "reranking"]
RAG_IDS = ["PLAIN-68", "PLAIN-2450", "PLAIN-153", "PLAIN-2850", "PLAIN-2750",
           "PLAIN-270", "PLAIN-2470", "PLAIN-2430", "PLAIN-238", "PLAIN-3342"]
falhas = []


def check(cond, msg):
    print(("OK    " if cond else "FALHA ") + msg)
    if not cond:
        falhas.append(msg)


# --- rankings ---
R = {}
for s in SYS:
    df = pd.read_csv(RES / f"results_{s}.csv", dtype={"query_id": str, "doc_id": str})
    R[s] = df
    g = df.groupby("query_id")
    n = g.size()
    check(all(c in df.columns for c in ["query_id", "doc_id", "rank", "score", "relevance"]), f"{s}: colunas mínimas")
    check(n.max() <= 100, f"{s}: no máximo 100 docs por consulta")
    check(bool(g["rank"].apply(lambda r: list(r) == list(range(1, len(r) + 1))).all()), f"{s}: ranks contíguos desde 1")
    check(df.duplicated(["query_id", "doc_id"]).sum() == 0, f"{s}: sem doc repetido por consulta")
    check(bool((df.relevance >= 0).all()), f"{s}: relevance >= 0")
    if s != "reranking":
        check(bool(g["score"].apply(lambda x: (np.diff(x.values) <= 1e-12).all()).all()), f"{s}: score não crescente")
    if s == "lexical":
        print(f"      lexical: {df.query_id.nunique()} consultas presentes; {int((n < 100).sum())} com < 100 docs (D7: não completar)")
    else:
        check(df.query_id.nunique() == 323 and n.min() == 100, f"{s}: 323 consultas × 100 docs")

h = R["hybrid"].set_index(["query_id", "rank"])["doc_id"]
r = R["reranking"].set_index(["query_id", "rank"])["doc_id"]
check(bool((h[h.index.get_level_values(1) > 50] == r[r.index.get_level_values(1) > 50]).all()),
      "reranking: posições 51–100 iguais às do híbrido")
hs = R["hybrid"][R["hybrid"]["rank"] <= 50].groupby("query_id")["doc_id"].apply(frozenset)
rs = R["reranking"][R["reranking"]["rank"] <= 50].groupby("query_id")["doc_id"].apply(frozenset)
check(bool((hs == rs).all()), "reranking: top-50 é permutação do top-50 híbrido")

# --- métricas ---
m = pd.read_csv(RES / "metrics_per_query.csv", dtype={"query_id": str})
check(list(m.columns)[:6] == ["query_id", "model", "P10", "R10", "AP10", "NDCG10"], "metrics: colunas")
check(len(m) == 1292 and m.duplicated(["query_id", "model"]).sum() == 0, "metrics: 1.292 linhas, pares únicos")
mi = m.set_index(["model", "query_id"])
dcg = {}
for s in SYS:
    df = R[s]
    top = df[df["rank"] <= 10]
    p10 = top.groupby("query_id")["relevance"].apply(lambda x: (x > 0).sum() / 10).reindex(mi.loc[s].index).fillna(0)
    check(float((p10 - mi.loc[s]["P10"]).abs().max()) < 1e-9, f"{s}: P10 do CSV = P10 recalculado do ranking")
    t = top.assign(g=top.relevance / np.log2(top["rank"] + 1))
    dcg[s] = t.groupby("query_id")["g"].sum().reindex(mi.loc[s].index).fillna(0)
I = pd.DataFrame({s: (dcg[s] / mi.loc[s]["NDCG10"]).where(mi.loc[s]["NDCG10"] > 0) for s in SYS})
check(float(((I.max(axis=1) - I.min(axis=1)) / I.mean(axis=1)).max()) < 1e-9,
      "NDCG10: IDCG implícito idêntico entre os 4 modelos (ganho linear, ideal dos qrels)")
print(m.groupby("model")[["P10", "R10", "AP10", "NDCG10"]].mean().round(4).to_string())
print("      → MAP@10 do relatório = coluna AP10 acima")

# --- typo ---
t = pd.read_csv(RES / "typo_experiment.csv", dtype={"query_id": str})
check(len(t) == 10, "typo: 10 linhas")
check(set(t.tipo_erro) == {"insercao", "remocao", "substituicao", "transposicao"}, "typo: 4 tipos de erro")
check(bool((t.NDCG10_original > 0).all()), "typo: NDCG10_original > 0")
lx = mi.loc["lexical"]["NDCG10"]
check(np.allclose(t.NDCG10_original.values, lx.reindex(t.query_id).values), "typo: NDCG10_original = metrics (lexical)")
for suf in ["", "_sem"]:
    o, e, c = t[f"NDCG10{suf}_original"], t[f"NDCG10{suf}_erro"], t[f"NDCG10{suf}_corrigida"]
    rec = np.where(o - e > 0, (c - e) / (o - e), np.nan)
    check(np.allclose(np.nan_to_num(rec, nan=-9), np.nan_to_num(t[f"recuperacao{suf}"].astype(float), nan=-9)),
          f"typo: recuperacao{suf} segue a fórmula do RF-4")

# --- RAG ---
g = pd.read_csv(RES / "rag_experiment.csv", dtype=str, keep_default_na=False)
check(len(g) == 20 and sorted(g.query_id.unique()) == sorted(RAG_IDS), "rag: 20 linhas, 10 consultas oficiais")
check(set(g.condition) == {"A", "B"} and not g.duplicated(["query_id", "condition"]).any(), "rag: A e B por consulta")
A, B = g[g.condition == "A"], g[g.condition == "B"]
check(bool((A.context_doc_ids == "").all()) and not A.prompt.str.contains("MED-").any(), "rag: condição A sem contexto")
check(bool((B.context_doc_ids.str.split(";").str.len() == 5).all()), "rag: condição B com 5 doc_ids")
check(bool((A.rastreabilidade == "N/A").all()), "rag: rastreabilidade N/A em A")
for c in ["fidelidade", "completude", "utilidade"]:
    check(bool(g[c].isin(["0", "1", "2"]).all()), f"rag: {c} ∈ {{0,1,2}}")
check(bool(B.rastreabilidade.isin(["0", "1", "2"]).all()), "rag: rastreabilidade ∈ {0,1,2} em B")
check(all((g[c].str.strip() != "").all() for c in g.columns if c.startswith("just_")), "rag: justificativas preenchidas")
for col in ["model_id", "temperature", "seed", "max_output_tokens", "language"]:
    if col in g:
        check(g[col].nunique() == 1, f"rag: {col} igual nas 20 linhas")

print(f"\n{len(falhas)} falha(s).")
sys.exit(1 if falhas else 0)
