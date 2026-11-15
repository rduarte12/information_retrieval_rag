from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from src.data_loader import doc_text, load_dataset
from src.fusion import read_trec_run
from src.orchestrator import load_config
from src.preprocessing import preprocess


def select_rf7_cases(per_query_df: pd.DataFrame) -> list[dict[str, Any]]:
    """Select 3 sem_win and 3 lex_win queries deterministically by delta = NDCG_sem - NDCG_lex."""
    piv = per_query_df.pivot(index="query_id", columns="model", values="NDCG10")
    records: list[dict[str, Any]] = []
    for qid, row in piv.iterrows():
        ndcg_lex = float(row["lexical"])
        ndcg_sem = float(row["semantic"])
        records.append(
            {
                "query_id": str(qid),
                "ndcg_lex": ndcg_lex,
                "ndcg_sem": ndcg_sem,
                "delta": ndcg_sem - ndcg_lex,
            }
        )

    sem_top3 = sorted(records, key=lambda r: (-r["delta"], r["query_id"]))[:3]
    taken = {r["query_id"] for r in sem_top3}
    lex_pool = [r for r in records if r["query_id"] not in taken]
    lex_top3 = sorted(lex_pool, key=lambda r: (r["delta"], r["query_id"]))[:3]

    out: list[dict[str, Any]] = []
    for r in sem_top3:
        out.append({"group": "sem_win", **r})
    for r in lex_top3:
        out.append({"group": "lex_win", **r})
    return out


def select_rf10_cases(
    per_query_df: pd.DataFrame,
    hybrid_top10_rel_counts: Mapping[str, int],
) -> list[dict[str, Any]]:
    """Select gain1, gain2, nochange, and loss queries for RF-10."""
    piv = per_query_df.pivot(index="query_id", columns="model", values="NDCG10")
    records: list[dict[str, Any]] = []
    for qid, row in piv.iterrows():
        ndcg_hyb = float(row["hybrid"])
        ndcg_rer = float(row["reranking"])
        records.append(
            {
                "query_id": str(qid),
                "ndcg_hybrid": ndcg_hyb,
                "ndcg_rerank": ndcg_rer,
                "delta": ndcg_rer - ndcg_hyb,
            }
        )

    gains = sorted(records, key=lambda r: (-r["delta"], r["query_id"]))[:2]
    taken = {r["query_id"] for r in gains}

    negatives = [r for r in records if r["query_id"] not in taken and r["delta"] < 0.0]
    loss = sorted(negatives, key=lambda r: (r["delta"], r["query_id"]))[:1]
    if loss:
        taken.add(loss[0]["query_id"])

    nochange_pool = [
        r
        for r in records
        if r["query_id"] not in taken and hybrid_top10_rel_counts.get(r["query_id"], 0) >= 1
    ]
    nochange = sorted(nochange_pool, key=lambda r: (abs(r["delta"]), r["query_id"]))[:1]

    out: list[dict[str, Any]] = []
    if len(gains) >= 1:
        out.append({"case": "gain1", **gains[0]})
    if len(gains) >= 2:
        out.append({"case": "gain2", **gains[1]})
    if nochange:
        out.append({"case": "nochange", **nochange[0]})
    if loss:
        out.append({"case": "loss", **loss[0]})
    return out


def select_rf14_queries(unblinded_rows: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    """Select melhorou, pouca_diferenca, nao_ajudou by mean (B - A) over 3 criteria."""
    by_qc: dict[tuple[str, str], float] = {}
    qids: set[str] = set()
    for row in unblinded_rows:
        qid = str(row["query_id"])
        cond = str(row["condition"])
        qids.add(qid)
        mean3 = (
            int(row["fidelidade"]) + int(row["completude"]) + int(row["utilidade"])
        ) / 3.0
        by_qc[(qid, cond)] = mean3

    diffs = [
        (qid, by_qc[(qid, "B")] - by_qc[(qid, "A")])
        for qid in sorted(qids)
        if (qid, "A") in by_qc and (qid, "B") in by_qc
    ]
    melhorou = min(diffs, key=lambda x: (-x[1], x[0]))[0]
    nao_ajudou = min(diffs, key=lambda x: (x[1], x[0]))[0]
    remaining = [d for d in diffs if d[0] not in {melhorou, nao_ajudou}] or diffs
    pouca_diferenca = min(remaining, key=lambda x: (abs(x[1]), x[0]))[0]
    return {
        "melhorou": melhorou,
        "pouca_diferenca": pouca_diferenca,
        "nao_ajudou": nao_ajudou,
    }


def build_rf14_trace(
    selected_qids: Sequence[str],
    qrels: Mapping[str, Mapping[str, int]],
    runs_by_model: Mapping[str, Mapping[str, Sequence[tuple[str, float]]]],
    contexts: Mapping[str, Sequence[Mapping[str, Any]]],
) -> list[dict[str, Any]]:
    """Trace every relevant doc of selected RAG queries across all 4 pipeline stages."""
    rows: list[dict[str, Any]] = []
    for qid in selected_qids:
        rank_maps = {
            model: {doc_id: rank for rank, (doc_id, _) in enumerate(runs_by_model[model].get(qid, []), start=1)}
            for model in ("lexical", "semantic", "hybrid", "reranking")
        }
        ctx_ids = {str(item["doc_id"]) for item in contexts.get(qid, [])}
        rel_docs = sorted(doc_id for doc_id, grade in qrels.get(qid, {}).items() if grade > 0)
        for doc_id in rel_docs:
            rows.append(
                {
                    "query_id": qid,
                    "doc_id": doc_id,
                    "relevance": int(qrels[qid][doc_id]),
                    "rank_lexical": rank_maps["lexical"].get(doc_id, ""),
                    "rank_semantic": rank_maps["semantic"].get(doc_id, ""),
                    "rank_hybrid": rank_maps["hybrid"].get(doc_id, ""),
                    "rank_rerank": rank_maps["reranking"].get(doc_id, ""),
                    "in_context": doc_id in ctx_ids,
                }
            )
    return rows


def run_all_analyses(
    config_path: Path | str = Path("config/config.yaml"),
    artifacts_dir: Path | str = Path("artifacts"),
    results_dir: Path | str = Path("results"),
) -> dict[str, Any]:
    """Generate rf7_cases, rf7_top5, rf10_cases, rf10_moves (and rf14_trace if scores exist)."""
    art_dir = Path(artifacts_dir)
    res_dir = Path(results_dir)
    out_dir = art_dir / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(config_path)
    dataset = load_dataset(config["data"]["root"])
    qrels_test = dataset.qrels["test"]
    per_query_df = pd.read_csv(res_dir / "metrics_per_query.csv", dtype={"query_id": str, "model": str})

    runs = {
        "lexical": read_trec_run(art_dir / "runs" / "lexical_test.trec"),
        "semantic": read_trec_run(art_dir / "runs" / "semantic_test.trec"),
        "hybrid": read_trec_run(art_dir / "runs" / "hybrid_test.trec"),
        "reranking": read_trec_run(art_dir / "runs" / "reranking_test.trec"),
    }

    # RF-7
    rf7_cases = select_rf7_cases(per_query_df)
    for row in rf7_cases:
        row["query"] = dataset.queries[row["query_id"]]
    pd.DataFrame(
        rf7_cases,
        columns=["group", "query_id", "query", "ndcg_lex", "ndcg_sem", "delta"],
    ).to_csv(out_dir / "rf7_cases.csv", index=False, encoding="utf-8", lineterminator="\n")

    pp = config["preprocessing"]
    rf7_top5_rows: list[dict[str, Any]] = []
    for case in rf7_cases:
        qid = case["query_id"]
        q_terms = set(
            preprocess(
                dataset.queries[qid],
                lowercase=bool(pp["lowercase"]),
                remove_stopwords=bool(pp["remove_stopwords"]),
                stemmer=pp["stemmer"],
            )
        )
        for sys_name in ("lexical", "semantic"):
            for rank, (doc_id, score) in enumerate(runs[sys_name].get(qid, [])[:5], start=1):
                doc_tokens = preprocess(
                    doc_text(dataset.docs[doc_id]),
                    lowercase=bool(pp["lowercase"]),
                    remove_stopwords=bool(pp["remove_stopwords"]),
                    stemmer=pp["stemmer"],
                )
                counts = Counter(doc_tokens)
                tf_q = sum(counts[t] for t in q_terms)
                rf7_top5_rows.append(
                    {
                        "query_id": qid,
                        "system": sys_name,
                        "rank": rank,
                        "doc_id": doc_id,
                        "score": score,
                        "relevance": int(qrels_test.get(qid, {}).get(doc_id, 0)),
                        "tf_query_terms": tf_q,
                        "doc_len": len(doc_tokens),
                    }
                )
    pd.DataFrame(rf7_top5_rows).to_csv(
        out_dir / "rf7_top5.csv", index=False, encoding="utf-8", lineterminator="\n"
    )

    # RF-10
    hybrid_top10_rel = {
        qid: sum(1 for doc_id, _ in runs["hybrid"].get(qid, [])[:10] if qrels_test.get(qid, {}).get(doc_id, 0) > 0)
        for qid in dataset.query_ids["test"]
    }
    rf10_cases = select_rf10_cases(per_query_df, hybrid_top10_rel)
    pd.DataFrame(
        rf10_cases,
        columns=["case", "query_id", "ndcg_hybrid", "ndcg_rerank", "delta"],
    ).to_csv(out_dir / "rf10_cases.csv", index=False, encoding="utf-8", lineterminator="\n")

    ce_df = pd.read_parquet(art_dir / "rerank_scores_test.parquet")
    ce_lookup = {
        (str(r["query_id"]), str(r["doc_id"])): float(r["ce_score"])
        for _, r in ce_df.iterrows()
    }
    rf10_moves_rows: list[dict[str, Any]] = []
    for case in rf10_cases:
        qid = case["query_id"]
        h_ranks = {doc_id: r for r, (doc_id, _) in enumerate(runs["hybrid"].get(qid, []), start=1)}
        r_ranks = {doc_id: r for r, (doc_id, _) in enumerate(runs["reranking"].get(qid, []), start=1)}
        top10_union = sorted(
            {d for d, _ in runs["hybrid"].get(qid, [])[:10]}
            | {d for d, _ in runs["reranking"].get(qid, [])[:10]}
        )
        for doc_id in top10_union:
            rh = h_ranks.get(doc_id)
            rr = r_ranks.get(doc_id)
            if rh is not None and rr is not None and abs(rr - rh) >= 5:
                rf10_moves_rows.append(
                    {
                        "query_id": qid,
                        "doc_id": doc_id,
                        "rank_hybrid": rh,
                        "rank_rerank": rr,
                        "relevance": int(qrels_test.get(qid, {}).get(doc_id, 0)),
                        "ce_score": ce_lookup.get((qid, doc_id), ""),
                    }
                )
    pd.DataFrame(
        rf10_moves_rows,
        columns=["query_id", "doc_id", "rank_hybrid", "rank_rerank", "relevance", "ce_score"],
    ).to_csv(out_dir / "rf10_moves.csv", index=False, encoding="utf-8", lineterminator="\n")

    # RF-14 (if human evaluation scores exist)
    rf14_generated = False
    scores_path = art_dir / "rag" / "scores.csv"
    key_path = art_dir / "rag" / "blind_key.json"
    if scores_path.exists() and key_path.exists():
        from src.human_eval import unblind_and_summarize

        unblinded = unblind_and_summarize(
            scores_path, key_path, art_dir / "rag" / "human_eval_summary.csv"
        )
        selected = select_rf14_queries(unblinded)
        contexts = json.loads((art_dir / "rag" / "contexts.json").read_text(encoding="utf-8"))
        trace_rows = build_rf14_trace(list(selected.values()), qrels_test, runs, contexts)
        pd.DataFrame(trace_rows).to_csv(
            out_dir / "rf14_trace.csv", index=False, encoding="utf-8", lineterminator="\n"
        )
        rf14_generated = True

    return {
        "rf7_cases": len(rf7_cases),
        "rf7_top5": len(rf7_top5_rows),
        "rf10_cases": len(rf10_cases),
        "rf10_moves": len(rf10_moves_rows),
        "rf14_generated": rf14_generated,
    }


if __name__ == "__main__":
    summary = run_all_analyses()
    print(json.dumps(summary, indent=2))
