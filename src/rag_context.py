from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.data_loader import load_dataset
from src.evaluation import evaluate_rag_retrieval
from src.fusion import read_trec_run
from src.orchestrator import guard_stage, load_config


def _load_rag_query_ids(rag_queries_path: Path) -> list[str]:
    payload = json.loads(rag_queries_path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [str(q) for q in payload]
    if "query_ids" in payload:
        return [str(q) for q in payload["query_ids"]]
    return [str(q) for q in payload.keys()]


def extract_rag_contexts(
    config_path: Path | str = Path("config/config.yaml"),
    artifacts_dir: Path | str = Path("artifacts"),
) -> dict[str, Any]:
    """Extract top-5 reranked documents for the 10 RAG queries and compute R5/P5/Hit5."""
    art_dir = Path(artifacts_dir)
    config = load_config(config_path)
    guard_stage("rag", split="test", config=config, artifacts_dir=art_dir)

    dataset = load_dataset(config["data"]["root"])
    rerank_run = read_trec_run(art_dir / "runs" / "reranking_test.trec")
    rag_qids = _load_rag_query_ids(art_dir / "rag_queries.json")

    contexts: dict[str, list[dict[str, Any]]] = {}
    for qid in rag_qids:
        top5 = rerank_run[qid][:5]
        contexts[qid] = [
            {
                "rank": rank,
                "doc_id": doc_id,
                "title": dataset.docs[doc_id].get("title", ""),
                "text": dataset.docs[doc_id].get("text", ""),
            }
            for rank, (doc_id, _) in enumerate(top5, start=1)
        ]

    rag_dir = art_dir / "rag"
    rag_dir.mkdir(parents=True, exist_ok=True)
    contexts_path = rag_dir / "contexts.json"
    contexts_path.write_text(json.dumps(contexts, indent=2, ensure_ascii=False), encoding="utf-8")

    rag_df = evaluate_rag_retrieval(rerank_run, dataset.qrels["test"], rag_qids)
    metrics_path = rag_dir / "rag_retrieval_metrics.csv"
    rag_df.to_csv(metrics_path, index=False, encoding="utf-8", lineterminator="\n")

    rag_agg = {
        "R@5": float(rag_df["R5"].mean()),
        "P@5": float(rag_df["P5"].mean()),
        "Hit@5": float(rag_df["Hit5"].mean()),
    }
    return {
        "n_queries": len(contexts),
        "rag_retrieval_aggregate": rag_agg,
        "contexts_path": str(contexts_path),
        "metrics_path": str(metrics_path),
    }


if __name__ == "__main__":
    summary = extract_rag_contexts()
    print(json.dumps(summary, indent=2))
