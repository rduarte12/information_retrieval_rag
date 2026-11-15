from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.data_loader import load_qrels
from src.evaluation import evaluate_run
from src.orchestrator import guard_stage, load_config, record_stage_result, save_config


def read_trec_run(path: Path | str) -> dict[str, list[tuple[str, float]]]:
    """Read a 6-column TREC run file into {query_id: [(doc_id, score), ...]}."""
    grouped: dict[str, list[tuple[int, str, float]]] = defaultdict(list)
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            parts = stripped.split()
            qid, _, doc_id, rank_str, score_str = parts[:5]
            grouped[qid].append((int(rank_str), doc_id, float(score_str)))

    run: dict[str, list[tuple[str, float]]] = {}
    for qid, items in grouped.items():
        items.sort(key=lambda x: (x[0], x[1]))
        run[qid] = [(doc_id, score) for _, doc_id, score in items]
    return run


def write_trec_run(
    run: Mapping[str, Sequence[tuple[str, float]]],
    path: Path | str,
    tag: str = "hybrid",
) -> None:
    """Write a TREC run preserving high-precision float scores for RRF."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for query_id in sorted(run.keys()):
        for rank, (doc_id, score) in enumerate(run[query_id], start=1):
            lines.append(f"{query_id} Q0 {doc_id} {rank} {score:.15g} {tag}")
    out_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def rrf_single_query(
    rankings: Sequence[Sequence[tuple[str, float]]],
    k: int = 60,
    depth: int = 100,
    top_k: int = 100,
) -> list[tuple[str, float]]:
    """Compute Reciprocal Rank Fusion for one query across multiple rankings.

    Rank starts at 1; absent documents contribute 0; ties break by doc_id ascending.
    """
    scores: dict[str, float] = defaultdict(float)
    for ranking in rankings:
        for rank, (doc_id, _) in enumerate(ranking[:depth], start=1):
            scores[doc_id] += 1.0 / (k + rank)

    ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    return ordered[:top_k]


def rrf(
    runs: Sequence[Mapping[str, Sequence[tuple[str, float]]]],
    k: int = 60,
    depth: int = 100,
    top_k: int = 100,
) -> dict[str, list[tuple[str, float]]]:
    """Fuse multiple runs per query using Reciprocal Rank Fusion."""
    all_qids = sorted({qid for run in runs for qid in run.keys()})
    return {
        qid: rrf_single_query(
            [run.get(qid, []) for run in runs],
            k=k,
            depth=depth,
            top_k=top_k,
        )
        for qid in all_qids
    }


def tune_rrf_dev(
    k_grid: Sequence[int] = (10, 60, 100),
    config_path: Path | str = Path("config/config.yaml"),
    artifacts_dir: Path | str = Path("artifacts"),
    data_root: Path | str = Path("data/nfcorpus"),
) -> dict[str, Any]:
    """Evaluate RRF k grid on dev, update config.yaml with best k, and export hybrid_dev.trec."""
    cfg_path = Path(config_path)
    art_dir = Path(artifacts_dir)
    config = load_config(cfg_path)
    guard_stage("rrf", split="dev", config=config, artifacts_dir=art_dir)

    depth = int(config["rrf"]["depth"])
    lex_run = read_trec_run(art_dir / "runs" / "lexical_dev.trec")
    sem_run = read_trec_run(art_dir / "runs" / "semantic_dev.trec")
    qrels_dev = load_qrels(Path(data_root) / "qrels" / "dev.tsv")
    query_ids_dev = sorted(qrels_dev.keys())

    grid_rows: list[dict[str, Any]] = []
    fused_by_k: dict[int, dict[str, list[tuple[str, float]]]] = {}
    for k in k_grid:
        fused = rrf([lex_run, sem_run], k=int(k), depth=depth, top_k=100)
        fused_by_k[int(k)] = fused
        _, agg = evaluate_run(fused, qrels_dev, query_ids_dev, model=f"rrf_k{k}")
        grid_rows.append(
            {
                "k": int(k),
                "ndcg10_dev": f"{agg['NDCG@10']:.6f}",
                "n_queries": len(query_ids_dev),
                "aggregate": agg,
            }
        )

    csv_path = art_dir / "dev" / "rrf_k_dev.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["k", "ndcg10_dev", "n_queries"])
        writer.writeheader()
        for row in grid_rows:
            writer.writerow(
                {
                    "k": row["k"],
                    "ndcg10_dev": row["ndcg10_dev"],
                    "n_queries": row["n_queries"],
                }
            )

    best = min(grid_rows, key=lambda r: (-float(r["ndcg10_dev"]), r["k"]))
    best_k = int(best["k"])

    config["rrf"]["k"] = best_k
    save_config(config, cfg_path)

    hybrid_dev_path = art_dir / "runs" / "hybrid_dev.trec"
    write_trec_run(fused_by_k[best_k], hybrid_dev_path, tag="hybrid")
    record_stage_result("rrf", csv_path, config=config, artifacts_dir=art_dir)
    record_stage_result("hybrid_dev", hybrid_dev_path, config=config, artifacts_dir=art_dir)

    return {
        "best_k": best_k,
        "grid": [
            {"k": r["k"], "ndcg10_dev": float(r["ndcg10_dev"]), "aggregate": r["aggregate"]}
            for r in grid_rows
        ],
    }


if __name__ == "__main__":
    summary = tune_rrf_dev()
    print(json.dumps(summary, indent=2))
