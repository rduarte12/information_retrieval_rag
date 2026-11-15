from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from src.evaluation import evaluate_run
from src.rerank import DEFAULT_RERANK_DEPTH, score_source_for_rank


def write_ranking_csv(
    run: Mapping[str, Sequence[tuple[str, float]]],
    qrels: Mapping[str, Mapping[str, int]],
    out_path: Path | str,
    model: str,
    rerank_depth: int = DEFAULT_RERANK_DEPTH,
    top_k: int = 100,
) -> int:
    """Write a ranking CSV in UTF-8 with '.' decimals per TDS §2.1-2.4."""
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    is_rerank = model == "reranking"
    fieldnames = ["query_id", "doc_id", "rank", "score", "relevance"]
    if is_rerank:
        fieldnames.append("score_source")

    n_rows = 0
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for qid in sorted(run.keys()):
            qrels_q = qrels.get(qid, {})
            for rank, (doc_id, score) in enumerate(run[qid][:top_k], start=1):
                row: dict[str, Any] = {
                    "query_id": qid,
                    "doc_id": doc_id,
                    "rank": rank,
                    "score": f"{float(score):.17g}",
                    "relevance": int(qrels_q.get(doc_id, 0)),
                }
                if is_rerank:
                    row["score_source"] = score_source_for_rank(rank, depth=rerank_depth)
                writer.writerow(row)
                n_rows += 1
    return n_rows


def write_metrics_per_query_csv(
    runs_by_model: Mapping[str, Mapping[str, Sequence[tuple[str, float]]]],
    qrels: Mapping[str, Mapping[str, int]],
    query_ids: Sequence[str],
    out_path: Path | str,
) -> dict[str, dict[str, float]]:
    """Evaluate all models and write results/metrics_per_query.csv per TDS §2.5."""
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    frames: list[pd.DataFrame] = []
    aggregates: dict[str, dict[str, float]] = {}
    for model in sorted(runs_by_model.keys()):
        per_q, agg = evaluate_run(runs_by_model[model], qrels, query_ids, model=model)
        frames.append(per_q)
        aggregates[model] = agg

    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(path, index=False, encoding="utf-8", lineterminator="\n")
    return aggregates


def _validate_single_ranking_csv(
    csv_path: Path,
    model: str,
    qrels_test: Mapping[str, Mapping[str, int]],
    query_ids_set: set[str],
    require_exact_100: bool = False,
) -> int:
    if not csv_path.exists():
        raise ValueError(f"Missing ranking CSV: {csv_path}")

    df = pd.read_csv(csv_path, dtype={"query_id": str, "doc_id": str})
    expected_cols = ["query_id", "doc_id", "rank", "score", "relevance"]
    if model == "reranking":
        expected_cols.append("score_source")
    if list(df.columns) != expected_cols:
        raise ValueError(f"{csv_path} columns mismatch: {list(df.columns)} != {expected_cols}")

    for qid, group in df.groupby("query_id", sort=False):
        if qid not in query_ids_set:
            raise ValueError(f"{csv_path} contains unknown query_id {qid!r}")
        ranks = group["rank"].tolist()
        if ranks != list(range(1, len(ranks) + 1)):
            raise ValueError(f"{csv_path} ranks for {qid!r} are not contiguous 1..n_q")
        if len(ranks) > 100:
            raise ValueError(f"{csv_path} has >100 rows for {qid!r}")
        if require_exact_100 and len(ranks) != 100:
            raise ValueError(f"{csv_path} must have 100 rows for {qid!r}, got {len(ranks)}")
        if group["doc_id"].nunique() != len(group):
            raise ValueError(f"{csv_path} has duplicate doc_id for {qid!r}")

        qrels_q = qrels_test.get(qid, {})
        expected_rel = [int(qrels_q.get(d, 0)) for d in group["doc_id"]]
        if group["relevance"].tolist() != expected_rel:
            raise ValueError(f"{csv_path} relevance mismatch for {qid!r}")

        if model == "reranking":
            sources = group["score_source"].tolist()
            expected_src = [score_source_for_rank(r) for r in ranks]
            if sources != expected_src:
                raise ValueError(f"{csv_path} score_source mismatch for {qid!r}")
            for src, sub in group.groupby("score_source", sort=False):
                scores = sub["score"].tolist()
                if any(scores[i] < scores[i + 1] - 1e-12 for i in range(len(scores) - 1)):
                    raise ValueError(
                        f"{csv_path} scores not non-increasing in {src!r} for {qid!r}"
                    )
        else:
            scores = group["score"].tolist()
            if any(scores[i] < scores[i + 1] - 1e-12 for i in range(len(scores) - 1)):
                raise ValueError(f"{csv_path} scores not non-increasing for {qid!r}")

    return len(df)


def validate_results(
    results_dir: Path | str,
    qrels_test: Mapping[str, Mapping[str, int]],
    query_ids_test: Sequence[str],
    artifacts_dir: Path | str | None = None,
    check_models: Sequence[str] = ("lexical", "semantic", "hybrid", "reranking"),
    require_semantic_exact_100: bool = True,
) -> dict[str, Any]:
    """Validate generated results CSVs against TDS §2 schemas and invariants."""
    res_dir = Path(results_dir)
    qids_set = set(query_ids_test)
    counts: dict[str, int] = {}

    for model in check_models:
        counts[f"{model}_rows"] = _validate_single_ranking_csv(
            res_dir / f"results_{model}.csv",
            model=model,
            qrels_test=qrels_test,
            query_ids_set=qids_set,
            require_exact_100=(model == "semantic" and require_semantic_exact_100),
        )

    metrics_path = res_dir / "metrics_per_query.csv"
    if metrics_path.exists():
        df_m = pd.read_csv(metrics_path, dtype={"query_id": str, "model": str})
        expected_m_cols = ["query_id", "model", "P10", "R10", "AP10", "NDCG10"]
        if list(df_m.columns) != expected_m_cols:
            raise ValueError(f"metrics_per_query.csv columns mismatch: {list(df_m.columns)}")
        if len(df_m) != len(query_ids_test) * len(check_models):
            raise ValueError(
                f"metrics_per_query.csv expected {len(query_ids_test) * len(check_models)} rows, "
                f"got {len(df_m)}"
            )
        if df_m.duplicated(subset=["query_id", "model"]).any():
            raise ValueError("metrics_per_query.csv has duplicate (query_id, model) pairs")
        counts["metrics_rows"] = len(df_m)

    if artifacts_dir is not None:
        typo_path = res_dir / "typo_experiment.csv"
        if not typo_path.exists():
            raise ValueError(f"Missing {typo_path}")
        df_typo = pd.read_csv(typo_path, dtype={"query_id": str})
        if len(df_typo) != 10:
            raise ValueError(f"typo_experiment.csv expected 10 rows, got {len(df_typo)}")
        counts["typo_rows"] = len(df_typo)

    return counts
