from __future__ import annotations

import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd
import torch
from sentence_transformers import CrossEncoder

from src.data_loader import doc_text, load_dataset
from src.evaluation import evaluate_run
from src.fusion import read_trec_run, write_trec_run
from src.orchestrator import guard_stage, load_config, record_stage_result, save_config

DEFAULT_RERANK_MODEL_ID = "cross-encoder/ms-marco-MiniLM-L2-v2"
DEFAULT_RERANK_MAX_LENGTH = 512
DEFAULT_RERANK_DEPTH = 50
DEFAULT_TORCH_THREADS = 4


def score_source_for_rank(rank: int, depth: int = DEFAULT_RERANK_DEPTH) -> str:
    """Return score_source label for a 1-based rank position."""
    return "cross_encoder" if rank <= depth else "hybrid_rrf"


def rerank_single_query(
    hybrid_ranking: Sequence[tuple[str, float]],
    ce_scores: Mapping[str, float],
    depth: int = DEFAULT_RERANK_DEPTH,
) -> list[tuple[str, float]]:
    """Reorder top-`depth` documents by Cross-Encoder score and keep tail untouched."""
    head = hybrid_ranking[:depth]
    tail = hybrid_ranking[depth:]

    reranked_head = sorted(
        ((doc_id, float(ce_scores[doc_id])) for doc_id, _ in head),
        key=lambda item: (-item[1], item[0]),
    )
    return reranked_head + list(tail)


def build_rerank_rows(
    query_id: str,
    ranking: Sequence[tuple[str, float]],
    depth: int = DEFAULT_RERANK_DEPTH,
) -> list[dict[str, Any]]:
    """Build tabular rows with explicit score_source per rank."""
    return [
        {
            "query_id": query_id,
            "doc_id": doc_id,
            "rank": rank,
            "score": float(score),
            "score_source": score_source_for_rank(rank, depth=depth),
        }
        for rank, (doc_id, score) in enumerate(ranking, start=1)
    ]


def _resolve_ce_revision(model: CrossEncoder, model_id: str, revision: str | None) -> str:
    if revision is not None:
        return revision
    commit_hash = getattr(model.model.config, "_commit_hash", None)
    if isinstance(commit_hash, str) and len(commit_hash) == 40:
        return commit_hash
    from huggingface_hub import HfApi

    info = HfApi().model_info(model_id)
    if info.sha:
        return str(info.sha)
    raise RuntimeError(f"Could not resolve 40-char revision hash for {model_id!r}")


class CrossEncoderReranker:
    """Two-stage Cross-Encoder reranker over top-k hybrid candidates."""

    def __init__(
        self,
        model_id: str = DEFAULT_RERANK_MODEL_ID,
        revision: str | None = None,
        max_length: int = DEFAULT_RERANK_MAX_LENGTH,
        torch_threads: int = DEFAULT_TORCH_THREADS,
    ) -> None:
        torch.set_num_threads(torch_threads)
        self.model_id = model_id
        self.max_length = max_length
        self.model = CrossEncoder(
            model_id,
            revision=revision,
            max_length=max_length,
            device="cpu",
            activation_fn=torch.nn.Identity(),
        )
        self.revision = _resolve_ce_revision(self.model, model_id, revision)

    def rerank_run(
        self,
        hybrid_run: Mapping[str, Sequence[tuple[str, float]]],
        docs: Mapping[str, Mapping[str, str]],
        queries: Mapping[str, str],
        depth: int = DEFAULT_RERANK_DEPTH,
        batch_size: int = 64,
    ) -> tuple[dict[str, list[tuple[str, float]]], pd.DataFrame, dict[str, Any]]:
        """Rerank top-`depth` documents for every query in `hybrid_run`."""
        query_ids = sorted(hybrid_run.keys())
        pair_meta: list[tuple[str, str, int, int, bool]] = []
        pair_texts: list[tuple[str, str]] = []

        for qid in query_ids:
            q_text = queries[qid]
            for h_rank, (doc_id, _) in enumerate(hybrid_run[qid][:depth], start=1):
                d_text = doc_text(docs[doc_id])
                encoded = self.model.tokenizer(
                    q_text,
                    d_text,
                    truncation=False,
                    add_special_tokens=True,
                )
                n_tokens = len(encoded["input_ids"])
                truncated = n_tokens > self.max_length
                pair_meta.append((qid, doc_id, h_rank, n_tokens, truncated))
                pair_texts.append((q_text, d_text))

        t0 = time.perf_counter()
        raw_scores = (
            self.model.predict(pair_texts, batch_size=batch_size, show_progress_bar=False)
            if pair_texts
            else []
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        ce_scores_by_query: dict[str, dict[str, float]] = {qid: {} for qid in query_ids}
        score_rows: list[dict[str, Any]] = []
        for (qid, doc_id, h_rank, n_tokens, truncated), ce_val in zip(pair_meta, raw_scores):
            score_f = float(ce_val)
            ce_scores_by_query[qid][doc_id] = score_f
            score_rows.append(
                {
                    "query_id": qid,
                    "doc_id": doc_id,
                    "hybrid_rank": h_rank,
                    "ce_score": score_f,
                    "n_tokens": n_tokens,
                    "truncated": truncated,
                    "model_rev": self.revision,
                }
            )

        reranked_run = {
            qid: rerank_single_query(hybrid_run[qid], ce_scores_by_query[qid], depth=depth)
            for qid in query_ids
        }
        scores_df = pd.DataFrame(
            score_rows,
            columns=[
                "query_id",
                "doc_id",
                "hybrid_rank",
                "ce_score",
                "n_tokens",
                "truncated",
                "model_rev",
            ],
        )

        n_pairs = len(pair_meta)
        n_truncated = sum(1 for _, _, _, _, trunc in pair_meta if trunc)
        pct_truncated = (100.0 * n_truncated / n_pairs) if n_pairs else 0.0
        rerank_time_ms = (elapsed_ms / len(query_ids)) if query_ids else 0.0

        manifest = {
            "model_id": self.model_id,
            "revision": self.revision,
            "max_length": int(self.max_length),
            "depth": int(depth),
            "activation": "identity",
            "n_pairs": n_pairs,
            "n_truncated": n_truncated,
            "pct_truncated": round(pct_truncated, 4),
            "rerank_time_ms": round(rerank_time_ms, 3),
        }
        return reranked_run, scores_df, manifest


def _record_validation_dev(csv_path: Path, system: str, ndcg10_dev: float) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []
    if csv_path.exists():
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            rows = [r for r in csv.DictReader(f) if r.get("system") != system]
    rows.append(
        {
            "system": system,
            "ndcg10_dev": f"{ndcg10_dev:.6f}",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    )
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["system", "ndcg10_dev", "timestamp"])
        writer.writeheader()
        writer.writerows(rows)


def validate_rerank_dev(
    config_path: Path | str = Path("config/config.yaml"),
    artifacts_dir: Path | str = Path("artifacts"),
    data_root: Path | str = Path("data/nfcorpus"),
) -> dict[str, Any]:
    """Run Cross-Encoder reranking on dev, save parquet/manifest/run, and record validation_dev."""
    cfg_path = Path(config_path)
    art_dir = Path(artifacts_dir)
    config = load_config(cfg_path)
    guard_stage("rerank_val", split="dev", config=config, artifacts_dir=art_dir)

    dataset = load_dataset(data_root)
    hybrid_dev = read_trec_run(art_dir / "runs" / "hybrid_dev.trec")

    rerank_cfg = config["rerank"]
    reranker = CrossEncoderReranker(
        model_id=rerank_cfg["model_id"],
        revision=rerank_cfg.get("revision"),
        max_length=int(rerank_cfg["max_length"]),
        torch_threads=int(config["runtime"]["torch_threads"]),
    )
    reranked_dev, scores_df, manifest = reranker.rerank_run(
        hybrid_dev,
        docs=dataset.docs,
        queries=dataset.queries,
        depth=int(rerank_cfg["depth"]),
    )

    parquet_path = art_dir / "rerank_scores_dev.parquet"
    scores_df.to_parquet(parquet_path, index=False)

    manifest_path = art_dir / "rerank_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    rerank_run_path = art_dir / "runs" / "reranking_dev.trec"
    write_trec_run(reranked_dev, rerank_run_path, tag="reranking")

    _, dev_agg = evaluate_run(
        reranked_dev,
        dataset.qrels["dev"],
        dataset.query_ids["dev"],
        model="reranking",
    )
    _record_validation_dev(
        art_dir / "dev" / "validation_dev.csv",
        system="reranking",
        ndcg10_dev=dev_agg["NDCG@10"],
    )

    config["rerank"]["revision"] = reranker.revision
    save_config(config, cfg_path)
    record_stage_result("rerank_val", rerank_run_path, config=config, artifacts_dir=art_dir)

    return {
        "manifest": manifest,
        "dev_aggregate": dev_agg,
    }


if __name__ == "__main__":
    summary = validate_rerank_dev()
    print(json.dumps(summary, indent=2))
