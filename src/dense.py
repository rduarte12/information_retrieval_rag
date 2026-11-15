from __future__ import annotations

import csv
import hashlib
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import transformers
import sentence_transformers
from sentence_transformers import SentenceTransformer

from src.data_loader import doc_text, load_dataset
from src.evaluation import evaluate_run

DEFAULT_MODEL_ID = "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"
DEFAULT_MAX_SEQ_LENGTH = 512
DEFAULT_BATCH_SIZE = 32
DEFAULT_TORCH_THREADS = 4


def model_slug(model_id: str) -> str:
    """Return a filesystem-safe slug for a HuggingFace model ID."""
    return model_id.replace("/", "--")


def top_k_from_scores(
    scores: np.ndarray,
    doc_ids: Sequence[str],
    k: int = 100,
) -> list[tuple[str, float]]:
    """Select top-k (doc_id, score) pairs sorted by (-score, doc_id).

    Handles ties at the k-th score boundary deterministically by including all
    candidates with score >= kth_score before sorting by (-score, doc_id).
    """
    if k <= 0 or len(doc_ids) == 0:
        return []
    n = len(doc_ids)
    if k >= n:
        indices = np.arange(n)
    else:
        kth_score = float(np.partition(scores, -k)[-k])
        indices = np.flatnonzero(scores >= kth_score)

    ranked = sorted(
        ((doc_ids[i], float(scores[i])) for i in indices),
        key=lambda item: (-item[1], item[0]),
    )
    return ranked[:k]


def validate_manifest(
    manifest: Mapping[str, Any],
    model_id: str,
    max_seq_length: int,
    revision: str | None = None,
) -> None:
    """Verify cached embedding manifest matches the requested configuration."""
    if manifest.get("model_id") != model_id:
        raise ValueError(
            f"Manifest model_id mismatch: expected {model_id!r}, got {manifest.get('model_id')!r}"
        )
    if manifest.get("max_seq_length") != max_seq_length:
        raise ValueError(
            f"Manifest max_seq_length mismatch: expected {max_seq_length}, "
            f"got {manifest.get('max_seq_length')}"
        )
    if revision is not None and manifest.get("revision") != revision:
        raise ValueError(
            f"Manifest revision mismatch: expected {revision!r}, got {manifest.get('revision')!r}"
        )


def write_trec_run(
    run: Mapping[str, Sequence[tuple[str, float]]],
    path: Path | str,
    tag: str = "semantic",
) -> None:
    """Write a run dictionary in standard 6-column TREC format."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for query_id in sorted(run.keys()):
        for rank, (doc_id, score) in enumerate(run[query_id], start=1):
            lines.append(f"{query_id} Q0 {doc_id} {rank} {score:.6f} {tag}")
    out_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _resolve_revision(model: SentenceTransformer, model_id: str, revision: str | None) -> str:
    if revision is not None:
        return revision
    commit_hash = getattr(model[0].auto_model.config, "_commit_hash", None)
    if isinstance(commit_hash, str) and len(commit_hash) == 40:
        return commit_hash
    from huggingface_hub import HfApi

    info = HfApi().model_info(model_id)
    if info.sha:
        return str(info.sha)
    raise RuntimeError(f"Could not resolve 40-char revision hash for {model_id!r}")


def _resolve_pooling(model: SentenceTransformer) -> str:
    if len(model) > 1 and hasattr(model[1], "get_pooling_mode_str"):
        return str(model[1].get_pooling_mode_str())
    return "mean"


def _compute_token_counts(
    tokenizer: Any,
    doc_ids: Sequence[str],
    texts: Sequence[str],
    max_seq_length: int,
    out_csv: Path,
) -> tuple[int, float]:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    n_truncated = 0
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["doc_id", "n_tokens", "truncated"])
        for doc_id, text in zip(doc_ids, texts):
            encoded = tokenizer(text, truncation=False, add_special_tokens=True)
            n_tokens = len(encoded["input_ids"])
            truncated = n_tokens > max_seq_length
            if truncated:
                n_truncated += 1
            writer.writerow([doc_id, n_tokens, str(truncated).lower()])
    pct_truncated = (100.0 * n_truncated / len(doc_ids)) if doc_ids else 0.0
    return n_truncated, pct_truncated


class DenseRetriever:
    """Bi-encoder semantic retriever using L2-normalized document embeddings."""

    def __init__(
        self,
        doc_ids: list[str],
        doc_emb: np.ndarray,
        model: SentenceTransformer,
        manifest: dict[str, Any],
    ) -> None:
        self.doc_ids = doc_ids
        self.doc_emb = doc_emb
        self.model = model
        self.manifest = manifest

    @classmethod
    def build_or_load(
        cls,
        docs: Mapping[str, Mapping[str, str]],
        embeddings_dir: Path | str,
        model_id: str = DEFAULT_MODEL_ID,
        revision: str | None = None,
        max_seq_length: int = DEFAULT_MAX_SEQ_LENGTH,
        batch_size: int = DEFAULT_BATCH_SIZE,
        torch_threads: int = DEFAULT_TORCH_THREADS,
    ) -> DenseRetriever:
        """Load cached document embeddings if valid, or encode and persist them."""
        emb_dir = Path(embeddings_dir)
        emb_path = emb_dir / "doc_emb.npy"
        ids_path = emb_dir / "doc_ids.json"
        manifest_path = emb_dir / "manifest.json"
        ntokens_path = emb_dir / "doc_ntokens.csv"

        torch.set_num_threads(torch_threads)

        if emb_path.exists() and ids_path.exists() and manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            validate_manifest(
                manifest,
                model_id=model_id,
                max_seq_length=max_seq_length,
                revision=revision,
            )
            doc_ids = json.loads(ids_path.read_text(encoding="utf-8"))
            doc_emb = np.load(emb_path).astype(np.float32, copy=False)
            model = SentenceTransformer(
                model_id,
                revision=manifest["revision"],
                device="cpu",
            )
            model.max_seq_length = max_seq_length
            return cls(doc_ids=doc_ids, doc_emb=doc_emb, model=model, manifest=manifest)

        emb_dir.mkdir(parents=True, exist_ok=True)
        model = SentenceTransformer(model_id, revision=revision, device="cpu")
        model.max_seq_length = max_seq_length
        resolved_revision = _resolve_revision(model, model_id, revision)

        doc_ids = sorted(docs.keys())
        texts = [doc_text(docs[doc_id]) for doc_id in doc_ids]

        n_truncated, pct_truncated = _compute_token_counts(
            model.tokenizer,
            doc_ids,
            texts,
            max_seq_length,
            ntokens_path,
        )

        t0 = time.perf_counter()
        doc_emb = model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        encode_time_s = time.perf_counter() - t0
        doc_emb = np.asarray(doc_emb, dtype=np.float32)

        np.save(emb_path, doc_emb)
        ids_path.write_text(json.dumps(doc_ids, indent=2), encoding="utf-8")
        npy_sha256 = hashlib.sha256(emb_path.read_bytes()).hexdigest()

        manifest: dict[str, Any] = {
            "model_id": model_id,
            "revision": resolved_revision,
            "library_versions": {
                "torch": torch.__version__,
                "transformers": transformers.__version__,
                "sentence_transformers": sentence_transformers.__version__,
            },
            "dim": int(doc_emb.shape[1]),
            "max_seq_length": int(max_seq_length),
            "pooling": _resolve_pooling(model),
            "normalized": True,
            "similarity": "cosine",
            "doc_representation": 'title + " " + text',
            "n_docs": len(doc_ids),
            "n_truncated": n_truncated,
            "pct_truncated": round(pct_truncated, 4),
            "encode_time_s": round(encode_time_s, 4),
            "hardware": f"{platform.machine()} ({platform.processor() or 'cpu'}), torch_threads={torch_threads}",
            "npy_sha256": npy_sha256,
        }
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return cls(doc_ids=doc_ids, doc_emb=doc_emb, model=model, manifest=manifest)

    def search(self, query: str, k: int = 100) -> list[tuple[str, float]]:
        """Retrieve top-k documents for a single raw query string."""
        q_emb = self.model.encode(
            query,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        q_vec = np.asarray(q_emb, dtype=np.float32)
        scores = self.doc_emb @ q_vec
        return top_k_from_scores(scores, self.doc_ids, k=k)

    def search_batch(
        self,
        queries: Mapping[str, str],
        k: int = 100,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> tuple[dict[str, list[tuple[str, float]]], float]:
        """Retrieve top-k documents for a mapping of query_id -> query_text.

        Returns (run_dict, avg_query_time_ms).
        """
        query_ids = sorted(queries.keys())
        if not query_ids:
            return {}, 0.0
        query_texts = [queries[qid] for qid in query_ids]

        t0 = time.perf_counter()
        q_emb = self.model.encode(
            query_texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        q_mat = np.asarray(q_emb, dtype=np.float32)
        scores_mat = q_mat @ self.doc_emb.T
        run = {
            qid: top_k_from_scores(scores_mat[idx], self.doc_ids, k=k)
            for idx, qid in enumerate(query_ids)
        }
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return run, elapsed_ms / len(query_ids)


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


def run_dense_pipeline(
    data_root: Path | str = Path("data/nfcorpus"),
    artifacts_dir: Path | str = Path("artifacts"),
    model_id: str = DEFAULT_MODEL_ID,
    max_seq_length: int = DEFAULT_MAX_SEQ_LENGTH,
    batch_size: int = DEFAULT_BATCH_SIZE,
    torch_threads: int = DEFAULT_TORCH_THREADS,
) -> dict[str, Any]:
    """Build/load dense embeddings, run dev (and record validation_dev.csv), then run test."""
    dataset = load_dataset(data_root)
    art_dir = Path(artifacts_dir)
    emb_dir = art_dir / "embeddings" / model_slug(model_id)

    retriever = DenseRetriever.build_or_load(
        dataset.docs,
        embeddings_dir=emb_dir,
        model_id=model_id,
        max_seq_length=max_seq_length,
        batch_size=batch_size,
        torch_threads=torch_threads,
    )

    dev_queries = {qid: dataset.queries[qid] for qid in sorted(dataset.qrels["dev"].keys())}
    dev_run, dev_ms = retriever.search_batch(dev_queries, k=100, batch_size=batch_size)
    write_trec_run(dev_run, art_dir / "runs" / "semantic_dev.trec", tag="semantic")

    _, dev_aggregate = evaluate_run(
        dev_run,
        dataset.qrels["dev"],
        dataset.query_ids["dev"],
        model="semantic",
    )
    _record_validation_dev(
        art_dir / "dev" / "validation_dev.csv",
        system="semantic",
        ndcg10_dev=dev_aggregate["NDCG@10"],
    )

    test_queries = {qid: dataset.queries[qid] for qid in sorted(dataset.qrels["test"].keys())}
    test_run, test_ms = retriever.search_batch(test_queries, k=100, batch_size=batch_size)
    write_trec_run(test_run, art_dir / "runs" / "semantic_test.trec", tag="semantic")

    return {
        "manifest": retriever.manifest,
        "dev_aggregate": dev_aggregate,
        "dev_query_time_ms": round(dev_ms, 3),
        "test_query_time_ms": round(test_ms, 3),
    }


if __name__ == "__main__":
    summary = run_dense_pipeline()
    print(json.dumps(summary, indent=2))
