from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
import sys
from typing import Any

_ROOT = Path(__file__).resolve().parent.parent
for _p in (str(_ROOT), str(_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.bm25 import BM25  # noqa: E402
from src.data_loader import doc_text, load_dataset
from src.fusion import read_trec_run, rrf, write_trec_run
from src.inverted_index import InvertedIndex
from src.orchestrator import guard_stage, load_config, record_stage_result
from src.preprocessing import preprocess
from src.rerank import CrossEncoderReranker
from src.results_writer import (
    validate_results,
    write_metrics_per_query_csv,
    write_ranking_csv,
)


def run_test_pipeline(
    config_path: Path | str = Path("config/config.yaml"),
    artifacts_dir: Path | str = Path("artifacts"),
    results_dir: Path | str = Path("results"),
    force: bool = False,
    reason: str | None = None,
) -> dict[str, Any]:
    """Execute the single frozen test-split evaluation and write results CSVs."""
    art_dir = Path(artifacts_dir)
    res_dir = Path(results_dir)
    config = load_config(config_path)

    guard_stage(
        "test",
        split="test",
        config=config,
        artifacts_dir=art_dir,
        force=force,
        reason=reason,
    )

    dataset = load_dataset(config["data"]["root"])
    test_qids = sorted(dataset.query_ids["test"])

    # 1. Lexical (BM25) on test
    pp = config["preprocessing"]
    processed_docs = {
        doc_id: preprocess(
            doc_text(doc),
            lowercase=bool(pp["lowercase"]),
            remove_stopwords=bool(pp["remove_stopwords"]),
            stemmer=pp["stemmer"],
        )
        for doc_id, doc in dataset.docs.items()
    }
    index = InvertedIndex(processed_docs)
    bm25 = BM25(index, k1=float(config["bm25"]["k1"]), b=float(config["bm25"]["b"]))
    bm25_k = int(config["bm25"]["k"])

    lex_run: dict[str, list[tuple[str, float]]] = {}
    cand_rows: list[dict[str, Any]] = []
    t0 = time.perf_counter()
    for qid in test_qids:
        q_tokens = preprocess(
            dataset.queries[qid],
            lowercase=bool(pp["lowercase"]),
            remove_stopwords=bool(pp["remove_stopwords"]),
            stemmer=pp["stemmer"],
        )
        all_cands = bm25.search(q_tokens, k=None)
        if len(all_cands) < bm25_k:
            print(f"[AVISO] Consulta {qid} com {len(all_cands)} candidatos (< {bm25_k})")
        cand_rows.append({"query_id": qid, "n_candidates": len(all_cands)})
        lex_run[qid] = all_cands[:bm25_k]
    bm25_ms = (time.perf_counter() - t0) * 1000.0 / len(test_qids)

    lex_trec = art_dir / "runs" / "lexical_test.trec"
    write_trec_run(lex_run, lex_trec, tag="lexical")
    cand_csv = art_dir / "lexical_candidates_test.csv"
    with cand_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["query_id", "n_candidates"])
        writer.writeheader()
        writer.writerows(cand_rows)

    # 2. Semantic (loaded from S4 test run)
    sem_trec = art_dir / "runs" / "semantic_test.trec"
    sem_run = read_trec_run(sem_trec)

    # 3. Hybrid RRF
    t0 = time.perf_counter()
    hyb_run = rrf(
        [lex_run, sem_run],
        k=int(config["rrf"]["k"]),
        depth=int(config["rrf"]["depth"]),
        top_k=100,
    )
    rrf_ms = (time.perf_counter() - t0) * 1000.0 / len(test_qids)
    hyb_trec = art_dir / "runs" / "hybrid_test.trec"
    write_trec_run(hyb_run, hyb_trec, tag="hybrid")

    # 4. Cross-Encoder Reranking
    rerank_cfg = config["rerank"]
    reranker = CrossEncoderReranker(
        model_id=rerank_cfg["model_id"],
        revision=rerank_cfg.get("revision"),
        max_length=int(rerank_cfg["max_length"]),
        torch_threads=int(config["runtime"]["torch_threads"]),
    )
    rer_run, rer_df, rer_manifest = reranker.rerank_run(
        hyb_run,
        docs=dataset.docs,
        queries=dataset.queries,
        depth=int(rerank_cfg["depth"]),
    )
    rer_trec = art_dir / "runs" / "reranking_test.trec"
    write_trec_run(rer_run, rer_trec, tag="reranking")
    rer_df.to_parquet(art_dir / "rerank_scores_test.parquet", index=False)
    (art_dir / "rerank_manifest_test.json").write_text(
        json.dumps(rer_manifest, indent=2), encoding="utf-8"
    )

    # 5. Write CSVs and validate
    qrels_test = dataset.qrels["test"]
    depth = int(rerank_cfg["depth"])
    runs_by_model = {
        "lexical": lex_run,
        "semantic": sem_run,
        "hybrid": hyb_run,
        "reranking": rer_run,
    }
    for model_name, run_data in runs_by_model.items():
        write_ranking_csv(
            run_data,
            qrels_test,
            res_dir / f"results_{model_name}.csv",
            model=model_name,
            rerank_depth=depth,
        )

    aggregates = write_metrics_per_query_csv(
        runs_by_model,
        qrels_test,
        test_qids,
        res_dir / "metrics_per_query.csv",
    )
    validation = validate_results(
        results_dir=res_dir,
        qrels_test=qrels_test,
        query_ids_test=test_qids,
        artifacts_dir=art_dir,
    )

    summary = {
        "aggregates": aggregates,
        "timings_ms": {
            "bm25_query_ms": round(bm25_ms, 3),
            "rrf_query_ms": round(rrf_ms, 3),
            "rerank_query_ms": rer_manifest["rerank_time_ms"],
        },
        "rerank_manifest_test": rer_manifest,
        "validation": validation,
    }
    summary_path = art_dir / "test_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    record_stage_result("test", res_dir / "metrics_per_query.csv", config=config, artifacts_dir=art_dir)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run single frozen test evaluation (S8).")
    parser.add_argument("--force", action="store_true", help="Override test_run.lock mismatch")
    parser.add_argument("--reason", type=str, default=None, help="Reason for --force override")
    args = parser.parse_args()
    summary = run_test_pipeline(force=args.force, reason=args.reason)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
