# scripts/run_s2_dev.py
"""
Executa a avaliação de pré-processamento e busca lexical BM25 no split DEV (Sprint S2).

Gera:
  - artifacts/dev/preprocessing_dev.csv
  - artifacts/runs/lexical_dev.trec
  - artifacts/lexical_candidates_dev.csv
  - artifacts/index_stats_dev.json
  - artifacts/vocab/surface_{cfg_hash}.json
  - config/config.yaml (com a melhor configuração no dev, frozen=false)
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import ir_measures
import yaml
from ir_measures import nDCG

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from bm25 import BM25
from data_loader import doc_text, load_dataset
from inverted_index import InvertedIndex
from preprocessing import config_hash, preprocess, surface_vocab


def write_trec_run(path: Path, run: dict[str, list[tuple[str, float]]], tag: str = "bm25") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for qid in sorted(run.keys()):
            for rank, (doc_id, score) in enumerate(run[qid], start=1):
                fh.write(f"{qid} Q0 {doc_id} {rank} {score:.6f} {tag}\n")


def eval_ndcg10(qrels: dict[str, dict[str, int]], run: dict[str, list[tuple[str, float]]]) -> float:
    run_dict = {qid: dict(items) for qid, items in run.items()}
    res = ir_measures.calc_aggregate([nDCG @ 10], qrels, run_dict)
    return float(res[nDCG @ 10])


def main() -> None:
    data_root = ROOT / "data" / "nfcorpus"
    artifacts_dir = ROOT / "artifacts"
    dataset = load_dataset(data_root)

    raw_docs = {doc_id: doc_text(doc) for doc_id, doc in dataset.docs.items()}
    dev_qids = dataset.query_ids["dev"]
    dev_qrels = dataset.qrels["dev"]

    candidates = [
        ("raw", {"lowercase": True, "remove_stopwords": False, "stemmer": None}),
        ("stop", {"lowercase": True, "remove_stopwords": True, "stemmer": None}),
        ("porter", {"lowercase": True, "remove_stopwords": False, "stemmer": "porter"}),
        ("stop_porter", {"lowercase": True, "remove_stopwords": True, "stemmer": "porter"}),
        ("snowball", {"lowercase": True, "remove_stopwords": False, "stemmer": "snowball"}),
        ("stop_snowball", {"lowercase": True, "remove_stopwords": True, "stemmer": "snowball"}),
    ]

    dev_dir = artifacts_dir / "dev"
    dev_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    best_cfg_id = None
    best_cfg = None
    best_ndcg = -1.0
    best_artifacts = None

    for cfg_id, cfg in candidates:
        processed_docs = {doc_id: preprocess(txt, cfg) for doc_id, txt in raw_docs.items()}
        index = InvertedIndex(processed_docs)
        bm25 = BM25(index, k1=1.2, b=0.75)

        run: dict[str, list[tuple[str, float]]] = {}
        cand_counts: dict[str, int] = {}
        for qid in dev_qids:
            q_tokens = preprocess(dataset.queries[qid], cfg)
            ranking = bm25.search(q_tokens, k=100)
            run[qid] = ranking
            cand_counts[qid] = bm25.last_n_candidates

        ndcg10 = eval_ndcg10(dev_qrels, run)
        rows.append({
            "config_id": cfg_id,
            "lowercase": cfg["lowercase"],
            "remove_stopwords": cfg["remove_stopwords"],
            "stemmer": cfg["stemmer"] or "none",
            "ndcg10_dev": round(ndcg10, 6),
            "n_queries": len(dev_qids),
        })
        print(f"Config {cfg_id:15s} | NDCG@10 (dev) = {ndcg10:.6f} | vocab = {index.vocab_size}")

        if ndcg10 > best_ndcg:
            best_ndcg = ndcg10
            best_cfg_id = cfg_id
            best_cfg = cfg
            best_artifacts = (index, bm25, run, cand_counts)

    # Salva preprocessing_dev.csv
    csv_path = dev_dir / "preprocessing_dev.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["config_id", "lowercase", "remove_stopwords", "stemmer", "ndcg10_dev", "n_queries"],
        )
        writer.writeheader()
        writer.writerows(rows)

    assert best_cfg is not None and best_artifacts is not None
    best_index, best_bm25, best_run, best_cands = best_artifacts
    cfg_h = config_hash(best_cfg)

    print(f"\nMelhor configuração no dev: {best_cfg_id} (NDCG@10 = {best_ndcg:.6f}, hash={cfg_h[:12]})")

    # Salva run lexical do dev
    write_trec_run(artifacts_dir / "runs" / "lexical_dev.trec", best_run, tag="bm25")

    # Registra candidatos por consulta e avisa < 100 (D7 / TDS-DEC-4)
    cand_path = artifacts_dir / "lexical_candidates_dev.csv"
    under_100 = []
    with cand_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["query_id", "n_candidates"])
        for qid in dev_qids:
            n_c = best_cands[qid]
            writer.writerow([qid, n_c])
            if n_c < 100:
                under_100.append((qid, n_c))

    if under_100:
        print(f"[AVISO D7] {len(under_100)} consultas do dev tiveram menos de 100 candidatos:")
        for qid, n_c in under_100[:10]:
            print(f"  - {qid}: {n_c} candidatos")
        if len(under_100) > 10:
            print(f"  ... e mais {len(under_100) - 10} consultas.")

    # Salva index_stats_dev.json (RF-2)
    stats = best_index.stats()
    stats["avg_query_time_ms"] = best_bm25.avg_query_time_ms
    stats["n_queries"] = best_bm25.n_queries
    stats["config_id"] = best_cfg_id
    stats["config_hash"] = cfg_h
    (artifacts_dir / "index_stats_dev.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    # Salva vocabulário de superfície (C-4 / TDS-1)
    vocab_dir = artifacts_dir / "vocab"
    vocab_dir.mkdir(parents=True, exist_ok=True)
    s_vocab = surface_vocab(raw_docs)
    (vocab_dir / f"surface_{cfg_h}.json").write_text(
        json.dumps(dict(s_vocab.most_common()), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    # Cria/atualiza config/config.yaml (conforme TDS §0.6)
    config_dir = ROOT / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_data = {
        "data": {"root": "data/nfcorpus"},
        "preprocessing": best_cfg,
        "bm25": {"k1": 1.2, "b": 0.75, "k": 100},
        "dense": {
            "model_id": "sentence-transformers/multi-qa-MiniLM-L6-cos-v1",
            "revision": None,
            "max_seq_length": 512,
            "batch_size": 16,
        },
        "rrf": {"k": 60, "depth": 100},
        "rerank": {
            "model_id": "cross-encoder/ms-marco-MiniLM-L2-v2",
            "revision": None,
            "max_length": 512,
            "depth": 50,
        },
        "spell": {"k": 2, "jaccard_min": 0.3, "max_edit": 2, "distance": "osa"},
        "llm": {
            "model_id": "gemini-3.1-flash-lite",
            "temperature": 0.0,
            "seed": 42,
            "max_output_tokens": 1024,
            "thinking": None,
            "language": "Portuguese",
        },
        "runtime": {"torch_threads": 4},
        "frozen": False,
    }
    (config_dir / "config.yaml").write_text(
        yaml.safe_dump(config_data, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    print(f"config/config.yaml salvo com preprocessing={best_cfg_id}, frozen=False.")


if __name__ == "__main__":
    main()
