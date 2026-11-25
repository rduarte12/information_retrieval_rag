"""
Mede o tempo médio de processamento das consultas na busca semântica (bi-encoder).

Carrega os embeddings já calculados em artifacts/embeddings/ e o mesmo modelo usado no
pipeline. Mede duas coisas sobre as 323 consultas do test:
  - consulta a consulta (encode da consulta + similaridade com os 3633 documentos + top 100);
  - em lote, do mesmo jeito que src.dense gera as runs.
Não grava runs nem altera nenhum arquivo de results/. Só escreve artifacts/semantic_timing.json.

Uso, na raiz do repositório:
    python scripts/medir_tempo_semantico.py
"""
from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for _p in (str(ROOT), str(ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.data_loader import load_dataset  # noqa: E402
from src.dense import DenseRetriever, model_slug  # noqa: E402
from src.orchestrator import load_config  # noqa: E402


def main() -> None:
    cfg = load_config(ROOT / "config" / "config.yaml")
    dense = cfg["dense"]
    threads = cfg.get("runtime", {}).get("torch_threads", 4)
    dataset = load_dataset(ROOT / cfg["data"]["root"])
    emb_dir = ROOT / "artifacts" / "embeddings" / model_slug(dense["model_id"])
    if not (emb_dir / "doc_emb.npy").exists():
        sys.exit(f"Embeddings não encontrados em {emb_dir}. Rode python -m src.dense antes.")

    retriever = DenseRetriever.build_or_load(
        dataset.docs,
        embeddings_dir=emb_dir,
        model_id=dense["model_id"],
        max_seq_length=dense["max_seq_length"],
        batch_size=dense["batch_size"],
        torch_threads=threads,
    )
    qids = sorted(dataset.qrels["test"].keys())
    queries = {q: dataset.queries[q] for q in qids}

    retriever.search(queries[qids[0]], k=100)  # aquecimento, fora da medição

    tempos = []
    for q in qids:
        t0 = time.perf_counter()
        retriever.search(queries[q], k=100)
        tempos.append((time.perf_counter() - t0) * 1000.0)

    _, lote_ms = retriever.search_batch(queries, k=100, batch_size=dense["batch_size"])

    out = {
        "model_id": dense["model_id"],
        "n_queries": len(qids),
        "split": "test",
        "torch_threads": threads,
        "consulta_a_consulta_ms": {
            "media": round(statistics.mean(tempos), 3),
            "mediana": round(statistics.median(tempos), 3),
            "max": round(max(tempos), 3),
        },
        "lote_media_por_consulta_ms": round(lote_ms, 3),
        "batch_size": dense["batch_size"],
    }
    dst = ROOT / "artifacts" / "semantic_timing.json"
    dst.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"Gravado em {dst.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
