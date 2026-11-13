# src/data_loader.py
"""
Carregador do dataset NFCorpus no formato BEIR (C-0 / TDS-0).

Lê corpus.jsonl, queries.jsonl e qrels/{dev,test}.tsv preservando IDs como str,
fornece a representação padronizada de documento (TDS-DEC-2) e gera os
artefatos de estatísticas e consultas sorteadas para o experimento RAG (Req. 12).
"""

from __future__ import annotations

import json
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


@dataclass
class NFCorpusDataset:
    docs: dict[str, dict[str, str]]
    queries: dict[str, str]
    qrels: dict[str, dict[str, int]]
    query_ids: dict[str, list[str]]

    def doc_text(self, doc_id: str) -> str:
        return doc_text(self.docs[doc_id])


def doc_text(doc: Mapping[str, str]) -> str:
    """Concatena title e text conforme TDS-DEC-2."""
    title = doc.get("title", "").strip()
    text = doc.get("text", "").strip()
    return f"{title} {text}".strip()


def load_corpus(path: Path | str) -> dict[str, dict[str, str]]:
    """Lê corpus.jsonl devolvendo {doc_id: {"title": str, "text": str}}."""
    docs: dict[str, dict[str, str]] = {}
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            doc_id = str(row["_id"])
            docs[doc_id] = {
                "title": str(row.get("title", "")),
                "text": str(row.get("text", "")),
            }
    return docs


def load_queries(path: Path | str) -> dict[str, str]:
    """Lê queries.jsonl devolvendo {query_id: text}."""
    queries: dict[str, str] = {}
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            queries[str(row["_id"])] = str(row.get("text", ""))
    return queries


def load_qrels(path: Path | str) -> dict[str, dict[str, int]]:
    """Lê qrels/{split}.tsv pulando o cabeçalho, mantendo IDs como str e graus int."""
    qrels: dict[str, dict[str, int]] = {}
    with Path(path).open("r", encoding="utf-8") as fh:
        for idx, line in enumerate(fh):
            stripped = line.strip()
            if not stripped:
                continue
            parts = stripped.split("\t")
            if idx == 0 and parts[0].lower() == "query-id":
                continue
            qid, doc_id, score = str(parts[0]), str(parts[1]), int(parts[2])
            qrels.setdefault(qid, {})[doc_id] = score
    return qrels


def load_dataset(root: Path | str) -> NFCorpusDataset:
    """Carrega corpus, queries e splits dev/test de um diretório NFCorpus (BEIR)."""
    root_path = Path(root)
    docs = load_corpus(root_path / "corpus.jsonl")
    queries = load_queries(root_path / "queries.jsonl")
    qrels = {
        "dev": load_qrels(root_path / "qrels" / "dev.tsv"),
        "test": load_qrels(root_path / "qrels" / "test.tsv"),
    }
    query_ids = {split: sorted(qrels[split].keys()) for split in ("dev", "test")}
    return NFCorpusDataset(docs=docs, queries=queries, qrels=qrels, query_ids=query_ids)


def select_rag_queries(
    test_qrels: Mapping[str, Mapping[str, int]],
    queries: Mapping[str, str],
    n: int = 10,
    seed: int = 42,
) -> tuple[list[str], list[str]]:
    """Reproduz o sorteio determinístico do Req. 12 sobre consultas terminadas em '?'."""
    eligible = [q for q in sorted(test_qrels.keys()) if queries[q].strip().endswith("?")]
    sampled = random.Random(seed).sample(eligible, n)
    return eligible, sampled


def compute_data_stats(
    docs: Mapping[str, Mapping[str, str]],
    queries: Mapping[str, str],
    qrels: Mapping[str, Mapping[str, Mapping[str, int]]],
) -> dict:
    """Calcula estatísticas do corpus e distribuição de relevância por split."""
    splits_stats: dict[str, dict] = {}
    for split, split_qrels in qrels.items():
        grade_counts: Counter[int] = Counter()
        rel_per_query: list[int] = []
        zero_rel_queries: list[str] = []

        for qid in sorted(split_qrels.keys()):
            judgments = split_qrels[qid]
            n_rel = 0
            for grade in judgments.values():
                grade_counts[grade] += 1
                if grade > 0:
                    n_rel += 1
            rel_per_query.append(n_rel)
            if n_rel == 0:
                zero_rel_queries.append(qid)

        n_queries = len(split_qrels)
        splits_stats[split] = {
            "n_queries": n_queries,
            "n_judgments": sum(grade_counts.values()),
            "grade_distribution": {str(g): grade_counts[g] for g in sorted(grade_counts)},
            "n_queries_zero_rel": len(zero_rel_queries),
            "zero_rel_query_ids": zero_rel_queries,
            "avg_rel_per_query": (sum(rel_per_query) / n_queries) if n_queries else 0.0,
        }

    return {
        "n_docs": len(docs),
        "n_queries_total": len(queries),
        "splits": splits_stats,
    }


def generate_artifacts(
    data_root: Path | str = Path("data/nfcorpus"),
    artifacts_dir: Path | str = Path("artifacts"),
) -> tuple[dict, dict]:
    """Gera artifacts/data_stats.json e artifacts/rag_queries.json."""
    dataset = load_dataset(data_root)
    out_dir = Path(artifacts_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stats = compute_data_stats(dataset.docs, dataset.queries, dataset.qrels)
    (out_dir / "data_stats.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    eligible, selected = select_rag_queries(
        dataset.qrels["test"], dataset.queries, n=10, seed=42
    )
    rag_payload = {
        "seed": 42,
        "n_eligible": len(eligible),
        "query_ids": selected,
        "queries": {qid: dataset.queries[qid] for qid in selected},
    }
    (out_dir / "rag_queries.json").write_text(
        json.dumps(rag_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return stats, rag_payload


if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent
    stats, rag = generate_artifacts(root / "data" / "nfcorpus", root / "artifacts")
    print(f"Documentos: {stats['n_docs']} | Consultas totais: {stats['n_queries_total']}")
    for split, sp in stats["splits"].items():
        print(
            f"  {split}: {sp['n_queries']} consultas, {sp['n_judgments']} julgamentos, "
            f"graus={sp['grade_distribution']}, zero_rel={sp['n_queries_zero_rel']}, "
            f"media_rel={sp['avg_rel_per_query']:.2f}"
        )
    print(f"RAG ({rag['n_eligible']} elegíveis): {rag['query_ids']}")
