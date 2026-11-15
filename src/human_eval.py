from __future__ import annotations

import csv
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence


def generate_blind_sheet(
    responses: Sequence[Mapping[str, Any]],
    contexts: Mapping[str, Sequence[Mapping[str, Any]]],
    out_sheet: Path | str,
    out_key: Path | str,
    seed: int = 42,
) -> None:
    """Generate shuffled blind evaluation sheet (without condition) and key."""
    items = list(responses)
    rng = random.Random(seed)
    rng.shuffle(items)

    sheet_path = Path(out_sheet)
    key_path = Path(out_key)
    sheet_path.parent.mkdir(parents=True, exist_ok=True)
    key_path.parent.mkdir(parents=True, exist_ok=True)

    blind_key: dict[str, str] = {}
    fieldnames = [
        "item_id",
        "query_id",
        "question",
        "answer",
        "top5_context",
        "fidelidade",
        "just_fidelidade",
        "completude",
        "just_completude",
        "utilidade",
        "just_utilidade",
        "rastreabilidade",
        "just_rastreabilidade",
    ]
    with sheet_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for idx, rec in enumerate(items, start=1):
            item_id = f"ITEM-{idx:02d}"
            qid = rec["query_id"]
            blind_key[item_id] = rec["condition"]
            ctx_text = "\n\n".join(
                f"[{c['doc_id']}] {c.get('title', '')}\n{c.get('text', '')}".strip()
                for c in contexts.get(qid, [])
            )
            writer.writerow(
                {
                    "item_id": item_id,
                    "query_id": qid,
                    "question": rec.get("question", ""),
                    "answer": rec["answer"],
                    "top5_context": ctx_text,
                    "fidelidade": "",
                    "just_fidelidade": "",
                    "completude": "",
                    "just_completude": "",
                    "utilidade": "",
                    "just_utilidade": "",
                    "rastreabilidade": "",
                    "just_rastreabilidade": "",
                }
            )

    key_path.write_text(json.dumps(blind_key, indent=2), encoding="utf-8")


def unblind_and_summarize(
    scores_csv: Path | str,
    blind_key_json: Path | str,
    out_summary_csv: Path | str,
) -> list[dict[str, Any]]:
    """Unblind scores, force rastreabilidade='N/A' for condition A, and compute averages."""
    key = json.loads(Path(blind_key_json).read_text(encoding="utf-8"))
    with Path(scores_csv).open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    unblinded: list[dict[str, Any]] = []
    scores_by_cond_crit: dict[tuple[str, str], list[int]] = defaultdict(list)

    for row in rows:
        item_id = row["item_id"]
        cond = key[item_id]
        rec = dict(row)
        rec["condition"] = cond

        for crit in ("fidelidade", "completude", "utilidade"):
            val = int(rec[crit])
            if val not in (0, 1, 2):
                raise ValueError(f"Invalid score {val} for {item_id} {crit}")
            if not rec[f"just_{crit}"].strip():
                raise ValueError(f"Empty justification for {item_id} {crit}")
            scores_by_cond_crit[(cond, crit)].append(val)

        if cond == "A":
            rec["rastreabilidade"] = "N/A"
        else:
            val = int(rec["rastreabilidade"])
            if val not in (0, 1, 2):
                raise ValueError(f"Invalid score {val} for {item_id} rastreabilidade")
            if not rec["just_rastreabilidade"].strip():
                raise ValueError(f"Empty justification for {item_id} rastreabilidade")
            scores_by_cond_crit[(cond, "rastreabilidade")].append(val)

        unblinded.append(rec)

    summary_path = Path(out_summary_csv)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["condition", "criterio", "media", "n"],
            lineterminator="\n",
        )
        writer.writeheader()
        for (cond, crit), vals in sorted(scores_by_cond_crit.items()):
            writer.writerow(
                {
                    "condition": cond,
                    "criterio": crit,
                    "media": f"{(sum(vals) / len(vals)):.4f}",
                    "n": len(vals),
                }
            )

    return unblinded


RAG_EXPERIMENT_FIELDNAMES = [
    "query_id",
    "query",
    "condition",
    "context_doc_ids",
    "prompt",
    "answer",
    "fidelidade",
    "completude",
    "utilidade",
    "rastreabilidade",
    "just_fidelidade",
    "just_completude",
    "just_utilidade",
    "just_rastreabilidade",
    "fidelity_reference_doc_ids",
    "model_id",
    "model_version",
    "temperature",
    "seed",
    "max_output_tokens",
    "language",
]


def export_rag_experiment_csv(
    unblinded_rows: list[dict[str, Any]],
    responses_jsonl: Path | str,
    out_csv: Path | str,
) -> Path:
    """Export final rag_experiment.csv matching TDS section 2.7 schema."""
    responses_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    ref_docs_by_query: dict[str, str] = {}
    with Path(responses_jsonl).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            qid = rec["query_id"]
            cond = rec["condition"]
            responses_by_key[(qid, cond)] = rec
            if cond == "B":
                ref_docs_by_query[qid] = ";".join(rec.get("context_doc_ids", []))

    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sorted_rows = sorted(unblinded_rows, key=lambda r: (r["query_id"], r["condition"]))

    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=RAG_EXPERIMENT_FIELDNAMES,
            lineterminator="\n",
        )
        writer.writeheader()
        for row in sorted_rows:
            qid = row["query_id"]
            cond = row["condition"]
            resp = responses_by_key[(qid, cond)]
            params = resp.get("params", {})
            ctx_ids = ";".join(resp.get("context_doc_ids", [])) if cond == "B" else ""
            just_rast = (
                "Sem contexto externo (condição A)"
                if cond == "A"
                else row["just_rastreabilidade"]
            )
            writer.writerow(
                {
                    "query_id": qid,
                    "query": row["question"],
                    "condition": cond,
                    "context_doc_ids": ctx_ids,
                    "prompt": resp["prompt"],
                    "answer": resp["answer"],
                    "fidelidade": int(row["fidelidade"]),
                    "completude": int(row["completude"]),
                    "utilidade": int(row["utilidade"]),
                    "rastreabilidade": "N/A" if cond == "A" else row["rastreabilidade"],
                    "just_fidelidade": row["just_fidelidade"],
                    "just_completude": row["just_completude"],
                    "just_utilidade": row["just_utilidade"],
                    "just_rastreabilidade": just_rast,
                    "fidelity_reference_doc_ids": ref_docs_by_query.get(qid, ""),
                    "model_id": resp["model_id"],
                    "model_version": resp["model_version"],
                    "temperature": params.get("temperature", ""),
                    "seed": params.get("seed", ""),
                    "max_output_tokens": params.get("max_output_tokens", ""),
                    "language": params.get("language", ""),
                }
            )
    return out_path

