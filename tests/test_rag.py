from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.fusion import read_trec_run
from src.human_eval import generate_blind_sheet, unblind_and_summarize
from src.llm import GeminiClient, build_prompt
from src.rag_context import extract_rag_contexts

REPO_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = REPO_ROOT / "artifacts"


def test_prompt_templates_snapshot():
    ctx = [
        {"rank": 1, "doc_id": "MED-101", "title": "Omega 3", "text": "Omega 3 reduces inflammation."},
        {"rank": 2, "doc_id": "MED-202", "title": "Diet", "text": "Fish oil study."},
    ]
    prompt_a = build_prompt("A", "Does omega-3 help?", context=None, language="pt-BR")
    prompt_b = build_prompt("B", "Does omega-3 help?", context=ctx, language="pt-BR")

    assert "MED-101" not in prompt_a
    assert "MED-202" not in prompt_a
    assert "Does omega-3 help?" in prompt_a

    assert "[MED-101] Omega 3" in prompt_b
    assert "[MED-202] Diet" in prompt_b
    assert "somente com base no contexto" in prompt_b.lower()
    assert "colchetes" in prompt_b.lower()
    assert "evidência suficiente" in prompt_b.lower()
    assert "Does omega-3 help?" in prompt_b


def test_llm_client_mock_and_retry(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    calls = {"n": 0}

    class FakeModels:
        def generate_content(self, model, contents, config):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("429 RESOURCE_EXHAUSTED")
            return SimpleNamespace(
                text="Resposta simulada [MED-101].",
                model_version="gemini-3.1-flash-lite-001",
                candidates=[SimpleNamespace(finish_reason="STOP")],
                usage_metadata=SimpleNamespace(
                    prompt_token_count=50,
                    candidates_token_count=12,
                    thoughts_token_count=0,
                ),
            )

    monkeypatch.setattr("time.sleep", lambda _: None)
    client = GeminiClient(
        model_id="gemini-3.1-flash-lite",
        temperature=0.0,
        seed=42,
        max_output_tokens=1024,
        thinking=None,
        language="pt-BR",
        raw_client=SimpleNamespace(models=FakeModels()),
    )
    record = client.generate(query_id="Q1", condition="B", prompt="Prompt B")
    assert calls["n"] == 2
    assert record["answer"] == "Resposta simulada [MED-101]."
    assert record["finish_reason"] == "STOP"
    assert record["model_version"] == "gemini-3.1-flash-lite-001"


def test_blind_sheet_hides_condition_and_unblind_forces_na(tmp_path: Path):
    responses = [
        {"query_id": "Q1", "condition": "A", "question": "Q1?", "answer": "Ans A"},
        {"query_id": "Q1", "condition": "B", "question": "Q1?", "answer": "Ans B [MED-1]"},
    ]
    contexts = {
        "Q1": [{"rank": 1, "doc_id": "MED-1", "title": "T1", "text": "Text 1"}],
    }
    sheet_path = tmp_path / "blind_sheet.csv"
    key_path = tmp_path / "blind_key.json"
    generate_blind_sheet(responses, contexts, sheet_path, key_path, seed=42)

    with sheet_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        assert "condition" not in (reader.fieldnames or [])
        rows = list(reader)
    assert len(rows) == 2
    for r in rows:
        assert r["condition"] if "condition" in r else True
        assert all(v not in ("A", "B") for v in r.values())

    # Simulate filled scores.csv
    scores_path = tmp_path / "scores.csv"
    with scores_path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = list(rows[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            r["fidelidade"] = "2"
            r["just_fidelidade"] = "Ok"
            r["completude"] = "1"
            r["just_completude"] = "Parcial"
            r["utilidade"] = "2"
            r["just_utilidade"] = "Util"
            r["rastreabilidade"] = "2"  # Will be forced to N/A for condition A
            r["just_rastreabilidade"] = "Citou"
            writer.writerow(r)

    summary_path = tmp_path / "human_eval_summary.csv"
    unblinded = unblind_and_summarize(scores_path, key_path, summary_path)

    cond_a = [r for r in unblinded if r["condition"] == "A"][0]
    cond_b = [r for r in unblinded if r["condition"] == "B"][0]
    assert cond_a["rastreabilidade"] == "N/A"
    assert cond_b["rastreabilidade"] == "2"

    with summary_path.open("r", encoding="utf-8", newline="") as f:
        sum_rows = list(csv.DictReader(f))
    assert not any(r["condition"] == "A" and r["criterio"] == "rastreabilidade" for r in sum_rows)


@pytest.mark.data
def test_contexts_match_reranking_top5():
    contexts_path = ARTIFACTS_DIR / "rag" / "contexts.json"
    metrics_path = ARTIFACTS_DIR / "rag" / "rag_retrieval_metrics.csv"
    rag_queries_path = ARTIFACTS_DIR / "rag_queries.json"
    rerank_trec = ARTIFACTS_DIR / "runs" / "reranking_test.trec"

    assert contexts_path.exists()
    assert metrics_path.exists()

    rag_qids = json.loads(rag_queries_path.read_text(encoding="utf-8"))
    if isinstance(rag_qids, dict):
        rag_qids = rag_qids.get("query_ids", list(rag_qids.keys()))
    contexts = json.loads(contexts_path.read_text(encoding="utf-8"))
    rerank_run = read_trec_run(rerank_trec)

    assert set(contexts.keys()) == set(rag_qids)
    assert len(contexts) == 10
    for qid in rag_qids:
        items = contexts[qid]
        assert len(items) == 5
        expected_top5 = [doc_id for doc_id, _ in rerank_run[qid][:5]]
        assert [item["doc_id"] for item in items] == expected_top5
