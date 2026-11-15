from __future__ import annotations

import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from google import genai
from google.genai import types

from src.data_loader import load_queries
from src.orchestrator import load_config, record_stage_result


def build_prompt(
    condition: str,
    question: str,
    context: Sequence[Mapping[str, Any]] | None = None,
    language: str = "pt-BR",
) -> str:
    """Build exact prompt template for Condition A (no RAG) or B (with Top-5 RAG)."""
    if condition == "A":
        return (
            f"Instrução: Responda à pergunta de forma clara e objetiva ({language}).\n\n"
            f"Pergunta: {question}"
        )

    if condition == "B":
        if not context:
            raise ValueError("Condition 'B' requires non-empty context.")
        blocks = [
            f"[{item['doc_id']}] {item.get('title', '').strip()}\n{item.get('text', '').strip()}".strip()
            for item in context
        ]
        context_str = "\n\n".join(blocks)
        return (
            f"Instruções ({language}):\n"
            "1. Responda somente com base no contexto fornecido abaixo.\n"
            "2. Cite os identificadores dos documentos utilizados entre colchetes (ex.: [MED-123]).\n"
            "3. Declare explicitamente quando o contexto não tiver evidência suficiente para responder.\n\n"
            f"Contexto:\n{context_str}\n\n"
            f"Pergunta: {question}"
        )

    raise ValueError(f"Unsupported condition: {condition!r}")


class GeminiClient:
    """Wrapper over google-genai SDK with retry/backoff and structured logging."""

    def __init__(
        self,
        model_id: str = "gemini-3.1-flash-lite",
        temperature: float = 0.0,
        seed: int | None = 42,
        max_output_tokens: int = 1024,
        thinking: str | None = None,
        language: str = "pt-BR",
        raw_client: Any | None = None,
    ) -> None:
        self.model_id = model_id
        self.temperature = temperature
        self.seed = seed
        self.max_output_tokens = max_output_tokens
        self.thinking = thinking
        self.language = language
        if raw_client is not None:
            self.client = raw_client
        else:
            api_key = os.environ.get("GEMINI_API_KEY")
            if not api_key:
                raise RuntimeError("GEMINI_API_KEY environment variable is not set.")
            self.client = genai.Client(api_key=api_key)

    def _build_config(self) -> types.GenerateContentConfig:
        kwargs: dict[str, Any] = {
            "temperature": self.temperature,
            "max_output_tokens": self.max_output_tokens,
        }
        if self.seed is not None:
            kwargs["seed"] = self.seed
        return types.GenerateContentConfig(**kwargs)

    def generate(
        self,
        query_id: str,
        condition: str,
        prompt: str,
        context_doc_ids: Sequence[str] = (),
        max_retries: int = 5,
    ) -> dict[str, Any]:
        """Generate a single response with exponential backoff on 429 errors."""
        cfg = self._build_config()
        last_err: Exception | None = None
        response = None
        for attempt in range(max_retries):
            try:
                response = self.client.models.generate_content(
                    model=self.model_id,
                    contents=prompt,
                    config=cfg,
                )
                break
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                msg = str(exc).lower()
                if "429" in msg or "resource_exhausted" in msg or "rate" in msg:
                    time.sleep(2**attempt)
                    continue
                raise
        if response is None:
            raise RuntimeError(f"Failed after {max_retries} retries: {last_err}")

        finish_reason = "UNKNOWN"
        if getattr(response, "candidates", None):
            finish_reason = str(response.candidates[0].finish_reason)

        usage_meta = getattr(response, "usage_metadata", None)
        usage = {
            "input_tokens": getattr(usage_meta, "prompt_token_count", None),
            "output_tokens": getattr(usage_meta, "candidates_token_count", None),
            "thinking_tokens": getattr(usage_meta, "thoughts_token_count", None),
        }
        return {
            "query_id": query_id,
            "condition": condition,
            "prompt": prompt,
            "context_doc_ids": list(context_doc_ids) if condition == "B" else [],
            "answer": getattr(response, "text", "") or "",
            "finish_reason": finish_reason,
            "model_id": self.model_id,
            "model_version": getattr(response, "model_version", self.model_id),
            "params": {
                "temperature": self.temperature,
                "seed": self.seed,
                "max_output_tokens": self.max_output_tokens,
                "thinking": self.thinking,
                "language": self.language,
            },
            "usage": usage,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }


def run_rag_generation(
    config_path: Path | str = Path("config/config.yaml"),
    artifacts_dir: Path | str = Path("artifacts"),
) -> dict[str, Any]:
    """Generate the 20 RAG responses (10 queries x {A, B}) with incremental caching."""
    art_dir = Path(artifacts_dir)
    config = load_config(config_path)
    llm_cfg = config["llm"]

    contexts = json.loads((art_dir / "rag" / "contexts.json").read_text(encoding="utf-8"))
    queries = load_queries(Path(config["data"]["root"]) / "queries.jsonl")

    out_path = art_dir / "rag" / "responses.jsonl"
    existing: dict[tuple[str, str], dict[str, Any]] = {}
    if out_path.exists():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                existing[(rec["query_id"], rec["condition"])] = rec

    client = GeminiClient(
        model_id=llm_cfg["model_id"],
        temperature=float(llm_cfg["temperature"]),
        seed=llm_cfg.get("seed"),
        max_output_tokens=int(llm_cfg["max_output_tokens"]),
        thinking=llm_cfg.get("thinking"),
        language=llm_cfg["language"],
    )

    records: list[dict[str, Any]] = []
    for qid in sorted(contexts.keys()):
        q_text = queries[qid]
        ctx_items = contexts[qid]
        for cond in ("A", "B"):
            key = (qid, cond)
            if key in existing:
                records.append(existing[key])
                continue
            prompt = build_prompt(
                cond,
                q_text,
                context=ctx_items if cond == "B" else None,
                language=llm_cfg["language"],
            )
            doc_ids = [item["doc_id"] for item in ctx_items] if cond == "B" else []
            rec = client.generate(qid, cond, prompt, context_doc_ids=doc_ids)
            records.append(rec)
            with out_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    sha = hashlib.sha256(out_path.read_bytes()).hexdigest()
    record_stage_result("rag_responses", out_path, config=config, artifacts_dir=art_dir)
    return {"n_responses": len(records), "responses_sha256": sha}


if __name__ == "__main__":
    summary = run_rag_generation()
    print(json.dumps(summary, indent=2))
