"""
Copia para results/evidencias/ os arquivos pequenos de artifacts/ que o relatório cita.

Os 7 CSVs obrigatórios continuam na raiz de results/. Arquivos grandes e regeneráveis
(embeddings .npy, runs .trec, scores .parquet, responses.jsonl) ficam em artifacts/.

Uso, na raiz do repositório, na máquina que tem a pasta artifacts/ completa:
    python scripts/export_evidencias.py

Rode de novo sempre que um artefato mudar: os arquivos de results/evidencias/ são sobrescritos.
"""
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ART = ROOT / "artifacts"
OUT = ROOT / "results" / "evidencias"

# padrão glob (relativo a artifacts/) -> subpasta de destino em results/evidencias/
PADROES = {
    "data_stats.json": "",
    "rag_queries.json": "",
    "index_stats_*.json": "indice",
    "lexical_candidates_*.csv": "indice",
    "dev/*.csv": "dev",
    "embeddings/*/manifest.json": "modelos/bi_encoder",
    "embeddings/*/doc_ntokens.csv": "modelos/bi_encoder",
    "rerank_manifest*.json": "modelos/cross_encoder",
    "typo/*.csv": "typo",
    "typo/*.json": "typo",
    "rag/contexts.json": "rag",
    "rag/rag_retrieval_metrics.csv": "rag",
    "rag/human_eval_summary.csv": "rag",
    "rag/agreement.json": "rag",
    "rag/avaliacao_*.csv": "rag",
    "rag/consenso_divergencias.csv": "rag",
    "analysis/*.csv": "analise",
    "freeze_log.json": "",
    "test_summary.json": "",
    "semantic_timing.json": "modelos/bi_encoder",
}
PROIBIDOS = {".npy", ".trec", ".parquet", ".pt", ".bin", ".safetensors"}
LIMITE_BYTES = 5 * 1024 * 1024

if not ART.exists():
    sys.exit(f"Pasta {ART} não encontrada: rode este script na máquina que gerou os artefatos.")

copiados, ausentes = [], []
for padrao, sub in PADROES.items():
    achados = sorted(ART.glob(padrao))
    if not achados:
        ausentes.append(padrao)
    for src in achados:
        if "_corrigido" in src.stem:
            print(f"  ignorado (arquivo temporário de verificação): {src.relative_to(ROOT)}")
            continue
        if src.suffix in PROIBIDOS or src.stat().st_size > LIMITE_BYTES:
            print(f"  ignorado (grande ou binário): {src.relative_to(ROOT)}")
            continue
        dst = OUT / sub / src.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copiados.append(dst.relative_to(ROOT))

print(f"{len(copiados)} arquivo(s) copiado(s) para {OUT.relative_to(ROOT)}:")
for c in copiados:
    print(f"  {c}")
if ausentes:
    print("\nPadrões sem arquivo correspondente (confira se o nome mudou na implementação):")
    for a in ausentes:
        print(f"  artifacts/{a}")
