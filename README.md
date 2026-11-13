# Trabalho Prático 2 — Busca Lexical, Semântica, Híbrida e RAG (SCC0282, RI 2026)

Sistema de Recuperação de Informação sobre o **NFCorpus (BEIR)** que combina:
- busca lexical (BM25 sobre índice invertido próprio);
- busca semântica (bi-encoder);
- fusão híbrida (RRF);
- reranking (cross-encoder);
- RAG com LLM.

A avaliação usa P@10, R@10, MAP@10 e NDCG@10, e o experimento de RAG compara respostas com e sem contexto.

**Disciplina:** SCC0282 — Recuperação de Informação, ICMC-USP, 2º semestre de 2026 · **Prazo:** 19/11/2026, 23:59

| Integrante | Nº USP | Papel |
|---|---|---|
| Rafael Mendonça Duarte | 16817608 | Especificação (requisitos, arquitetura, TDS) e busca lexical (reaproveitada do T1) |
| Joao Pedro Correa Caetano | 16987067 | Implementação dos demais componentes e experimentos |

> **Status:** em desenvolvimento. A especificação está concluída. O índice invertido e o BM25 estão implementados e testados. Os demais componentes estão por implementar (ver [Status](#status-dos-componentes)).

---

## Como analisar este repositório

Leia nesta ordem. Cada documento parte do anterior e não repete o conteúdo dele.

| # | Arquivo | O que é | Quando consultar |
|---|---|---|---|
| 0 | `docs/enunciado/Trabalho_Pratico_2_RI_2026.pdf` | Enunciado oficial | Fonte de verdade em qualquer dúvida |
| 1 | `docs/t2_requisitos.md` | **O quê** fazer: os 15 requisitos (RF-1…RF-15), restrições críticas (C1…C22), métricas, schemas mínimos, ambiguidades | Para saber se algo é obrigatório |
| 2 | `docs/t2_decisoes.md` | Log de decisões (D1…D10): modelos, LLM, regras de escopo | Antes de escolher qualquer modelo ou parâmetro |
| 3 | `docs/t2_arquitetura.md` | Componentes C-0…C-14, interfaces, fluxo (diagramas Mermaid), armazenamento, riscos | Para entender como as peças se conectam |
| 4 | `docs/t2_tds.md` | **Como** implementar: contratos de I/O, sub-tarefas, critérios de conclusão verificáveis, schemas exatos dos CSVs, checklist final | Durante a implementação; é o documento de trabalho |
| 5 | `docs/t2_plano_execucao.md` | Ordem das sprints, checklists e **testes-contrato** de cada etapa | Para saber o que fazer a seguir |
| 6 | `docs/t2_busca_lexical_bm25_contexto.md` | Como o BM25 do T1 foi adaptado (índice invertido refeito do zero) | Ao mexer em `src/inverted_index.py` ou `src/bm25.py` |
| — | `AGENTS.md` | Regras para assistentes de IA (e humanos) que trabalham no código | **Obrigatório** antes de usar qualquer LLM no repositório |

**Precedência em caso de conflito:** enunciado > requisitos > decisões > arquitetura > TDS > plano. Um conflito encontrado é registrado em `docs/t2_decisoes.md`, e não resolvido em silêncio no código.

**Convenção de IDs:**
- `RF-N`: requisito N do enunciado;
- `Cn`: restrição crítica;
- `C-N`: componente da arquitetura;
- `TDS-N`: especificação do componente C-N;
- `Dn`: decisão;
- `TDS-DEC-n`: decisão de implementação;
- `Sn`: sprint do plano.

> Os documentos de `docs/t2_*.md`, `AGENTS.md`, `CLAUDE.md` e `.claude/` apoiam o desenvolvimento com IA e serão **removidos antes da entrega** (decisão D10).

---

## Estrutura

```
.
├── README.md
├── AGENTS.md                         # regras para agentes de IA
├── CLAUDE.md                         # aponta para AGENTS.md (Claude Code)
├── .claude/settings.json             # Claude Code sem atribuição em commits/PRs
├── .githooks/                        # hook commit-msg que bloqueia atribuição de IA
├── .github/workflows/                # mesma verificação no GitHub (Actions)
├── docs/                             # especificação (ver tabela acima)
├── src/                              # um módulo por componente (TDS §0.4)
│   ├── inverted_index.py             # C-2 — implementado
│   └── bm25.py                       # C-3 — implementado
├── tests/                            # pytest: testes-contrato (ver plano)
│   └── reference/bm25_t1.py          # BM25 original do T1, só como oráculo de regressão
├── results/                          # os 7 CSVs entregues (RF-15)
└── requirements.txt
```

Criados durante a implementação: `scripts/`, `config/config.yaml`, `artifacts/` (não versionado), `data/` (não versionado).

## Status dos componentes

| Componente | Módulo | Status |
|---|---|---|
| C-0 Carregador NFCorpus | `src/data_loader.py` | **Feito**: 10 testes em `tests/test_data_loader.py` (6 unitários + 4 `@pytest.mark.data`) |
| C-1 Pré-processamento | `src/preprocessing.py` | A fazer (S2) |
| C-2 Índice invertido | `src/inverted_index.py` | **Feito**: 28 testes em `tests/test_bm25.py` |
| C-3 BM25 | `src/bm25.py` | **Feito**; geração das runs a fazer (S2) |
| C-4 Correção ortográfica | `src/spell.py` | A fazer (S7) |
| C-5 Busca semântica | `src/dense.py` | A fazer (S4) |
| C-6 Avaliação | `src/evaluation.py` | A fazer (S3) |
| C-7 RRF | `src/fusion.py` | A fazer (S5) |
| C-8 Reranking | `src/rerank.py` | A fazer (S6) |
| C-9/C-10 Contexto + LLM | `src/rag_context.py`, `src/llm.py` | A fazer (S9) |
| C-11 Avaliação humana | `src/human_eval.py` | A fazer (S9) |
| C-12 Análises | `src/analysis.py` | A fazer (S10) |
| C-13 CSVs | `src/results_writer.py` | A fazer (S8) |
| C-14 Orquestrador | `src/orchestrator.py` | A fazer (S5) |

## Configuração técnica (detalhes em `docs/t2_decisoes.md`)

| Item | Escolha |
|---|---|
| Tokenização | minúsculas + regex `[a-z0-9]+`, sem pontuação (D8) |
| Busca lexical | BM25 (k1 = 1,2; b = 0,75), índice invertido próprio (D1) |
| Bi-encoder | `sentence-transformers/multi-qa-MiniLM-L6-cos-v1`, cosseno, busca exaustiva, texto cru (D4) |
| Fusão | RRF, k ∈ {10, 60, 100} escolhido no dev |
| Cross-encoder | `cross-encoder/ms-marco-MiniLM-L2-v2`, top-50 do híbrido (D5) |
| LLM (RAG) | `gemini-3.1-flash-lite`, Gemini API, free tier (D2) |
| Hardware de referência | CPU Intel i5 (4 threads), 20 GB RAM, sem GPU |

## Instalação e execução

```bash
git clone https://github.com/rduarte12/information_retrieval_rag.git
cd information_retrieval_rag
git config core.hooksPath .githooks        # OBRIGATÓRIO: ativa o hook anti-atribuição de IA

python -m venv .venv
source .venv/bin/activate                  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pytest                                     # 28 testes devem passar

# Dados: baixar e extrair em data/nfcorpus/
#   https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/nfcorpus.zip
cp .env.example .env                       # definir GEMINI_API_KEY (nunca versionar)
```

Execução prevista (a criar nas sprints S5–S9):

```bash
python scripts/run_dev.py     # ajustes e validações no dev; congela o config
python scripts/run_test.py    # execução única no test → results/*.csv
python scripts/run_typo.py    # experimento de erros ortográficos (RF-4)
python scripts/run_rag.py     # RAG A/B (RF-11/12); depois a avaliação humana (RF-13)
```

Versão do Python e das bibliotecas: **Python 3.14.10**; versões exatas das bibliotecas fixadas em [`requirements.txt`](requirements.txt).

## Regras do projeto (resumo)

- O **test** é usado uma única vez, com a configuração congelada. Todo ajuste é feito no **dev**.
- O índice invertido é implementado do zero; a busca lexical nunca percorre todos os documentos.
- Os resultados entregues são os 7 CSVs de `results/` (schemas no TDS §2), não saídas de notebook.
- Ranking lexical com menos de 100 documentos **não é completado**, e isso é registrado e informado no relatório (D7).
- Não versionar: dataset, embeddings, modelos, índices serializados, `.env`.

## Autoria e uso de IA

Assistentes de IA são usados como apoio; o uso é declarado no relatório. **Commits e PRs têm apenas os integrantes como autores.** Três travas garantem isso (D9):

1. **Configuração da ferramenta.** No Claude Code, `.claude/settings.json` (já no repositório) contém:
   ```json
   { "attribution": { "commit": "", "pr": "" } }
   ```
   Em outras ferramentas (Copilot, Codex, Cursor etc.), desative a opção equivalente de coautoria ou assinatura.
2. **Hook local** `.githooks/commit-msg`. Recusa commits com trailers `Co-authored-by` de IA, "Generated with …", `Claude-Session:`, 🤖 ou autor bot. É versionado, mas o Git só o executa depois de `git config core.hooksPath .githooks`, uma vez por clone.
3. **GitHub Action** `.github/workflows/no-ai-attribution.yml`. Repete a verificação em todo push e PR; pega o que escapar do hook (ex.: `--no-verify`).
