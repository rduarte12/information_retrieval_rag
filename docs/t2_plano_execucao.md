# Plano de Execução — T2 (SCC0282, RI 2026)

> Guia de implementação em sprints. Cada sprint implementa um ou mais componentes do `docs/t2_tds.md` e só termina quando os **testes-contrato** dela passam.
> Responsável pela implementação: Joao Pedro Correa Caetano. Especificação: Rafael Mendonça Duarte.
> Prazo de entrega: **19/11/2026, 23:59**. As sprints não têm data própria: seguem a ordem abaixo, respeitando as dependências.

## Como usar

1. Siga a ordem das sprints: as dependências são reais. A busca semântica pode andar em paralelo, porque só depende da S1.
2. Em cada sprint:
   1. escreva **primeiro** os testes listados em "Contrato";
   2. implemente até eles passarem;
   3. marque o checklist.
3. Ao fim de cada sessão (humana ou de IA), atualize a tabela "Status dos componentes" do `README.md`.
4. Qualquer decisão nova vai para `docs/t2_decisoes.md`.

## Testes como contrato

O **critério de conclusão** de cada componente no TDS vira teste automatizado em `tests/`. O teste é escrito **antes** do código, a partir do TDS, e define o que "pronto" significa.

| Regra | Por quê |
|---|---|
| Um arquivo de teste por componente: `tests/test_<módulo>.py` | Rastreabilidade C-N ↔ teste |
| Testes unitários usam dados sintéticos pequenos; o NFCorpus real só entra em testes marcados `@pytest.mark.data` | Rodam em segundos e sem download |
| **Não é permitido** enfraquecer, apagar ou marcar `skip` num teste para fazê-lo passar sem aprovação humana | Senão o contrato deixa de valer (regra também no `AGENTS.md`) |
| Valores esperados vêm do TDS ou de cálculo à mão, nunca da saída do próprio código | Evita teste que só confirma o bug |
| `pytest` inteiro precisa passar antes de qualquer commit | O hook não verifica isso; é disciplina |

Os testes que exigem dados reais rodam com `pytest -m data`, depois de baixar o NFCorpus em `data/nfcorpus/`.

**Já existe:** `tests/test_bm25.py`, com 28 testes que passam, cobrindo C-2 e C-3:
- fórmula comparada com um oráculo manual;
- saturação de tf;
- casos de borda;
- termo repetido;
- desempate por `doc_id`;
- grade sem reconstruir o índice;
- não-varredura;
- estatísticas;
- ausência de bibliotecas proibidas;
- regressão contra o BM25 do T1 (`tests/reference/bm25_t1.py`).

---

## S0 — Ambiente

**Depende de:** nada.

- [ ] `python -m venv .venv`; instalar `requirements.txt`; depois `pip freeze` e fixar as versões com `==`.
- [ ] `git config core.hooksPath .githooks` (ativa o hook; ver README).
- [ ] Configurar a ferramenta de IA sem atribuição (Claude Code: `.claude/settings.json` já está no repo).
- [ ] Baixar e extrair o NFCorpus em `data/nfcorpus/` (URL no README).
- [ ] Criar `.env` a partir do `.env.example` com `GEMINI_API_KEY`.
- [ ] `pytest` → 28 testes passando.

**Pronto quando:** `pytest` passa num ambiente limpo, e um commit de teste com `Co-Authored-By: Claude` é recusado pelo hook (desfazer depois com `git reset`).

## S1 — Dados (TDS-0)

**Depende de:** S0. **Arquivos:** `src/data_loader.py`, `tests/test_data_loader.py`.

**Contrato:**
- [ ] 3.633 documentos, 3.237 consultas, 324 consultas no dev e 323 no test (`@pytest.mark.data`);
- [ ] todo `query_id` dos qrels existe em `queries`; IDs são `str`;
- [ ] `doc_text` = `title + " " + text` (unitário);
- [ ] o sorteio do Req. 12 reproduz exatamente as 10 consultas oficiais, com 42 elegíveis (`@pytest.mark.data`).

**Saídas:** `artifacts/data_stats.json`, `artifacts/rag_queries.json`.

## S2 — Pré-processamento + busca lexical (TDS-1, TDS-2, TDS-3)

**Depende de:** S1. **Arquivos:** `src/preprocessing.py`, `tests/test_preprocessing.py`.

**Contrato:**
- [ ] `tokenize("Cancer, diet. (Omega-3)!") == ["cancer","diet","omega","3"]` e os demais exemplos do TDS-1 (**sem pontuação**, D8);
- [ ] nenhum termo do índice construído sobre o NFCorpus fere `re.fullmatch(r"[a-z0-9]+", t)` (`@pytest.mark.data`);
- [ ] `preprocess` aplica stopwords e stemming conforme o config; `surface_tokens` sai do mesmo `tokenize`;
- [ ] `config_hash` muda quando o config muda;
- [ ] `tests/test_bm25.py` continua passando, mais a não-varredura sobre o NFCorpus (`@pytest.mark.data`).

**Checklist:**
- [ ] Escolher a configuração de pré-processamento no **dev** → `artifacts/dev/preprocessing_dev.csv`.
- [ ] Gerar `lexical_{dev,test}.trec` e `lexical_candidates_{split}.csv`, com aviso no console para consultas com menos de 100 candidatos (D7). **Não completar.**
- [ ] Gerar `artifacts/index_stats_*.json`, com vocabulário, postings, tempo de construção e tempo médio por consulta (RF-2).

## S3 — Avaliação (TDS-6)

**Depende de:** S1. **Arquivos:** `src/evaluation.py`, `tests/test_evaluation.py`, `tests/test_eval_vs_ir_measures.py`.

**Contrato:**
- [ ] P10, R10, AP10 e NDCG10 batem com rankings calculados à mão, incluindo run vazia, R = 0 e menos de 10 recuperados (denominador 10);
- [ ] `MAP@10 == mean(AP10)`;
- [ ] diferença ≤ 1e-9 em relação ao `ir_measures` em todas as consultas, com as ausentes da run contando 0;
- [ ] R5, P5 e Hit5 calculados à mão.

## S4 — Busca semântica (TDS-5) (pode correr em paralelo com S2/S3)

**Depende de:** S1. **Arquivos:** `src/dense.py`, `tests/test_dense.py`.

**Contrato:**
- [ ] `doc_emb.npy` com shape (3633, 384) e normas iguais a 1 ± 1e-5 (`@pytest.mark.data`);
- [ ] um documento usado como consulta tem a si mesmo como top-1;
- [ ] desempate por `doc_id`, estável na fronteira do top-k (unitário com matriz sintética);
- [ ] `manifest.json` com os 9 itens do RF-5.

**Checklist:**
- [ ] Texto cru (sem stopwords nem stemming); `normalize_embeddings=True`; `torch.set_num_threads(4)`.
- [ ] Runs `semantic_{dev,test}.trec`; NDCG@10 do dev em `validation_dev.csv` **antes** de qualquer execução no test.

## S5 — Orquestrador + RRF (TDS-14, TDS-7)

**Depende de:** S2, S3, S4. **Arquivos:** `src/orchestrator.py`, `src/fusion.py`, `tests/test_split_guard.py`, `tests/test_fusion.py`, `config/config.yaml`.

**Contrato:**
- [ ] chamar um estágio de ajuste com `split="test"` levanta `SplitGuardError`;
- [ ] o test exige `frozen: true`; um segundo run do test com outro `config_hash` é bloqueado;
- [ ] RRF com 2 runs pequenas bate com o cálculo à mão (rank começa em 1; documento ausente não contribui).

**Checklist:**
- [ ] Grade k ∈ {10, 60, 100} no dev → `rrf_k_dev.csv`; fixar o k no config.
- [ ] Runs `hybrid_{split}.trec`.

## S6 — Reranking (TDS-8)

**Depende de:** S5. **Arquivos:** `src/rerank.py`, `tests/test_rerank.py`.

**Contrato:**
- [ ] o top-50 reranqueado é uma permutação do top-50 do híbrido;
- [ ] as posições 51–100 são idênticas às do híbrido;
- [ ] a coluna `score_source` é coerente com o rank;
- [ ] `max_length=512` registrado e % de pares truncados calculada.

**Checklist:**
- [ ] Validação no dev (`validation_dev.csv`); tempo médio por etapa registrado.
- [ ] Ainda **não** congelar o config: o congelamento acontece no fim da S7 (passo 5 da AD §1.2).

## S7 — Correção ortográfica (TDS-4)

**Depende de:** S2, S3 (e S4 para a extensão D6). **Arquivos:** `src/spell.py`, `tests/test_spell.py`.

**Contrato:**
- [ ] `kgrams("casa",2) == {"$c","ca","as","sa","a$"}`;
- [ ] Jaccard em conjuntos fixos;
- [ ] OSA: `("ab","ba")=1`, `("abc","abd")=1`, `("abc","abcd")=1`, `("","abc")=3`;
- [ ] um erro de cada tipo é corrigido; um termo conhecido nunca é alterado;
- [ ] o gerador de erros é determinístico, e todo termo com erro fica fora do dicionário.

**Checklist:**
- [ ] Grade no dev → `spell_grid_dev.csv`; fixar os parâmetros; **congelar o config**.
- [ ] 10 consultas do test × 3 condições, no BM25 e no bi-encoder (D6) → `typo_experiment.csv`, `robustez_lex_sem.csv`.

## S8 — Execução no test + CSVs (TDS-13)

**Depende de:** S5–S7 com o config congelado. **Arquivos:** `src/results_writer.py`, `scripts/run_test.py`, `tests/test_results_schema.py`.

**Contrato** (schemas do TDS §2):
- [ ] os 7 CSVs existem com as colunas mínimas;
- [ ] ranks contíguos e ≤ 100 por consulta;
- [ ] semântico com 32.300 linhas;
- [ ] `metrics_per_query.csv` com 1.292 linhas e `mean(AP10)` igual ao MAP@10;
- [ ] `relevance` coerente com `qrels/test.tsv`.

**Checklist:**
- [ ] Rodar o test **uma vez** (`test_run.lock`).

## S9 — RAG + avaliação humana (TDS-9, 10, 11)

**Depende de:** S8. **Arquivos:** `src/rag_context.py`, `src/llm.py`, `src/human_eval.py`, `tests/test_rag.py`.

**Contrato:**
- [ ] `contexts.json` com as 10 consultas oficiais × 5 `doc_id`, iguais ao top-5 de `reranking_test.trec`;
- [ ] o prompt B contém as 3 instruções e o prompt A não contém nenhum `doc_id` (snapshot);
- [ ] a planilha cega não contém a condição;
- [ ] Rastreabilidade = `N/A` em A.

**Checklist:**
- [ ] Fechar as pendências de D2 (idioma, `max_output_tokens`, thinking, teste de seed).
- [ ] Gerar `responses.jsonl` **uma vez**; registrar o hash.
- [ ] Avaliação humana às cegas → `scores.csv` → `rag_experiment.csv`.

## S10 — Análises (TDS-12)

**Depende de:** S8, S9. **Arquivos:** `src/analysis.py`.

- [ ] Gerar `rf7_cases.csv`, `rf7_top5.csv`, `rf10_cases.csv`, `rf10_moves.csv` e `rf14_trace.csv` pelas regras determinísticas.
- [ ] Toda afirmação numérica do relatório aponta para uma célula desses CSVs.

## S11 — Relatório, revisão e entrega

- [ ] Relatório PDF (≤ 20 páginas sem as referências) com as seções a–g. Transcrever as justificativas de D1–D10 e TDS-DEC-n.
- [ ] Checklist completo do TDS §3.
- [ ] README final: instalação, versões, hardware, ordem de execução, status.
- [ ] **Limpeza (D10):** remover `docs/t2_*.md`, `AGENTS.md`, `CLAUDE.md` e `.claude/` (lista no TDS §5); revisar o README para não citar arquivos removidos.
- [ ] Zip para o Moodle sem dataset, embeddings, modelos ou `.env`.
