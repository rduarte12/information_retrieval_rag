# Technical Design Specification — Trabalho Prático 2 (SCC0282, RI 2026)

> **Base aprovada:**
> - `docs/t2_requisitos.md` (RD): requisitos **RF-N** e restrições **Cn**.
> - `docs/t2_arquitetura.md` (AD): componentes **C-N**.
> - `docs/t2_decisoes.md`: decisões **D1–D5**.
> - `docs/t2_busca_lexical_bm25_contexto.md`: implementação já existente de C-2/C-3.
>
> **Escopo:** **como** implementar cada C-N. Não toma decisões de modelo. Decisões de implementação tomadas aqui ficam marcadas **[TDS-DEC-n]**; pendências herdadas, **[PENDENTE: Dn]**.
> **Data:** 11/11/2026. Prazo de entrega: 19/11/2026, 23:59.

---

## 0. Convenções globais

### 0.1 Tipos e identificadores

| Nome | Definição |
|---|---|
| `doc_id` | `str`, `_id` do `corpus.jsonl` (ex.: `MED-10`). Comparação **lexicográfica** como string. |
| `query_id` | `str`, `_id` do `queries.jsonl` (ex.: `PLAIN-68`). Ordenação lexicográfica (é a que o Req. 12 usa). |
| `Docs` | `dict[doc_id, {"title": str, "text": str}]` |
| `Queries` | `dict[query_id, str]` |
| `Qrels` | `dict[query_id, dict[doc_id, int]]`, grau original do TSV |
| `Run` | `dict[query_id, list[tuple[doc_id, float]]]`, ordem = ranking; rank = índice + 1 |
| `Split` | `Literal["dev", "test"]` |

### 0.2 Regras de ordenação e desempate (valem para todo `Run`)

- **[TDS-DEC-1]** Ordenação por `score` decrescente; empate → `doc_id` crescente (lexicográfico).
  - Exceção do C-8: empate de score do reranker → posição no híbrido crescente.
- O `Run` de uma consulta pode ter **menos** que k itens (ex.: lexical sem candidatos suficientes). Uma lista vazia é válida e **precisa existir como chave**.

### 0.3 Representação do documento

- **[TDS-DEC-2]** `doc_text(d) = (title.strip() + " " + text.strip()).strip()`.
- É a mesma para C-1 (lexical), C-5 (bi-encoder), C-8 (cross-encoder) e C-9 (contexto do RAG), atendendo à restrição C18.
- Documento com `title` vazio fica só com o `text`.

### 0.4 Layout do repositório [TDS-DEC-3]

```
src/
  data_loader.py      # TDS-0
  preprocessing.py    # TDS-1
  inverted_index.py   # TDS-2 (existente)
  bm25.py             # TDS-3 (existente)
  spell.py            # TDS-4
  dense.py            # TDS-5
  evaluation.py       # TDS-6
  fusion.py           # TDS-7
  rerank.py           # TDS-8
  rag_context.py      # TDS-9
  llm.py              # TDS-10
  human_eval.py       # TDS-11
  analysis.py         # TDS-12
  results_writer.py   # TDS-13
  orchestrator.py     # TDS-14
scripts/
  run_dev.py          # passos 1–5 da AD §1.2
  run_test.py         # execução única no test
  run_typo.py         # experimento RF-4
  run_rag.py          # RF-11/12
tests/                # pytest
config/config.yaml    # TDS-14
artifacts/            # intermediários (não entregues, salvo indicação)
results/              # 7 CSVs do RF-15 (entregues)
```

### 0.5 Formatos internos
- **Run em disco:** formato TREC `query_id Q0 doc_id rank score tag`, separado por espaço, um arquivo por (sistema, split): `artifacts/runs/{system}_{split}.trec`. É lido diretamente por `ir_measures`.
- **Tempo:** `time.perf_counter()`; sem incluir carga de modelo; reportado em ms.
- **Aleatoriedade:** toda amostragem usa `random.Random(42)` local (nunca o RNG global).

### 0.6 Arquivo de configuração (`config/config.yaml`)

| Chave | Tipo | Valor / restrição | Fonte |
|---|---|---|---|
| `data.root` | str | caminho do `nfcorpus/` extraído | C-0 |
| `preprocessing.lowercase` | bool | — | RF-1 (dev) |
| `preprocessing.remove_stopwords` | bool | — | RF-1 (dev) |
| `preprocessing.stemmer` | str \| null | ex.: `porter`, `snowball`, null | RF-1 (dev) |
| `bm25.k1`, `bm25.b` | float | padrão 1.2 / 0.75; só mudam via dev | RF-3, D1 |
| `bm25.k` | int | 100 | RF-15 |
| `dense.model_id` | str | `sentence-transformers/multi-qa-MiniLM-L6-cos-v1` | D4 |
| `dense.revision` | str | commit hash do HF (preencher na 1ª execução) | Reprodutibilidade |
| `dense.max_seq_length` | int | 512 | D4 |
| `dense.batch_size` | int | ex.: 16 | — |
| `rrf.k` | int | escolhido no dev ∈ {10, 60, 100} | RF-8 |
| `rrf.depth` | int | 100 | RF-8 |
| `rerank.model_id` | str | `cross-encoder/ms-marco-MiniLM-L2-v2` | D5 |
| `rerank.revision` | str | commit hash | — |
| `rerank.max_length` | int | 512 | D5 |
| `rerank.depth` | int | 50 | RF-9 |
| `spell.k` | int | dev | RF-4 |
| `spell.jaccard_min` | float | dev | RF-4 |
| `spell.max_edit` | int | dev | RF-4 |
| `spell.distance` | str | `osa` [TDS-DEC-8] | RF-4 |
| `llm.model_id` | str | `gemini-3.1-flash-lite` [PENDENTE: D2] | D2 |
| `llm.temperature` | float | 0.0 [PENDENTE: D2] | RF-12 |
| `llm.seed` | int | 42 [PENDENTE: D2, se aceito] | RF-12 |
| `llm.max_output_tokens` | int | [PENDENTE: D2] | RF-11 |
| `llm.thinking` | str \| null | [PENDENTE: D2] | RF-11 |
| `llm.language` | str | [PENDENTE: D2] | RF-12 |
| `runtime.torch_threads` | int | 4 | D3 |
| `frozen` | bool | `true` só após o passo 5 da AD §1.2 | TDS-14 |

---

## 1. Especificação por Componente

### TDS-0 — Carregador do NFCorpus (← C-0)

**Contrato de entrada**

| Campo | Tipo | Formato | Origem | Obrigatório? |
|---|---|---|---|---|
| `root` | `str` | diretório com `corpus.jsonl`, `queries.jsonl`, `qrels/{train,dev,test}.tsv` | download oficial do BEIR (RD §2.5) | Sim |

**Contrato de saída**

| Campo | Tipo | Formato | Destino | Garantias |
|---|---|---|---|---|
| `docs` | `Docs` | §0.1 | C-1, C-5, C-8, C-9 | `len == 3633`; `doc_id` únicos |
| `queries` | `Queries` | §0.1 | todos | `len == 3237` |
| `qrels[split]` | `Qrels` | §0.1 | C-6, C-12 | lido do TSV com cabeçalho; graus `int` |
| `query_ids[split]` | `list[str]` | ordenada lexicograficamente | C-3, C-5, C-6, C-14 | test: `len == 323`; dev: `len == 324` `[EXTERNO: BEIR]` |
| `doc_text(doc_id)` | `str` | §0.3 | C-1, C-5, C-8, C-9 | determinística |

**Sub-tarefas atômicas**
1. Ler `corpus.jsonl` → saída: `docs` com 3.633 entradas (teste `test_load_corpus`).
2. Ler `queries.jsonl` → saída: `queries` com 3.237 entradas (teste `test_load_queries`).
3. Ler `qrels/dev.tsv` e `qrels/test.tsv`, pulando o cabeçalho → saída: `qrels["dev"]`, `qrels["test"]` (teste confere 324 e 323 chaves).
4. Derivar `query_ids[split] = sorted(qrels[split])` → saída: listas ordenadas; teste confere que todo `query_id` ∈ `queries`.
5. Gerar `artifacts/data_stats.json` com contagens por split, nº de julgamentos por grau, nº de consultas com zero documentos de grau > 0 e média de relevantes por consulta → saída: o JSON.
6. Reproduzir o sorteio do Req. 12 (código do enunciado) → saída: `artifacts/rag_queries.json` com `assert len(ids) == 42` e `assert queries_rag == ["PLAIN-68","PLAIN-2450","PLAIN-153","PLAIN-2850","PLAIN-2750","PLAIN-270","PLAIN-2470","PLAIN-2430","PLAIN-238","PLAIN-3342"]`.

**Critério de finalização**
- `pytest tests/test_data_loader.py` passa com todas as asserções de contagem acima.
- `artifacts/rag_queries.json` é idêntico à lista do enunciado, na mesma ordem.
- `artifacts/data_stats.json` existe e registra a distribuição de graus de `qrels/test.tsv`.

**Restrições de implementação**
- "Deverão ser utilizados os arquivos padronizados da coleção (corpus.jsonl, queries.jsonl e qrels)" (Metodologia; restrição C17).
- "As consultas de cada conjunto são aquelas que possuem julgamentos no respectivo arquivo de qrels" (Metodologia).

**Pontos de falha conhecidos**
- O TSV tem cabeçalho (`query-id corpus-id score`): não pule a primeira linha de dados nem leia o cabeçalho como dado.
- Ler os IDs como `int` quebra a ordenação do Req. 12. Mantenha `str`.
- O qrels do Hugging Face fica em outro repositório (`BeIR/nfcorpus-qrels`); usar o zip oficial evita divergência.
- Se `data_stats.json` mostrar consultas do test com 0 documentos de grau > 0, R@10 e AP@10 ficam indefinidos para elas. Ver TDS-6.

---

### TDS-1 — Pré-processador lexical (← C-1)

**Contrato de entrada**

| Campo | Tipo | Formato | Origem | Obrigatório? |
|---|---|---|---|---|
| `text` | `str` | documento (`doc_text`) ou consulta | C-0 | Sim |
| `cfg` | `dict` | `preprocessing.*` do config | TDS-14 | Sim |

**Contrato de saída**

| Campo | Tipo | Formato | Destino | Garantias |
|---|---|---|---|---|
| `tokens` | `list[str]` | tokens finais (após stopwords e stemming conforme `cfg`) | C-2, C-3 | determinístico; mesma função para documentos e consultas |
| `surface_tokens` | `list[str]` | tokens após tokenização + lowercase, **antes** de stopwords e stemming | C-4 (dicionário) | mesma tokenização dos `tokens` |
| `cfg_hash` | `str` | SHA-256 de `json.dumps(cfg, sort_keys=True)` + versões da biblioteca | C-2, C-4, C-14 | muda se e só se a configuração mudar |

**Sub-tarefas atômicas**
1. Implementar `tokenize(text) -> list[str]` **[TDS-DEC-13]**: lowercase + extração de tokens alfanuméricos por regex `[a-z0-9]+`. **Toda pontuação é descartada**: vírgulas, pontos, dois-pontos, ponto e vírgula, parênteses, colchetes, aspas, apóstrofos, hífens, barras etc. Nenhum token pode conter ou ser só pontuação. → saída: testes com strings fixas:
   - `tokenize("Cancer, diet. (Omega-3)!") == ["cancer", "diet", "omega", "3"]`;
   - `tokenize("vitamin B12; 1.5 mg") == ["vitamin", "b12", "1", "5", "mg"]`;
   - `tokenize("...") == []`;
   - para todo token `t` da coleção: `re.fullmatch(r"[a-z0-9]+", t)`.
2. Implementar `preprocess(text, cfg) -> list[str]` reaproveitando o T1 → saída: teste com saídas esperadas por configuração.
3. Implementar `surface_vocab(docs) -> Counter[str]` (frequência em documentos dos tokens de superfície) → saída: `artifacts/vocab/surface_{cfg_hash}.json`.
4. Implementar `config_hash(cfg)` → saída: teste de que configurações diferentes geram hashes diferentes.
5. Rodar no dev as configurações candidatas (se houver) com BM25 e registrar → saída: `artifacts/dev/preprocessing_dev.csv` (`config_id, lowercase, remove_stopwords, stemmer, ndcg10_dev, n_queries`).

**Critério de finalização**
- Testes do item 1 a 4 passam.
- `preprocessing_dev.csv` tem uma linha por configuração avaliada com `n_queries == 324`, e a configuração escolhida é a de maior `ndcg10_dev`.
- `config.yaml` contém a configuração escolhida.

**Restrições de implementação**
- "O grupo deverá descrever no relatório quais etapas foram utilizadas, as bibliotecas adotadas e as principais decisões" (RF-1).
- Escolha das etapas no dev (restrição C4).

**Pontos de falha conhecidos**
- Recursos do NLTK (`punkt`, `stopwords`) ausentes no ambiente: baixar explicitamente com versão registrada.
- Tokenizações diferentes entre `tokens` e `surface_tokens` quebram o critério (ii) do RF-4. As duas saídas **devem** vir do mesmo `tokenize`.
- Números, hífens e unidades químicas são comuns no NFCorpus (ex.: "omega-3", "B12"). Pela TDS-DEC-13, "omega-3" vira `["omega", "3"]`, igual nos documentos e nas consultas. Documentar no relatório.
- **Pontuação colada ao token** ("cancer," ≠ "cancer"; "diet." ≠ "diet") divide o vocabulário e derruba o recall. Foi um erro do T1. Tokenizadores como `nltk.word_tokenize` devolvem tokens só de pontuação ("," "." "(") e pedaços como "n't"; por isso a TDS-DEC-13 usa regex. Os testes do item 1 e a checagem `fullmatch` sobre a coleção são o contrato.

---

### TDS-2 — Índice invertido (← C-2) — **implementado** (D1); testes recriados em `tests/test_bm25.py` (o arquivo original do T1 se perdeu)

**Contrato de entrada**

| Campo | Tipo | Formato | Origem | Obrigatório? |
|---|---|---|---|---|
| `processed_docs` | `dict[doc_id, list[str]]` | `tokens` do TDS-1 para os 3.633 documentos | TDS-1 | Sim |

**Contrato de saída** (interface existente em `src/inverted_index.py`)

| Campo | Tipo | Formato | Destino | Garantias |
|---|---|---|---|---|
| `get_postings(t)` | `(np.int32[], np.int32[]) \| None` | `(doc_idx, tf)`, ordenado por `doc_idx` | TDS-3 | `doc_idx` segue `sorted(doc_ids)` |
| `df(t)` | `int` | tamanho da posting list | TDS-3 | `0` se o termo não existe |
| `doc_ids`, `doc_lens`, `avgdl`, `N` | — | — | TDS-3 | `N == 3633` |
| `stats()` | `dict` | `N_docs, avgdl, vocab_size, total_postings, build_time_s` | TDS-13 (relatório) | `total_postings == Σ len(set(tokens_d))` |
| `terms()` | `KeysView[str]` | vocabulário pós-processamento | diagnóstico | — |

**Sub-tarefas atômicas** (restantes)
1. Construir o índice com a configuração congelada → saída: `artifacts/index_stats_{cfg_hash}.json` com o `stats()`.
2. Rodar `tests/test_bm25.py` (recriado: fórmula à mão, bordas, desempate, grade sem reconstruir, não-varredura, estatísticas e regressão contra a implementação do T1 em `tests/reference/bm25_t1.py`, com corpus sintético) e acrescentar uma checagem de não-varredura sobre o NFCorpus real (documentos pontuados por consulta == |união das posting lists|) → saída: testes passando.

**Critério de finalização**
- `tests/test_bm25.py` passa: regressão contra o T1, unidade e não-varredura.
- `index_stats_*.json` existe e `total_postings` é igual à soma recomputada independentemente a partir de `processed_docs`.

**Restrições de implementação**
- "A recuperação lexical deverá obrigatoriamente utilizar um índice invertido implementado do zero" (RF-2).
- "Não será permitido percorrer integralmente todos os documentos da coleção a cada consulta" (RF-2).

**Pontos de falha conhecidos**
- Documento vazio após o pré-processamento: `doc_len = 0`; o BM25 com b = 1 continua definido porque `avgdl > 0` (já testado).
- Reconstruir o índice com outro `cfg_hash` sem invalidar as runs antigas: TDS-14 controla isso.

---

### TDS-3 — Recuperador lexical BM25 (← C-3) — **implementado** (D1)

**Contrato de entrada**

| Campo | Tipo | Formato | Origem | Obrigatório? |
|---|---|---|---|---|
| `query_tokens` | `list[str]` | `tokens` do TDS-1 aplicado à consulta | TDS-1 | Sim |
| `k` | `int \| None` | 100 | config | Sim |
| `k1`, `b` | `float` | config | TDS-14 | Não (padrão 1.2 / 0.75) |

**Contrato de saída**

| Campo | Tipo | Formato | Destino | Garantias |
|---|---|---|---|---|
| `ranking` | `list[(doc_id, float)]` | §0.2 | TDS-7, TDS-6, TDS-13 | `len ≤ 100`; só documentos com score > 0; ordem §0.2 |
| `Run` lexical | `Run` | `artifacts/runs/lexical_{split}.trec` | TDS-6, TDS-7 | uma chave por `query_id` do split, inclusive com lista vazia |
| `avg_query_time_ms` | `float` | — | relatório (RF-2) | média sobre as consultas do split |
| `n_candidates[q]` | `int` | — | TDS-12, TDS-13 | nº de documentos com score > 0 antes do corte em k |

**Sub-tarefas atômicas**
1. Escrever `run_lexical(split) -> Run` → saída: `artifacts/runs/lexical_{split}.trec`.
2. Registrar `n_candidates` por consulta → saída: `artifacts/lexical_candidates_{split}.csv` (`query_id, n_candidates`).
3. (Somente se k1/b forem alterados) Avaliar a grade k1 ∈ {0,5; 1,2; 2,0} × b ∈ {0; 0,75; 1} no dev → saída: `artifacts/dev/bm25_grid_dev.csv` (`k1, b, ndcg10_dev`).

**Critério de finalização**
- `lexical_test.trec` tem exatamente 323 `query_id` distintos (incluindo vazios registrados em `lexical_candidates_test.csv`).
- Nenhuma consulta tem mais de 100 linhas.
- Relendo o arquivo com `ir_measures.read_trec_run`, a ordem é igual à da memória.

**Restrições de implementação**
- "O mesmo modelo lexical deverá ser utilizado nos experimentos subsequentes" (RF-3).
- k1 e b "caso sejam alterados em relação aos valores usuais" escolhidos no dev (Metodologia).

**Pontos de falha conhecidos**
- Consultas com menos de 100 candidatos geram um CSV com menos linhas. **[TDS-DEC-4]** **Não completar** com documentos de score 0 (confirmado pelos autores em 13/11). Explicitar sempre que ocorrer:
  - aviso no console ao gerar a run (`query_id`, `n_candidates`);
  - registro em `artifacts/lexical_candidates_{split}.csv`;
  - menção no relatório (lista das consultas afetadas, ou "nenhuma consulta").
- Consulta toda fora do vocabulário → lista vazia; a chave precisa existir para que o TDS-6 conte métrica 0.

---

### TDS-4 — Corretor ortográfico (← C-4)

**Contrato de entrada**

| Campo | Tipo | Formato | Origem | Obrigatório? |
|---|---|---|---|---|
| `query` | `str` | texto cru | C-0 / consulta com erro | Sim |
| `dictionary` | `Counter[str]` | `surface_vocab(docs)` (termo → df) | TDS-1 | Sim |
| `kgram_index` | `dict[str, set[str]]` | k-grama → termos, com marcador de borda `$` | construído aqui | Sim |
| `k`, `jaccard_min`, `max_edit`, `distance` | `int, float, int, str` | config | TDS-14 | Sim |
| `dense_search` | função `search(query, k=100)` do TDS-5 | texto cru da consulta | TDS-5 | Sim (extensão semântica, [TDS-DEC-12]) |

**Contrato de saída**

| Campo | Tipo | Formato | Destino | Garantias |
|---|---|---|---|---|
| `corrected_query` | `str` | consulta reescrita: termos conhecidos inalterados e cada termo desconhecido substituído (ou mantido, se não houver candidato) | TDS-3 (via TDS-1) | termos ∈ dicionário nunca são alterados |
| `log` | `list[dict]` | `{term, n_kgram_candidates, n_after_jaccard, chosen, edit_distance, reason}` | `artifacts/typo/log.json` | uma entrada por termo desconhecido |

**Algoritmo** (definido pelo RF-4; a parametrização é do dev):
1. Tokenizar a consulta com `tokenize` (TDS-1). Um termo é **desconhecido** se ∉ `dictionary`.
2. **Candidatos:** termos que compartilham ≥ 1 k-grama (com `$` nas bordas) com o termo desconhecido.
3. **Filtro:** manter os candidatos com `Jaccard(kgramas(t), kgramas(c)) ≥ jaccard_min`.
4. **Seleção:** distância de edição **[TDS-DEC-8]** OSA (Damerau restrita: inserção, remoção, substituição e transposição adjacente, todas com custo 1). Manter os de distância ≤ `max_edit`. Escolher a menor distância; empate → maior `df`; empate → ordem lexicográfica.

**Seleção das 10 consultas e dos erros** [TDS-DEC-9]:
- **Elegíveis:** `query_ids["test"]` com `NDCG@10_original > 0` no BM25 congelado.
- **Sorteio:** `random.Random(42).sample(sorted(elegíveis), 10)`.
- **Tipos de erro:** `["insercao","remocao","substituicao","transposicao"]` atribuídos ciclicamente às 10 consultas, na ordem do sorteio. Resultado: 3, 3, 2, 2.
- **Termo alvo:** o token de superfície mais longo da consulta, com ≥ 4 caracteres, que não é stopword (empate → primeiro na consulta).
- **Posição e caractere:** sorteados com `random.Random(hash_estavel(query_id))`; caractere de inserção/substituição de `a–z`.
- **Validação:** se o termo resultante ∈ `dictionary`, sortear de novo (até 50 tentativas); se falhar, passar para o próximo termo elegível.

**Ajuste no dev** [TDS-DEC-10]:
- Aplicar o mesmo gerador de erros a **todas** as consultas elegíveis do dev (NDCG@10 > 0), com uma linha por consulta.
- Grade: `k ∈ {2, 3}` × `jaccard_min ∈ {0.2, 0.3, 0.4, 0.5}` × `max_edit ∈ {1, 2}`.
- Critério: maior `correction_accuracy` (termo corrigido == termo original). Desempate pela maior média de NDCG@10 corrigida.

**Sub-tarefas atômicas**
1. Implementar `kgrams(term, k)` com bordas `$` → saída: teste (`kgrams("casa",2) == {"$c","ca","as","sa","a$"}`).
2. Implementar `jaccard(A, B)` → saída: teste com conjuntos fixos.
3. Implementar `osa_distance(a, b)` → saída: testes (`("ab","ba") == 1`, `("abc","abd") == 1`, `("abc","abcd") == 1`, `("","abc") == 3`).
4. Implementar `build_kgram_index(dictionary, k)` → saída: teste de que todo termo aparece em todos os seus k-gramas.
5. Implementar `correct(query) -> (corrected, log)` → saída: teste com um erro de cada tipo e um termo conhecido inalterado.
6. Implementar o gerador de erros determinístico → saída: teste de que duas execuções produzem as mesmas consultas e que todo termo com erro ∉ `dictionary`.
7. Rodar a grade no dev → saída: `artifacts/dev/spell_grid_dev.csv` (`k, jaccard_min, max_edit, correction_accuracy, ndcg10_corr_mean, n_queries`).
8. Rodar as 3 condições nas 10 consultas do test → saída: `results/typo_experiment.csv` (schema §2.6).
9. **Extensão semântica [TDS-DEC-12]:**
   - rodar as mesmas `query_original`, `query_erro` e `query_corrigida` no bi-encoder (TDS-5), com o texto cru e o mesmo modelo, a mesma revisão e o mesmo `max_seq_length` do pipeline principal;
   - calcular o NDCG@10 de cada condição pelo TDS-6.
   
   → saída: colunas `NDCG10_sem_*`, `recuperacao_sem` e `status_sem` no `typo_experiment.csv`.
10. Resumir a robustez por sistema → saída: `artifacts/typo/robustez_lex_sem.csv` (`system ∈ {lexical, semantic}, queda_media = mean(NDCG10_original − NDCG10_erro), recuperacao_media (só status ok), n_ok, n_erro_nao_reduziu`).

**Critério de finalização**
- `typo_experiment.csv` tem 10 linhas.
- Cada `query_erro` contém exatamente um token ∉ dicionário que não está na `query_original`.
- `NDCG10_original > 0` em todas as linhas.
- `recuperacao` é `(NDCG10_corrigida − NDCG10_erro) / (NDCG10_original − NDCG10_erro)` (tolerância 1e-12) quando o denominador é > 0; caso contrário, `status == "erro_nao_reduziu"` e `recuperacao` é vazio.
- Os 4 tipos aparecem.
- **Extensão semântica:**
  - `NDCG10_sem_original` é igual ao NDCG10 da mesma consulta no `metrics_per_query.csv` (`model = semantic`), com tolerância de 1e-12;
  - `recuperacao_sem` segue a mesma fórmula com as colunas `_sem_`, e `status_sem` segue a mesma regra;
  - `robustez_lex_sem.csv` tem 2 linhas.

**Restrições de implementação** (RF-4)
- "k-gramas para geração de candidatos, coeficiente de Jaccard para filtragem e distância de edição para seleção".
- "(i) o dicionário … formado pelos termos da coleção antes do stemming/lematização"; "(ii) cada erro introduzido deverá gerar um termo inexistente no dicionário"; "(iii) … NDCG@10 na versão original seja maior que zero"; "(iv) … sistema lexical, utilizando NDCG@10".
- "Os demais experimentos deverão utilizar as consultas originais, sem correção" (restrição C8).
- "Aplicar as mesmas consultas com erro ao sistema semântico é opcional e pode enriquecer a discussão sobre a robustez de embeddings a erros ortográficos" (RF-4). Aqui ele é **incluído** (TDS-DEC-12).

**Pontos de falha conhecidos**
- **Seleção ancorada no lexical:** o critério (iii) (NDCG@10 original > 0) vale para o **BM25**. No semântico, uma consulta pode ter `NDCG10_sem_original == 0`. Nesse caso, a queda e a recuperação semânticas ficam indefinidas (`status_sem = erro_nao_reduziu`), e a consulta **não** é trocada por outra.
- **A correção depende do dicionário lexical:** a `query_corrigida` usada no semântico é a mesma do lexical. O experimento mede a robustez do bi-encoder ao erro, não um corretor próprio do semântico.
- **Não reutilizar embeddings de consulta:** as consultas com erro precisam ser codificadas na hora; nada de cache por `query_id`, que devolveria o vetor da consulta original.
- **Denominador ≤ 0:** o erro não reduziu o NDCG@10 (ex.: o termo alvo tinha pouco peso). Isso deve ser registrado e não é falha.
- **Recuperação < 0 ou > 1:** possível (a correção pode piorar ou superar o original). Não truncar.
- **Correção para outra palavra válida:** conta como erro de correção, não do gerador.
- **Termo alvo que é stopword ou removido pelo pré-processamento:** o erro não afeta o BM25. A regra de termo alvo evita stopwords.
- **Dicionário com ruído da coleção** (erros de digitação no próprio corpus): o candidato escolhido pode ser um termo raro; o desempate por `df` mitiga.

---

### TDS-5 — Recuperação semântica (← C-5)

**Contrato de entrada**

| Campo | Tipo | Formato | Origem | Obrigatório? |
|---|---|---|---|---|
| `docs` | `Docs` | via `doc_text` | C-0 | Sim |
| `query` | `str` | texto cru da consulta | C-0 | Sim |
| `dense.*` | config | `model_id`, `revision`, `max_seq_length`, `batch_size` | config (D4) | Sim |

**Contrato de saída**

| Campo | Tipo | Formato | Destino | Garantias |
|---|---|---|---|---|
| `doc_emb.npy` | `float32[3633, 384]` | `artifacts/embeddings/{slug}/doc_emb.npy` | C-5b | linhas L2-normalizadas (`abs(norm − 1) < 1e-5`); ordem = `doc_ids.json` |
| `doc_ids.json` | `list[str]` | mesma ordem das linhas | C-5b | `== sorted(docs)` |
| `manifest.json` | `dict` | schema §2.9 | relatório (RF-5) | todos os campos preenchidos |
| `Run` semântico | `Run` | `artifacts/runs/semantic_{split}.trec` | TDS-6, TDS-7 | exatamente 100 por consulta; ordem §0.2 |
| `query_time_ms` | `float` | média por split | relatório | inclui codificação da consulta e o produto matricial |

**Sub-tarefas atômicas**
1. Carregar `SentenceTransformer(model_id, revision=revision, device="cpu")` e fixar `max_seq_length = 512` → saída: valores registrados no manifesto.
2. Contar os tokens de cada documento com o tokenizer do modelo → saída: `artifacts/embeddings/{slug}/doc_ntokens.csv` (`doc_id, n_tokens, truncated`).
3. Codificar a coleção (`normalize_embeddings=True`, `torch.set_num_threads(4)`) com cronômetro → saída: `doc_emb.npy`, `doc_ids.json`, `encode_time_s` no manifesto.
4. Implementar `search(query, k=100)`: `q = encode(query)`, `s = E @ q`, top-k com desempate §0.2 → saída: teste de que o top-1 de um documento usado como consulta é ele mesmo.
5. Gerar as runs do dev e do test → saída: `semantic_dev.trec`, `semantic_test.trec`.
6. Avaliar no dev (validação D4) → saída: linha em `artifacts/dev/validation_dev.csv` (`system, ndcg10_dev`).

**Critério de finalização**
- `doc_emb.npy` tem shape `(3633, 384)` e todas as normas ∈ [1 − 1e-5, 1 + 1e-5].
- `semantic_test.trec` tem 323 consultas × 100 linhas.
- `manifest.json` contém os 9 itens do RF-5, com o % truncado calculado a partir de `doc_ntokens.csv`.
- `validation_dev.csv` tem a linha `semantic` **com data/hora anterior** à da primeira run do test (TDS-14).

**Restrições de implementação**
- "utilizando um modelo pré-treinado com arquitetura bi-encoder. Os embeddings dos documentos deverão ser calculados previamente e armazenados" (RF-5).
- O relatório deve informar "nome e versão do modelo, biblioteca utilizada, dimensionalidade dos embeddings, função de similaridade, normalização aplicada …, tempo necessário para gerar os embeddings dos documentos e tempo médio de processamento das consultas … comprimento máximo de entrada do modelo (em tokens) e se houve truncamento" (RF-5).

**Pontos de falha conhecidos**
- O modelo foi treinado com até 250 word pieces: documentos entre 250 e 512 tokens não são truncados, mas podem ser mal representados. Registrar `n_tokens` para a análise do RF-7.
- Alterar `max_seq_length` depois de codificar invalida o `.npy`. O manifesto deve guardar o valor e o loader deve conferir.
- `np.argpartition` não é estável: aplicar o desempate §0.2 depois de selecionar os candidatos, considerando empates na fronteira do k.

---

### TDS-6 — Avaliador (← C-6)

**Contrato de entrada**

| Campo | Tipo | Formato | Origem | Obrigatório? |
|---|---|---|---|---|
| `run` | `Run` | §0.1 | C-3, C-5, C-7, C-8 | Sim |
| `qrels` | `Qrels` | do split | C-0 | Sim |
| `query_ids` | `list[str]` | do split | C-0 | Sim |
| `model` | `str` | `lexical \| semantic \| hybrid \| reranking` | chamador | Sim |

**Contrato de saída**

| Campo | Tipo | Formato | Destino | Garantias |
|---|---|---|---|---|
| `per_query` | `DataFrame` | `query_id, model, P10, R10, AP10, NDCG10` | TDS-13, TDS-12 | uma linha por `query_id` de `query_ids` (inclusive run vazia) |
| `aggregate` | `dict` | `P@10, R@10, MAP@10, NDCG@10` | relatório | `MAP@10 == per_query.AP10.mean()` |
| `rag_metrics` | `DataFrame` | `query_id, R5, P5, Hit5` | TDS-13 | 10 linhas |

**Definições implementadas** (RD §4; `rel(q) = {d : g(d) > 0}`, `R = |rel(q)|`):
- `P10 = |top10 ∩ rel| / 10`.
- `R10 = |top10 ∩ rel| / R`.
- `AP10 = Σ_{k ≤ 10, r_k ∈ rel} P@k / R`.
- `NDCG10 = DCG@10 / IDCG@10`, com:
  - `DCG = Σ_{i=1}^{10} g(r_i) / log2(i + 1)`;
  - `IDCG` calculado com os 10 maiores graus de **todos** os julgamentos da consulta;
  - ganho linear.
- `R5`, `P5` e `Hit5` (Hit5 = 1 se |top5 ∩ rel| ≥ 1) usam a mesma relevância e o mesmo denominador `R`.

**[TDS-DEC-5]** Consulta com `R == 0`: R10 = AP10 = NDCG10 = 0, alinhado ao `trec_eval` e ao `ir_measures`, que retornam 0. Contar e registrar no `data_stats.json`.

**Sub-tarefas atômicas**
1. Implementar as 4 métricas por consulta → saída: testes com rankings feitos à mão (incluindo run vazia e R = 0).
2. Implementar a agregação → saída: teste `MAP@10 == mean(AP10)`.
3. Implementar a validação cruzada com `ir_measures` (`P(rel=1)@10`, `R(rel=1)@10`, `AP(rel=1)@10`, `nDCG@10`) → saída: `tests/test_eval_vs_ir_measures.py`.
4. Implementar R5/P5/Hit5 → saída: testes unitários.

**Critério de finalização**
- Para os 4 sistemas no test, `|métrica_própria − métrica_ir_measures| ≤ 1e-9` em **todas** as 323 consultas e nas 4 métricas.
- A comparação preenche com 0 as consultas ausentes da run antes de calcular as médias do `ir_measures`.
- `per_query` tem 323 linhas por sistema.

**Restrições de implementação** (RF-6)
- "Para Precision@10, Recall@10 e MAP@10, considere como relevante todo documento com grau de relevância maior que zero."
- "a soma de Precision@k nas posições k ≤ 10 que contêm documentos relevantes, dividida pelo número total de documentos relevantes da consulta nos qrels."
- "utilize ganho linear … e calcule o ranking ideal a partir de todos os julgamentos da consulta."
- "todas as métricas deverão ser preservadas por consulta."

**Pontos de falha conhecidos**
- **`ir_measures` ignora consultas ausentes da run:** a média fica inflada se uma consulta lexical voltar vazia. Sempre agregar pela lista `query_ids`, não pela run.
- **P@10 com menos de 10 recuperados:** o denominador continua 10.
- **Graus 0 no qrels:** são não relevantes para P/R/AP e têm ganho 0 no NDCG. Não confundir com "não julgado".
- **Documentos repetidos numa run:** deduplicar (ou falhar) antes de avaliar.

---

### TDS-7 — Fusão RRF (← C-7)

**Contrato de entrada**

| Campo | Tipo | Formato | Origem | Obrigatório? |
|---|---|---|---|---|
| `run_lex`, `run_sem` | `Run` | truncadas em `rrf.depth = 100` | TDS-3, TDS-5 | Sim |
| `k` | `int` | config ou grade do dev | TDS-14 | Sim |

**Contrato de saída**

| Campo | Tipo | Formato | Destino | Garantias |
|---|---|---|---|---|
| `Run` híbrido | `Run` | união (≤ 200 por consulta), ordenada por RRF; desempate §0.2 | TDS-8, TDS-6 | `score = Σ_s 1/(k + rank_s)` com `rank_s` começando em 1; ausente → contribuição 0 |
| `hybrid_{split}.trec` | arquivo | top-100 exportado | TDS-6, TDS-13 | ≤ 100 linhas por consulta |
| `rrf_k_dev.csv` | CSV | `k, ndcg10_dev, n_queries` | relatório | k ∈ {10, 60, 100} no mínimo |

**Sub-tarefas atômicas**
1. Implementar `rrf(runs, k, depth=100)` → saída: teste com 2 runs de 3 documentos e scores calculados à mão (incluindo um documento presente em só uma run).
2. Rodar a grade no dev → saída: `artifacts/dev/rrf_k_dev.csv`.
3. Fixar `rrf.k = argmax ndcg10_dev` no `config.yaml` (empate → menor k) → saída: config atualizado e registrado no `freeze_log.json` (TDS-14).
4. Gerar as runs híbridas do dev e do test → saída: `hybrid_{split}.trec`.

**Critério de finalização**
- `rrf_k_dev.csv` tem ≥ 3 linhas com `n_queries == 324`.
- O `k` do config é o argmax.
- Para 5 consultas sorteadas (seed 42), o score RRF do top-1 recomputado a partir das runs de entrada coincide com o arquivo (tolerância 1e-12).

**Restrições de implementação** (RF-8)
- "RRF(d) = Σ 1/(k + rank_s(d)) … rank_s(d) é a posição de d (começando em 1) … documentos ausentes de um dos rankings não recebem contribuição desse sistema. Deverão ser fundidos os 100 primeiros documentos de cada sistema."
- "avaliando pelo menos k ∈ {10, 60, 100}, escolhendo-se o valor com maior NDCG@10 médio no dev. … O valor selecionado deverá ser mantido na avaliação final."

**Pontos de falha conhecidos**
- Rank começando em 0 por engano: o teste do item 1 detecta.
- Lexical com menos de 100 itens: a fusão continua válida, e o documento só contribui de um lado.
- Muitos empates de RRF (documentos na mesma posição em um só sistema): o desempate §0.2 precisa ser aplicado.

---

### TDS-8 — Reranker (← C-8)

**Contrato de entrada**

| Campo | Tipo | Formato | Origem | Obrigatório? |
|---|---|---|---|---|
| `run_hybrid` | `Run` | ≥ 50 itens (exceto consultas com união menor) | TDS-7 | Sim |
| `docs`, `queries` | — | `doc_text`, texto cru da consulta | C-0 | Sim |
| `rerank.*` | config | `model_id`, `revision`, `max_length=512`, `depth=50` | config (D5) | Sim |

**Contrato de saída**

| Campo | Tipo | Formato | Destino | Garantias |
|---|---|---|---|---|
| `Run` reranqueado | `Run` | posições 1–50 = top-50 do híbrido reordenado pelo cross-encoder; 51–100 = posições 51–100 do híbrido na mesma ordem | TDS-6, TDS-9, TDS-13 | o conjunto dos 50 primeiros é igual ao do híbrido; as posições 51–100 são idênticas |
| `rerank_scores.parquet` | Parquet | `query_id, doc_id, hybrid_rank, ce_score, n_tokens, truncated, model_rev` | TDS-12 | 50 linhas por consulta |
| `rerank_time_ms` | `float` | média por consulta | relatório (RF-9) | só inferência |

**[TDS-DEC-6] Score no CSV de reranking:**
- Posições 1–50: `score = ce_score` (logit, sem ativação; registrar a ativação aplicada).
- Posições 51–100: `score` = score RRF do híbrido.
- A coluna extra `score_source ∈ {cross_encoder, hybrid_rrf}` (schema §2.4) explicita a mistura de escalas.

**Sub-tarefas atômicas**
1. Carregar `CrossEncoder(model_id, revision, max_length=512, device="cpu")` e registrar a ativação → saída: campos no `artifacts/rerank_manifest.json`.
2. Contar tokens dos pares (consulta, documento) → saída: colunas `n_tokens`, `truncated` no Parquet.
3. Pontuar os 50 pares por consulta com cronômetro → saída: `rerank_scores_{split}.parquet`.
4. Montar o ranking final → saída: `reranking_{split}.trec`; teste de invariância das posições 51–100.
5. Avaliar no dev (validação D5) → saída: linha `reranking` em `validation_dev.csv`.

**Critério de finalização**
- `reranking_test.trec` tem 323 consultas.
- Para toda consulta: `set(top50_rerank) == set(top50_hybrid)` e `rerank[50:100] == hybrid[50:100]`.
- `rerank_manifest.json` informa `max_length`, a % de pares truncados e o tempo médio.
- A linha `reranking` de `validation_dev.csv` precede a run do test.

**Restrições de implementação** (RF-9)
- "Os 50 primeiros documentos do ranking híbrido deverão ser reordenados … Os documentos originalmente abaixo da posição 50 deverão permanecer, após o reranking, na mesma ordem relativa do ranking híbrido."
- "O tempo médio das etapas também deverá ser registrado. Informe também o comprimento máximo de entrada do cross-encoder e se houve truncamento."

**Pontos de falha conhecidos**
- `predict` com ativação sigmoide versus logits: a ordem não muda, mas o valor no CSV sim. Registrar qual foi usada.
- Truncamento padrão `longest_first` corta a consulta se ela for longa; as consultas do NFCorpus são curtas, mas é preciso verificar `n_tokens`.
- O ganho do reranking não é garantido com o L2-v2 (AD §3.2): não "corrigir" resultados ruins mudando o modelo depois de ver o test (restrição C4).

---

### TDS-9 — Seletor de contexto (← C-9)

**Contrato de entrada**

| Campo | Tipo | Formato | Origem | Obrigatório? |
|---|---|---|---|---|
| `run_rerank_test` | `Run` | test | TDS-8 | Sim |
| `rag_query_ids` | `list[str]` | 10 IDs (TDS-0, item 6) | C-0 | Sim |
| `docs` | `Docs` | — | C-0 | Sim |

**Contrato de saída**

| Campo | Tipo | Formato | Destino | Garantias |
|---|---|---|---|---|
| `contexts.json` | `dict[query_id, list[{rank, doc_id, title, text}]]` | 5 itens por consulta, ordem do reranking | TDS-10, TDS-11 | `doc_id` = top-5 de `reranking_test.trec` |
| `rag_metrics` | via TDS-6 | `R5, P5, Hit5` | TDS-13, relatório | 10 linhas |

**Sub-tarefas atômicas**
1. Extrair o top-5 por consulta → saída: `artifacts/rag/contexts.json`.
2. Calcular R5/P5/Hit5 → saída: `artifacts/rag/rag_retrieval_metrics.csv` (`query_id, R5, P5, Hit5, n_rel_total`).

**Critério de finalização**
- `contexts.json` tem 10 chaves iguais ao `rag_queries.json`, com 5 itens cada.
- Os `doc_id` coincidem com as 5 primeiras linhas de `reranking_test.trec` para cada consulta.

**Restrições de implementação**
- "cada item do corpus.jsonl será tratado como uma unidade de contexto" (RF-11).
- "Top-5 documentos obtidos após o reranking, com seus respectivos identificadores" (RF-12).

**Pontos de falha conhecidos**
- Documentos longos aumentam o prompt (estimado em ~2K tokens no total). **[TDS-DEC-7]** Sem truncamento do texto no contexto, salvo se o prompt passar de 8.000 tokens (improvável); nesse caso, truncar cada documento igualmente e registrar.

---

### TDS-10 — Construtor de prompt + cliente LLM (← C-10)

**Contrato de entrada**

| Campo | Tipo | Formato | Origem | Obrigatório? |
|---|---|---|---|---|
| `question` | `str` | `queries[qid]` | C-0 | Sim |
| `context` | `list[dict]` \| None | `contexts[qid]` (só B) | TDS-9 | B: sim; A: não |
| `llm.*` | config | §0.6 | config (D2) | Sim |
| `GEMINI_API_KEY` | env var | — | ambiente | Sim (nunca no repositório) |

**Contrato de saída**

| Campo | Tipo | Formato | Destino | Garantias |
|---|---|---|---|---|
| `responses.jsonl` | JSONL | uma linha por chamada (schema §2.10) | TDS-11, TDS-12, TDS-13 | 20 linhas (10 × {A, B}); **imutável** após gerado |

**Templates de prompt** (texto exato no código e no `rag_experiment.csv`; idioma conforme `llm.language` [PENDENTE: D2]):
- **A:** instrução de sistema mínima ("Responda à pergunta.") + pergunta. Sem contexto e sem menção à coleção.
- **B:** instrução de sistema com os **três** elementos obrigatórios:
  1. responder **somente** com base no contexto;
  2. citar os `doc_id` usados entre colchetes;
  3. declarar explicitamente quando o contexto não tiver evidência suficiente.
  
  Depois: blocos `[doc_id] título\ntexto` na ordem do reranking, seguidos da pergunta.

**Sub-tarefas atômicas**
1. Implementar `build_prompt(condition, question, context)` → saída: teste de snapshot dos dois templates.
2. Implementar o cliente com `temperature`, `seed`, `max_output_tokens` e `thinking` do config, e retry exponencial para 429 (máx. 5 tentativas) → saída: teste com cliente simulado.
3. Teste de determinismo [PENDENTE: D2]: 2 chamadas idênticas → saída: `artifacts/rag/determinism_check.json` (`identical: bool`, `modelVersion`).
4. Gerar as 20 respostas **uma vez** → saída: `artifacts/rag/responses.jsonl`, mais o SHA-256 em `freeze_log.json`.

**Critério de finalização**
- `responses.jsonl` tem 20 linhas, com pares (query_id, condition) únicos.
- `model_id` e todos os parâmetros são iguais nas 20 linhas.
- `modelVersion` é igual nas 20 (se diferir, o critério falha e a geração é refeita inteira).
- `prompt` B contém as 3 instruções; `prompt` A não contém nenhum `doc_id`.
- O hash do arquivo está registrado.

**Restrições de implementação** (RF-11/12)
- "(A) sem RAG, em que a LLM recebe somente a pergunta … (B) com RAG, em que a LLM recebe a pergunta e os Top-5 documentos obtidos após o reranking."
- "O prompt do RAG deverá instruir o modelo a responder com base no contexto fornecido, indicar as fontes utilizadas e informar quando não houver evidência suficiente."
- "mesma LLM e os mesmos parâmetros"; "mesmo idioma nas duas condições"; temperatura 0 e seed fixa recomendadas.
- Sem credenciais na entrega (restrição C15).

**Pontos de falha conhecidos**
- Resposta truncada por `max_output_tokens` (`finishReason` diferente de `STOP`): registrar e aumentar o limite **para todas as 20** chamadas, regenerando o conjunto.
- Bloqueio por filtro de segurança em temas médicos: registrar o `finishReason` e não editar a pergunta.
- Cota do free tier esgotada no meio da geração: o cache incremental por (query_id, condition) permite retomar sem duplicar.

---

### TDS-11 — Avaliação humana cega (← C-11)

**Contrato de entrada**

| Campo | Tipo | Formato | Origem | Obrigatório? |
|---|---|---|---|---|
| `responses.jsonl` | JSONL | §2.10 | TDS-10 | Sim |
| `contexts.json` | JSON | top-5 por consulta | TDS-9 | Sim |

**Contrato de saída**

| Campo | Tipo | Formato | Destino | Garantias |
|---|---|---|---|---|
| `blind_sheet.csv` | CSV | `item_id, query_id, question, answer, top5_context` + colunas de nota e justificativa vazias | avaliador | 20 linhas embaralhadas com `random.Random(42)`; **sem** a coluna `condition` |
| `blind_key.json` | JSON | `item_id → condition` | TDS-13 | não aberto durante a avaliação |
| `scores.csv` | CSV | sheet preenchido | TDS-13 | notas ∈ {0, 1, 2}; justificativa não vazia; rastreabilidade de A = `N/A` após o desembaralhamento |

**Sub-tarefas atômicas**
1. Gerar a planilha embaralhada e a chave → saída: `blind_sheet.csv`, `blind_key.json` (teste: a planilha não contém "A"/"B" em nenhuma coluna).
2. Avaliar manualmente (Fidelidade sempre contra o `top5_context` daquela consulta) → saída: `scores.csv`.
3. Desembaralhar, impor `rastreabilidade = N/A` em A e calcular as médias por condição (A sem rastreabilidade) → saída: `artifacts/rag/human_eval_summary.csv` (`condition, criterio, media, n`).
4. (Se houver 2 avaliadores) calcular a concordância → saída: `artifacts/rag/agreement.json` (% de acordo e kappa de Cohen por critério).

**Critério de finalização**
- `scores.csv` tem 20 linhas × 4 notas válidas com justificativa.
- `human_eval_summary.csv` mostra `n == 10` por critério em B, `n == 10` em A para Fidelidade, Completude e Utilidade, e nenhuma linha de Rastreabilidade para A.

**Restrições de implementação** (RF-13)
- "notas 0, 1 ou 2 … Cada nota deverá possuir uma breve justificativa."
- "Fidelidade deverá ser avaliada comparando a resposta com os mesmos Top-5 documentos usados na condição com RAG."
- "A Rastreabilidade deverá ser registrada como 'N/A' e excluída da média dessa condição."

**Pontos de falha conhecidos**
- O cegamento é **parcial**: respostas B citam `doc_id` e as A não, então a condição fica dedutível. Declarar essa limitação no relatório.
- Com um único avaliador, não há concordância (RD: "quando possível"). Declarar.

---

### TDS-12 — Módulo de análises (← C-12)

**Contrato de entrada**

| Campo | Tipo | Formato | Origem | Obrigatório? |
|---|---|---|---|---|
| `per_query` | DataFrame | métricas dos 4 sistemas no test | TDS-6 | Sim |
| runs | `.trec` | 4 sistemas, test | TDS-3/5/7/8 | Sim |
| `rerank_scores_test.parquet`, `doc_ntokens.csv`, `index_stats` | — | — | TDS-8, TDS-5, TDS-2 | Sim |
| `responses.jsonl`, `scores.csv` | — | — | TDS-10, TDS-11 | RF-14 |

**Contrato de saída**

| Campo | Tipo | Formato | Destino | Garantias |
|---|---|---|---|---|
| `rf7_cases.csv` | CSV | `group ∈ {sem_win, lex_win}, query_id, query, ndcg_lex, ndcg_sem, delta` | relatório | 3 + 3 linhas; seleção determinística |
| `rf7_top5.csv` | CSV | `query_id, system, rank, doc_id, score, relevance, tf_query_terms, doc_len` | relatório | 6 consultas × 2 sistemas × 5 |
| `rf10_cases.csv` | CSV | `case ∈ {gain1, gain2, nochange, loss}, query_id, ndcg_hybrid, ndcg_rerank, delta` | relatório | 4 linhas |
| `rf10_moves.csv` | CSV | `query_id, doc_id, rank_hybrid, rank_rerank, relevance, ce_score` | relatório | documentos com \|Δrank\| ≥ 5 no top-10 de algum dos dois |
| `rf14_trace.csv` | CSV | `query_id, doc_id, relevance, rank_lexical, rank_semantic, rank_hybrid, rank_rerank, in_context` | relatório | para os relevantes das 3 consultas do RF-14 |

**Regras de seleção** [TDS-DEC-11]:
- **RF-7:** `delta = NDCG10_sem − NDCG10_lex`; os 3 maiores e os 3 menores; empate → `query_id` crescente.
- **RF-10:**
  - ganhos: os 2 maiores `delta = NDCG10_rerank − NDCG10_hybrid`;
  - piora: o menor `delta` entre os < 0;
  - "pouca ou nenhuma alteração": a consulta com o menor \|delta\| e com ≥ 1 relevante no top-10 do híbrido.
- **RF-14:** classificar as 10 consultas pela diferença média das notas (B − A) sobre Fidelidade, Completude e Utilidade (Rastreabilidade excluída, pois é N/A em A). Maior = "melhorou"; mais próxima de 0 = "pouca diferença"; menor = "não ajudou ou prejudicou".

**Sub-tarefas atômicas**
1. Gerar `rf7_cases.csv` e `rf7_top5.csv` → saída: os dois arquivos.
2. Gerar `rf10_cases.csv` e `rf10_moves.csv` → saída: os dois arquivos.
3. Gerar `rf14_trace.csv` → saída: o arquivo.
4. Calcular as evidências numéricas usadas no texto (tf dos termos, \|D\|, `n_tokens`, rank) → saída: colunas nos CSVs acima.

**Critério de finalização**
- Os 5 CSVs existem com as cardinalidades indicadas.
- Toda afirmação numérica do relatório nas seções RF-7, RF-10 e RF-14 aponta para uma célula de um desses arquivos.

**Restrições de implementação**
- "Afirmações como 'o documento era muito longo' … somente deverão ser feitas quando sustentadas por valores efetivamente observados nos próprios dados" (RF-7).
- "deverá ser verificado se a informação necessária chegou ou não ao contexto" (RF-14).
- Não comparar scores entre modelos (restrição C20).

**Pontos de falha conhecidos**
- Menos de uma consulta com delta < 0 no RF-10: registrar "nenhum caso de piora" e não forçar um caso.
- Empates em delta = 0 em muitas consultas lexicais/semânticas com NDCG = 0: a regra de desempate garante o determinismo.

---

### TDS-13 — Escritor de resultados (← C-13)

**Contrato de entrada:** runs do test (TDS-3/5/7/8), `per_query` (TDS-6), log do RF-4 (TDS-4), `responses.jsonl` + `scores.csv` + `blind_key.json` (TDS-10/11).

**Contrato de saída:** os 7 CSVs de `results/` (schemas na §2).

**Sub-tarefas atômicas**
1. Escrever os 4 CSVs de ranking a partir das runs do test, juntando `relevance` do qrels (0 se não julgado) → saída: `results_{lexical,semantic,hybrid,reranking}.csv`.
2. Escrever `metrics_per_query.csv` → saída: o arquivo.
3. Escrever `typo_experiment.csv` → saída: o arquivo.
4. Escrever `rag_experiment.csv` → saída: o arquivo.
5. Implementar `validate_results()` com todas as checagens da §2 → saída: `tests/test_results_schema.py` passando.

**Critério de finalização**
- `pytest tests/test_results_schema.py` passa.
- `mean(AP10)` por `model` em `metrics_per_query.csv` é igual ao MAP@10 reportado no relatório, com 4 casas.

**Restrições de implementação**
- "Os arquivos de ranking deverão conter, no mínimo, query_id, doc_id, rank, score e relevance … os 100 primeiros documentos de cada consulta" (RF-15).
- "Resultados existentes apenas como saída de células de notebook não serão considerados" (restrição C12).

**Pontos de falha conhecidos**
- `float` com vírgula decimal (locale pt-BR): forçar `.` e UTF-8.
- O Excel converte IDs como `MED-10`? Não converte, mas evitar abrir e salvar os CSVs nele.

---

### TDS-14 — Orquestrador com trava de split (← C-14)

**Contrato de entrada**

| Campo | Tipo | Formato | Origem | Obrigatório? |
|---|---|---|---|---|
| `config.yaml` | YAML | §0.6 | repositório | Sim |
| `stage` | `str` | `preproc \| bm25 \| dense_val \| rrf \| rerank_val \| spell \| test \| typo \| rag` | CLI | Sim |

**Contrato de saída**

| Campo | Tipo | Formato | Destino | Garantias |
|---|---|---|---|---|
| `freeze_log.json` | JSON | `{config_hash, frozen_at, stage_results: {stage: {file, sha256, timestamp}}}` | auditoria | append-only |
| `test_run.lock` | arquivo | `config_hash` + timestamp | auditoria | criado na 1ª execução do test |

**Regras:**
- Qualquer estágio de ajuste ou validação chamado com `split="test"` levanta `SplitGuardError`.
- O estágio `test` exige `frozen: true` e as linhas `semantic` e `reranking` em `validation_dev.csv`.
- Se `test_run.lock` já existir com **outro** `config_hash`, o estágio `test` falha, exceto com `--force --reason "<texto>"`, que fica registrado no `freeze_log.json`.

**Sub-tarefas atômicas**
1. Implementar `SplitGuard` → saída: teste de que `stage="rrf", split="test"` levanta exceção.
2. Implementar o congelamento (hash do config + registro) → saída: `freeze_log.json`.
3. Implementar o lock do test → saída: teste com uma segunda execução bloqueada.
4. Implementar os scripts `run_dev.py`, `run_test.py`, `run_typo.py` e `run_rag.py` na ordem da AD §1.2 → saída: execução de ponta a ponta sem erro.

**Critério de finalização**
- `run_dev.py` → `run_test.py` → `run_typo.py` → `run_rag.py`, executados em ambiente limpo, geram os 7 CSVs.
- Os timestamps do `freeze_log.json` mostram que todas as entradas do dev precedem `test_run.lock`.

**Restrições de implementação**
- Restrições C3, C4, C5 e C7 do RD.
- "O valor selecionado deverá ser mantido na avaliação final sobre o conjunto test" (RF-8).

**Pontos de falha conhecidos**
- Mudança de pré-processamento depois do RF-4 invalida o dicionário e as consultas com erro: o `cfg_hash` do TDS-1 deve constar no `typo_experiment` (`artifacts/typo/meta.json`).

---

## 2. Schemas Completos dos Artefatos de Entrega

Codificação: UTF-8, separador `,`, decimal `.`, cabeçalho na 1ª linha, `\n` como fim de linha. Os exemplos de linha são **ilustrativos** (valores fictícios).

### 2.1 `results_lexical.csv`

| Coluna | Tipo | Descrição | Restrição | Componente gerador |
|---|---|---|---|---|
| `query_id` | str | ID da consulta do test | ∈ `query_ids["test"]`; não nulo | TDS-13 ← TDS-3 |
| `doc_id` | str | ID do documento | ∈ corpus; único por `query_id` | TDS-13 ← TDS-3 |
| `rank` | int | posição | 1..n_q, contígua, n_q ≤ 100 | TDS-13 |
| `score` | float | score BM25 | > 0; não crescente com o rank | TDS-3 |
| `relevance` | int | grau no qrels do test | ≥ 0; 0 se não julgado | TDS-13 ← C-0 |

Exemplo: `PLAIN-68,MED-1234,1,14.2381,2`

Checagens: ≤ 100 linhas por consulta; 323 consultas presentes **ou** ausência justificada em `artifacts/lexical_candidates_test.csv` (n_candidates = 0) [TDS-DEC-4].

### 2.2 `results_semantic.csv`

Mesmo schema do §2.1, com:
- `score`: similaridade do cosseno ∈ [−1, 1], não crescente com o rank;
- exatamente **100** linhas por consulta;
- 323 consultas (32.300 linhas).

Exemplo: `PLAIN-68,MED-2001,1,0.7124,1`

### 2.3 `results_hybrid.csv`

Mesmo schema do §2.1, com:
- `score`: valor RRF ∈ (0, 2/(k + 1)], não crescente com o rank;
- `rank` ≤ 100.

Exemplo (k = 60): `PLAIN-68,MED-1234,1,0.032787,2`

### 2.4 `results_reranking.csv`

| Coluna | Tipo | Descrição | Restrição | Componente gerador |
|---|---|---|---|---|
| `query_id` | str | ID da consulta | ∈ test | TDS-13 ← TDS-8 |
| `doc_id` | str | ID do documento | único por consulta | TDS-8 |
| `rank` | int | posição final | 1..n_q ≤ 100 | TDS-8 |
| `score` | float | `ce_score` (rank ≤ 50) ou RRF (rank > 50) [TDS-DEC-6] | não crescente **dentro** de cada `score_source` | TDS-8 |
| `relevance` | int | grau no qrels | ≥ 0 | TDS-13 |
| `score_source` | str | origem do score (coluna adicional) | ∈ {`cross_encoder`, `hybrid_rrf`}; `cross_encoder` ⇔ rank ≤ 50 | TDS-8 |

Exemplos: `PLAIN-68,MED-2001,1,7.8812,1,cross_encoder` · `PLAIN-68,MED-0991,51,0.016129,0,hybrid_rrf`

### 2.5 `metrics_per_query.csv`

| Coluna | Tipo | Descrição | Restrição | Componente gerador |
|---|---|---|---|---|
| `query_id` | str | ID da consulta | ∈ test | TDS-6 |
| `model` | str | sistema | ∈ {`lexical`, `semantic`, `hybrid`, `reranking`} | TDS-6 |
| `P10` | float | Precision@10 | ∈ [0, 1]; múltiplo de 0,1 | TDS-6 |
| `R10` | float | Recall@10 (denominador = relevantes no qrels) | ∈ [0, 1] | TDS-6 |
| `AP10` | float | AP@10 (denominador = relevantes no qrels) | ∈ [0, 1] | TDS-6 |
| `NDCG10` | float | nDCG@10, ganho linear, ideal de todos os julgamentos | ∈ [0, 1] | TDS-6 |

Exemplo: `PLAIN-68,reranking,0.4,0.0952,0.0731,0.4410`

Checagens:
- 323 × 4 = **1.292 linhas**; par (`query_id`, `model`) único;
- `mean(AP10)` por `model` == MAP@10 reportado;
- igual ao `ir_measures` com tolerância de 1e-9.

### 2.6 `typo_experiment.csv`

| Coluna | Tipo | Descrição | Restrição | Componente gerador |
|---|---|---|---|---|
| `query_id` | str | ID da consulta | ∈ test; 10 únicos | TDS-4 |
| `query_original` | str | texto original | = `queries[query_id]` | TDS-4 |
| `query_erro` | str | texto com um erro | exatamente 1 token ∉ dicionário que não está na original | TDS-4 |
| `query_corrigida` | str | saída do corretor | não nulo (pode ser igual a `query_erro` se não houver candidato) | TDS-4 |
| `tipo_erro` | str | tipo do erro | ∈ {`insercao`, `remocao`, `substituicao`, `transposicao`} | TDS-4 |
| `NDCG10_original` | float | NDCG@10 BM25 | ∈ (0, 1] | TDS-6 |
| `NDCG10_erro` | float | NDCG@10 BM25 | ∈ [0, 1] | TDS-6 |
| `NDCG10_corrigida` | float | NDCG@10 BM25 | ∈ [0, 1] | TDS-6 |
| `termo_original` | str | termo alterado (adicional) | ∈ dicionário | TDS-4 |
| `termo_erro` | str | termo com erro (adicional) | ∉ dicionário | TDS-4 |
| `termo_corrigido` | str | escolha do corretor (adicional) | nullable | TDS-4 |
| `recuperacao` | float | fórmula do RF-4 (adicional) | nullable; vazio ⇔ denominador ≤ 0 | TDS-4 |
| `status` | str | adicional | ∈ {`ok`, `erro_nao_reduziu`} | TDS-4 |
| `NDCG10_sem_original` | float | NDCG@10 bi-encoder, consulta original (adicional, TDS-DEC-12) | ∈ [0, 1]; = `metrics_per_query` (semantic) | TDS-4 + TDS-5 |
| `NDCG10_sem_erro` | float | NDCG@10 bi-encoder, consulta com erro (adicional) | ∈ [0, 1] | TDS-4 + TDS-5 |
| `NDCG10_sem_corrigida` | float | NDCG@10 bi-encoder, consulta corrigida (adicional) | ∈ [0, 1] | TDS-4 + TDS-5 |
| `recuperacao_sem` | float | fórmula do RF-4 com as colunas `_sem_` (adicional) | nullable; vazio ⇔ denominador ≤ 0 | TDS-4 |
| `status_sem` | str | adicional | ∈ {`ok`, `erro_nao_reduziu`} | TDS-4 |

Exemplo: `PLAIN-1008,"breast cancer cells feed on cholesterol","breast cancer cells feed on cholesterl","breast cancer cells feed on cholesterol",remocao,0.5123,0.2011,0.5123,cholesterol,cholesterl,cholesterol,1.0,ok,0.4870,0.4512,0.4870,1.0,ok`

### 2.7 `rag_experiment.csv`

| Coluna | Tipo | Descrição | Restrição | Componente gerador |
|---|---|---|---|---|
| `query_id` | str | ID da consulta | ∈ as 10 do Req. 12 | TDS-13 |
| `query` | str | pergunta | = `queries[query_id]` | TDS-13 |
| `condition` | str | condição | ∈ {`A`, `B`}; par (`query_id`, `condition`) único | TDS-10 |
| `context_doc_ids` | str | IDs enviados à LLM, separados por `;`, na ordem do rank | B: 5 IDs; A: vazio | TDS-9 |
| `prompt` | str | prompt completo enviado | não nulo | TDS-10 |
| `answer` | str | resposta da LLM | não nulo | TDS-10 |
| `fidelidade` | int | nota | ∈ {0, 1, 2} | TDS-11 |
| `completude` | int | nota | ∈ {0, 1, 2} | TDS-11 |
| `utilidade` | int | nota | ∈ {0, 1, 2} | TDS-11 |
| `rastreabilidade` | str | nota | B: ∈ {`0`, `1`, `2`}; A: `N/A` | TDS-11 |
| `just_fidelidade` | str | justificativa | não vazio | TDS-11 |
| `just_completude` | str | justificativa | não vazio | TDS-11 |
| `just_utilidade` | str | justificativa | não vazio | TDS-11 |
| `just_rastreabilidade` | str | justificativa | B: não vazio; A: "Sem contexto externo (condição A)" | TDS-11 |
| `fidelity_reference_doc_ids` | str | Top-5 usados para julgar a Fidelidade (adicional) | = `context_doc_ids` de B da mesma consulta | TDS-11 |
| `model_id`, `model_version`, `temperature`, `seed`, `max_output_tokens`, `language` | str/float/int | parâmetros (adicionais) | iguais nas 20 linhas | TDS-10 |

Exemplo (B): `PLAIN-68,"Do ... ?",B,MED-10;MED-14;MED-301;MED-77;MED-912,"<prompt>","<resposta> [MED-10]",2,1,2,2,"...","...","...","...",MED-10;MED-14;MED-301;MED-77;MED-912,gemini-3.1-flash-lite,<versão>,0.0,42,<n>,<idioma>`

### 2.8 Artefatos obrigatórios não-CSV

| Artefato | Conteúdo mínimo | Restrição | Gerado por |
|---|---|---|---|
| Código-fonte (`src/`, `scripts/`, `tests/`, `config/`) | tudo o que é necessário para reproduzir | sem dataset, embeddings, modelos, índice ou venv; entregue no Moodle | todos |
| `README.md` | campos da §3 | — | manual |
| Relatório PDF | seções a–g (RD §6.4) | ≤ 20 páginas sem as referências; sem anexos para contornar | manual |
| `requirements.txt` | versões **fixadas** (`==`) | inclui `sentence-transformers`, `torch`, `transformers`, `ir_measures`, `numpy`, `pandas`, `nltk`, `google-genai` | manual |

### 2.9 `manifest.json` (embeddings; não entregue, mas citado no relatório)

| Campo | Tipo | Descrição | Restrição |
|---|---|---|---|
| `model_id` | str | ID do HF | = `dense.model_id` |
| `revision` | str | commit hash | 40 caracteres hexadecimais |
| `library_versions` | dict | `torch`, `transformers`, `sentence_transformers` | não nulo |
| `dim` | int | dimensão | 384 |
| `max_seq_length` | int | — | 512 |
| `pooling` | str | — | do modelo (mean) |
| `normalized` | bool | — | true |
| `similarity` | str | — | `cosine` |
| `doc_representation` | str | — | `title + " " + text` |
| `n_docs` | int | — | 3633 |
| `n_truncated`, `pct_truncated` | int, float | documentos com `n_tokens > 512` | ≥ 0 |
| `encode_time_s` | float | tempo para codificar a coleção | > 0 |
| `hardware` | str | CPU, threads, RAM | não nulo |
| `npy_sha256` | str | hash do `.npy` | 64 caracteres hexadecimais |

### 2.10 `responses.jsonl` (intermediário; base do `rag_experiment.csv`)

| Campo | Tipo | Restrição |
|---|---|---|
| `query_id` | str | ∈ as 10 do Req. 12 |
| `condition` | str | ∈ {A, B} |
| `prompt` | str | — |
| `context_doc_ids` | list[str] | B: 5; A: [] |
| `answer` | str | — |
| `finish_reason` | str | esperado `STOP` |
| `model_id`, `model_version` | str | iguais nas 20 linhas |
| `params` | dict | `temperature, seed, max_output_tokens, thinking, language` |
| `usage` | dict | tokens de entrada, saída e "thinking" |
| `timestamp` | str | ISO-8601 |

---

## 3. Checklist de Validação Final

### Código-fonte
- [ ] `src/`, `scripts/`, `tests/`, `config/config.yaml` e `requirements.txt` com versões fixadas presentes no zip do Moodle (não só um link).
- [ ] `pytest` passa: data_loader, bm25 (inclui não-varredura), spell, dense, eval × ir_measures (tolerância 1e-9), rrf, rerank (invariância das posições 51–100), results_schema, split_guard.
- [ ] O índice invertido não importa nenhuma biblioteca de indexação textual (revisão de imports em `inverted_index.py` e `bm25.py`).
- [ ] Nenhum arquivo grande no zip: `*.npy`, `*.parquet`, modelos, `nfcorpus/`, venv.
- [ ] Nenhuma credencial: `grep -ri "api_key\|AIza" .` sem ocorrências fora do `.env.example`.

### Arquivos CSV (`results/`)
- [ ] Os 7 arquivos existem com as colunas mínimas da §2.
- [ ] Arquivos de ranking: ≤ 100 linhas por consulta; ranks contíguos a partir de 1; `relevance` coerente com `qrels/test.tsv`.
- [ ] `results_semantic.csv` tem 32.300 linhas.
- [ ] Consultas lexicais com menos de 100 linhas listadas em `lexical_candidates_test.csv` e declaradas no relatório.
- [ ] `metrics_per_query.csv`: 1.292 linhas; `mean(AP10)` == MAP@10 do relatório (4 casas) para cada `model`.
- [ ] `typo_experiment.csv`: 10 linhas; 4 tipos; `NDCG10_original > 0`; `termo_erro` ∉ dicionário; `recuperacao` conforme a fórmula.
- [ ] `typo_experiment.csv` (extensão semântica): colunas `NDCG10_sem_*` preenchidas nas 10 linhas; `NDCG10_sem_original` igual ao `metrics_per_query.csv`; `robustez_lex_sem.csv` gerado e citado no relatório.
- [ ] `rag_experiment.csv`: 20 linhas; as 10 consultas oficiais; parâmetros iguais em A e B; rastreabilidade `N/A` em A; justificativas não vazias.

### README (campos obrigatórios — Entrega 2)
- [ ] Integrantes (nome completo e Nº USP).
- [ ] Instalação e execução (passo a passo, CPU).
- [ ] Versão do Python e principais bibliotecas com versões.
- [ ] Modelo lexical (BM25; k1, b).
- [ ] Modelo de embeddings (D4, com revisão).
- [ ] Modelo de reranking (D5, com revisão).
- [ ] LLM (D2: ID, versão, parâmetros).
- [ ] Hardware (CPU, threads, RAM).
- [ ] Ordem de execução: `run_dev.py` → `run_test.py` → `run_typo.py` → `run_rag.py` (+ avaliação humana).
- [ ] Como baixar o NFCorpus e regenerar índice e embeddings; como configurar `GEMINI_API_KEY`.

### Relatório PDF (seções obrigatórias — Entrega 4)
- [ ] a. Cabeçalho: título, autores (nome completo e Nº USP), filiação, email.
- [ ] b. Introdução.
- [ ] c. Metodologia: pré-processamento, índice invertido, busca lexical, tolerância a erros, semântica, híbrida, reranking, RAG.
- [ ] d. Avaliação: dataset, splits, qrels, métricas (com convenções), modelos, parâmetros, ambiente, protocolo; **leitura adotada da restrição C4** (D3).
- [ ] e. Resultados e Análise: lexical × semântico (3 + 3 casos), tolerância a erros, híbrido, reranking (4 casos), tempos, RAG, análise de falhas (3 casos).
- [ ] f. Considerações finais.
- [ ] g. Uso de ferramentas de IA.
- [ ] Estatísticas do índice (vocabulário, postings, tempo de construção, tempo médio por consulta).
- [ ] Os 9 itens do RF-5; `max_length` e truncamento do cross-encoder; itens da LLM (RF-11), incluindo custo (US$ 0, free tier) e uso dos dados.
- [ ] Toda afirmação numérica das análises aponta para um CSV do TDS-12.
- [ ] ≤ 20 páginas sem as referências.

### Pré-processamento e busca lexical
- [ ] Nenhum token do índice contém pontuação (`re.fullmatch(r"[a-z0-9]+", t)` para todo termo de `index.terms()`).
- [ ] Consultas lexicais com menos de 100 candidatos: contagem em `lexical_candidates_test.csv`, **sem completar** (TDS-DEC-4), e declaração explícita no relatório, inclusive "nenhuma consulta" se for o caso.

### Experimentos (condições, seeds, reprodutibilidade)
- [ ] `rag_queries.json` == lista oficial do Req. 12 (`assert` passa).
- [ ] `validation_dev.csv` (semântico e reranking), `rrf_k_dev.csv`, `spell_grid_dev.csv` e, se usados, `preprocessing_dev.csv` e `bm25_grid_dev.csv` com timestamps anteriores ao `test_run.lock`.
- [ ] `config.yaml` com `frozen: true`; `config_hash` igual ao do `test_run.lock`.
- [ ] Seeds fixas: sorteio do RAG (42), gerador de erros (42 + hash do `query_id`), embaralhamento da avaliação cega (42), LLM (`seed` 42, se aceito).
- [ ] Revisões dos modelos do HF registradas (`manifest.json`, `rerank_manifest.json`).
- [ ] `responses.jsonl` gerado uma única vez (hash no `freeze_log.json`); `modelVersion` igual nas 20 linhas.
- [ ] Experimento de correção isolado: as runs principais usam as consultas originais (verificar que nenhum `.trec` principal depende do `spell.py`).

---

## 4. Decisões de implementação tomadas neste TDS

| ID | Decisão | Motivo |
|---|---|---|
| TDS-DEC-1 | Desempate por `doc_id` crescente | Determinismo; já usado no BM25 (D1) |
| TDS-DEC-2 | `doc_text = title + " " + text` | Restrição C18; representação única |
| TDS-DEC-3 | Layout do repositório | Rastreabilidade C-N ↔ módulo |
| TDS-DEC-4 | Lexical com menos de 100 candidatos: não completar | Não inventar documentos com score 0; RD §6.1 (ambiguidade) |
| TDS-DEC-5 | R = 0 → métricas 0 | Alinhamento com trec_eval/ir_measures |
| TDS-DEC-6 | Score misto no CSV de reranking + `score_source` | RF-15 exige `score`; RD marca a ambiguidade |
| TDS-DEC-7 | Contexto sem truncamento (limite de segurança de 8K tokens) | ~2K tokens estimados |
| TDS-DEC-8 | Distância OSA (transposição = 1) | RF-4 inclui "troca da ordem" como tipo de erro |
| TDS-DEC-9 | Seleção das consultas e dos erros por sorteio com seed | Evitar escolher casos favoráveis (restrição C3) |
| TDS-DEC-10 | Ajuste do corretor no dev com erros sintéticos e grade fixa | RF-4: "Limiares … poderão ser ajustados no conjunto dev" |
| TDS-DEC-11 | Regras determinísticas de seleção de casos (RF-7/10/14) | Reprodutibilidade; evita escolha a dedo |
| TDS-DEC-12 | Incluir a extensão semântica opcional do RF-4 (mesmas consultas com erro e corrigidas no bi-encoder) | Custo quase nulo (embeddings já calculados); enriquece a discussão de robustez lexical × semântica (D6) |
| TDS-DEC-13 | Tokenização por regex `[a-z0-9]+` sobre texto em minúsculas, descartando toda pontuação | Evita tokens com pontuação colada (erro do T1); mesma regra para documentos, consultas e dicionário do RF-4 |

---

## 5. Limpeza antes da entrega

Os documentos de especificação existem para apoiar o desenvolvimento com IA. **Depois que o código estiver pronto e os resultados gerados**, e antes da entrega, remover do repositório (decisão dos autores):
- `docs/t2_requisitos.md`, `docs/t2_arquitetura.md`, `docs/t2_tds.md`, `docs/t2_decisoes.md`, `docs/t2_plano_execucao.md`, `docs/t2_busca_lexical_bm25_contexto.md`;
- `AGENTS.md`, `CLAUDE.md`, `.docs/`.

Ficam: o hook (`.githooks/`), a Action (`.github/`), o código, os testes, os resultados e o README, revisado para não citar arquivos removidos. As justificativas que o relatório precisa (D1–D6, TDS-DEC-n) devem ser transcritas para o relatório **antes** da remoção.
