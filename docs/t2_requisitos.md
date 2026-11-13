# Documento de Requisitos — Trabalho Prático 2 (SCC0282, RI 2026)

> **Fonte primária:** `Trabalho_Pratico_2_RI_2026.pdf` (Prof. Marcelo Manzato; divulgação 02/10/2026; 9 páginas, incluindo o roadmap da p. 9).
> **Escopo:** descreve **o quê** deve ser feito. Não toma decisões de design, não escolhe modelos e não propõe implementações.
> **Convenções:**
> - `Req. N` = requisito N do enunciado. As demais origens citam a seção do enunciado (`Objetivo`, `Metodologia`, `Análise crítica`, `Uso de IA`, `Entrega`, `Defesa`, `Observações finais`, `Roadmap p. 9`).
> - `[NÃO ESPECIFICADO — decisão do grupo]` = o enunciado não define.
> - `[AMBIGUIDADE: …]` = há mais de uma leitura; segue a interpretação mais conservadora (**→ Conservador:**).
> - `[EXTERNO]` = informação que **não** está no enunciado (formato BEIR, preços, specs de modelos). Deve ser conferida antes de ir para o relatório.
>
> Elaborado em 11/11/2026. Preços e catálogos de modelos (seção 7) mudam com frequência; confira-os no dia da execução.

---

## 0. Metadados do trabalho

| Item | Valor | Origem |
|---|---|---|
| Disciplina | SCC0282 Recuperação de Informação — 2º Sem/2026, ICMC-USP | Cabeçalho |
| Divulgação | 02/10/2026 | Cabeçalho |
| **Prazo** | **19/11/2026, 23:59** | Cabeçalho |
| Grupos | Preferencialmente a mesma configuração do T1; **até 2 pessoas** | Cabeçalho |
| Objetivo | Sistema de RI que combine busca lexical e semântica, com tolerância a erros, fusão de rankings, reranking e RAG; investigação experimental das diferenças lexical × semântico e da interação entre componentes | Objetivo |
| Pipeline geral | documentos → pré-processamento e indexação → recuperação lexical + recuperação semântica → busca híbrida → reranking → seleção de contexto → LLM → resposta | Objetivo |
| Exigência transversal | Implementar não basta: é preciso fazer avaliação quantitativa, análise por consulta e investigação de casos de sucesso e falha com evidências dos próprios experimentos | Objetivo |
| Mapeamento do roadmap | Etapa *n* da figura = requisito *n−1* (etapa 1 = preparação do dataset) | Objetivo, Roadmap p. 9 |

---

## 1. Restrições Críticas

Restrições explícitas cuja violação compromete a validade do trabalho ou de um componente avaliado.

| # | Restrição | Origem | Consequência da violação |
|---|---|---|---|
| C1 | O **índice invertido** deve ser implementado **do zero** pelo grupo. São permitidas estruturas genéricas da linguagem ou de bibliotecas auxiliares, **não** mecanismos prontos que construam e consultem o índice textual. | Metodologia; Req. 2 | Componente "indexação e recuperação lexical" (≈10%) inválido; critério 1 das Observações finais não atendido. |
| C2 | É proibido percorrer **todos os documentos** da coleção a cada consulta para calcular o ranking lexical. | Req. 2 | Igual a C1. Obs.: a restrição vale **apenas** para o índice invertido lexical; a busca semântica pode ser exaustiva (Req. 5). |
| C3 | Os **qrels do test** não podem ser usados para modificar consultas, escolher parâmetros manualmente, alterar rankings ou favorecer exemplos específicos. | Metodologia | Vazamento de dados: resultados do test deixam de ser avaliação final; afeta reprodutibilidade e análise (20%). |
| C4 | A escolha de **modelo de embeddings**, **modelo de reranking**, **etapas de pré-processamento** e **parâmetros do modelo lexical** (ex.: k1 e b, se diferentes dos usuais) deve ser feita com base no **dev**, nunca comparando alternativas no test. | Metodologia | Igual a C3. |
| C5 | O **dev** é o único split permitido para ajustes; o **test** fica reservado à avaliação final. | Metodologia | Igual a C3. |
| C6 | Um **único** modelo lexical (Vetorial TF-IDF + cosseno **ou** BM25) como baseline, e o **mesmo** modelo em todos os experimentos seguintes. | Req. 3 | Comparações inválidas (mais de um componente variando ao mesmo tempo). |
| C7 | O **k do RRF** deve ser escolhido no **dev** (maior NDCG@10 médio, avaliando pelo menos {10, 60, 100}) e **mantido** no test. | Req. 8 | Vazamento de dados no componente híbrido (≈10%). |
| C8 | A **correção ortográfica** é um experimento separado: os demais experimentos usam as **consultas originais, sem correção**. | Req. 4 | Contaminação de todos os resultados posteriores. |
| C9 | As **10 consultas do RAG** são fixas (seed 42) e iguais para todos os grupos: PLAIN-68, PLAIN-2450, PLAIN-153, PLAIN-2850, PLAIN-2750, PLAIN-270, PLAIN-2470, PLAIN-2430, PLAIN-238, PLAIN-3342. | Req. 12 | Protocolo do RAG (≈20%) não atendido. |
| C10 | Condições A e B do RAG com a **mesma LLM** e os **mesmos parâmetros**; respostas no **mesmo idioma** nas duas condições. | Req. 12 | Comparação A × B inválida. |
| C11 | Cada item do `corpus.jsonl` é **uma unidade de contexto** no experimento obrigatório do RAG (chunking só como opcional, preservando o `doc_id`). | Req. 11 | Perde-se a correspondência com os qrels; Recall@5/P@5/Hit@5 deixam de ser calculáveis. |
| C12 | Resultados devem ser entregues em **arquivos CSV independentes** dos notebooks. Resultados só em saída de célula **não contam** como entregues. | Req. 15; Entrega 3 | Arquivos de resultado considerados não entregues. |
| C13 | Código entregue **no Moodle**: link para repositório **não substitui** o arquivo. | Entrega 1 | Código considerado não entregue. |
| C14 | **Não** incluir arquivos grandes regeneráveis (dataset, embeddings, modelos, índice serializado, ambientes virtuais). | Entrega 1 | Entrega fora da especificação. |
| C15 | **Não** incluir chaves de API, senhas ou credenciais. | Entrega 3 | Entrega fora da especificação (e risco de segurança). |
| C16 | Relatório em PDF com **até 20 páginas** (só as referências ficam fora do limite); anexos não podem ser usados para contornar o limite. | Entrega 4 | Entrega fora da especificação. |
| C17 | Usar o **NFCorpus (BEIR)** com os arquivos padronizados (`corpus.jsonl`, `queries.jsonl`, qrels). | Metodologia | Resultados não comparáveis com os dos outros grupos. |
| C18 | Representação do documento = **`title` + `text` concatenados** em **todos** os sistemas (lexical, semântico, reranking), **salvo justificativa** no relatório. | Metodologia | Comparações entre sistemas inconsistentes. |
| C19 | Afirmações sobre frequência de termos, tamanho de documentos, posição no ranking, similaridade ou custo só valem com **evidência dos próprios experimentos**. | Req. 7; Análise crítica | Análise crítica (parte dos 20%) penalizada. |
| C20 | Scores de modelos diferentes **não** devem ser comparados diretamente sem justificativa ou normalização. | Análise crítica | Idem C19. |
| C21 | As comparações devem alterar **apenas** o componente investigado. | Análise crítica | Idem C19. |
| C22 | Todos os integrantes devem compreender integralmente o código, as decisões e as técnicas. | Metodologia; Uso de IA; Defesa | Defesa individual (≈10%). |

---

## 2. Dataset e Splits

### 2.1 Corpus

| Item | Valor | Origem |
|---|---|---|
| Nome | NFCorpus, versão do benchmark **BEIR** | Metodologia |
| Nº de documentos | **3.633** | Req. 5 |
| Campos usados | `title` e `text` (concatenados, salvo justificativa) | Metodologia |
| Formato dos arquivos | `corpus.jsonl`, `queries.jsonl`, `qrels/{train,dev,test}.tsv` | Metodologia |
| Relevantes por consulta | Em média, **cerca de 38** documentos relevantes | Req. 13 |
| Esquema de campos `[EXTERNO]` | Formato BEIR: corpus `{"_id", "title", "text", "metadata"}`; queries `{"_id", "text", "metadata"}`; qrels TSV com cabeçalho `query-id  corpus-id  score` | Documentação BEIR — conferir localmente |
| Escala dos graus de relevância | `[NÃO ESPECIFICADO no enunciado]`. O enunciado só diz "grau > 0 = relevante" e "preservar os graus no NDCG". Conferir a distribuição de `score` em `qrels/test.tsv`. | — |

### 2.2 Arquivos obrigatórios e funções

| Arquivo | Função | Origem |
|---|---|---|
| `corpus.jsonl` | Documentos (unidade de indexação e de contexto do RAG) | Metodologia; Req. 11 |
| `queries.jsonl` | Textos de **todas** as consultas de todos os splits (**3.237** no total) | Metodologia |
| `qrels/dev.tsv` | Define as consultas do dev; uso em avaliação **e** ajuste das configurações permitidas | Metodologia |
| `qrels/test.tsv` | Define as consultas do test (**323**); uso **apenas** em avaliação | Metodologia |
| `qrels/train.tsv` | Incluído no arquivo oficial. **Nenhum uso previsto no enunciado** `[NÃO ESPECIFICADO — decisão do grupo]` | Metodologia |

### 2.3 Splits

| Split | Papel | Pode | Não pode | Nº de consultas |
|---|---|---|---|---|
| **train** | Não citado além de existir no arquivo oficial | `[NÃO ESPECIFICADO]` | `[NÃO ESPECIFICADO]` | `[NÃO ESPECIFICADO no enunciado]` `[EXTERNO: 2.590 no BEIR — conferir]` |
| **dev** | Ajuste de parâmetros | Escolher modelo de embeddings, modelo de reranking, pré-processamento, k1/b, k do RRF, limiares da correção ortográfica | — | `[NÃO ESPECIFICADO no enunciado]` `[EXTERNO: 324 no BEIR — conferir]` |
| **test** | Avaliação final | Avaliar (Req. 6–10, 13); fornecer as 10 consultas da correção ortográfica (Req. 4) e as 10 do RAG (Req. 12) | Modificar consultas, escolher parâmetros, alterar rankings, favorecer exemplos, comparar alternativas de modelo/pré-processamento | **323** |

> Regra de pertencimento (Metodologia): as consultas de um split são **as que têm julgamentos no qrels daquele split**, e não todas as do `queries.jsonl`.
> Conferência de consistência `[EXTERNO]`: 2.590 + 324 + 323 = 3.237, que é o total informado no enunciado.

### 2.4 As 10 consultas fixas do RAG (Req. 12)

**Universo:** as **42** consultas do test cujo texto termina em `"?"`. **Sorteio:** `random.Random(42).sample(ids, 10)`.

| # (ordem do sorteio) | query_id |
|---|---|
| 1 | PLAIN-68 |
| 2 | PLAIN-2450 |
| 3 | PLAIN-153 |
| 4 | PLAIN-2850 |
| 5 | PLAIN-2750 |
| 6 | PLAIN-270 |
| 7 | PLAIN-2470 |
| 8 | PLAIN-2430 |
| 9 | PLAIN-238 |
| 10 | PLAIN-3342 |

Código de referência (transcrito do enunciado):

```python
import random
# test_qrels: consultas com julgamentos em qrels/test.tsv; queries: textos de queries.jsonl
ids = [q for q in sorted(test_qrels.keys()) if queries[q].strip().endswith("?")]
queries_rag = random.Random(42).sample(ids, 10)
```

O enunciado manda conferir se o código produz a mesma lista. Para a lista sair idêntica:
- `sorted()` ordena os ids como **strings** (ordem lexicográfica: `PLAIN-1008` vem antes de `PLAIN-102`).
- `len(ids)` deve dar **42** (assert recomendado).
- A **ordem** de saída do `sample` faz parte da reprodução.

> ⚠️ Não foi possível conferir a lista neste ambiente (download do NFCorpus bloqueado). **Pendência:** rodar o snippet localmente com `assert len(ids) == 42` e `assert queries_rag == [...]`.

### 2.5 URLs de download (Metodologia)

| Recurso | URL |
|---|---|
| Arquivo oficial BEIR (corpus + queries + qrels train/dev/test) — **recomendado** | https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/nfcorpus.zip |
| Lista de datasets BEIR | https://github.com/beir-cellar/beir/wiki/Datasets-available |
| NFCorpus no Hugging Face | https://huggingface.co/datasets/BeIR/nfcorpus |
| Qrels no Hugging Face (repositório **separado**) | `BeIR/nfcorpus-qrels` |

---

## 3. Requisitos Funcionais

> **Reaproveitamento do T1:** o enunciado autoriza reuso **explicitamente apenas no Req. 3** ("Código desenvolvido no Trabalho Prático 1 poderá ser reutilizado e adaptado"). Nos demais, o reuso não é mencionado. O T1 usou Cranfield, MAP sem corte e binarização "grau ≥ 1" (`project_scope.md`); o T2 usa NFCorpus, MAP@10 e "grau > 0".

### RF-1 — Pré-processamento (Req. 1; Roadmap etapa 2)
- **Descrição:** leitura dos documentos, tokenização, normalização, remoção de stopwords, stemming, lematização e demais etapas de representação de documentos e consultas.
- **Restrições específicas:**
  - Bibliotecas externas são permitidas (Req. 1; Metodologia).
  - O relatório deve descrever as etapas usadas, as bibliotecas e as principais decisões de representação.
  - A escolha das etapas é feita no **dev** (C4).
  - Documento = `title` + `text` (C18).
- **Parâmetros obrigatórios:** nenhum.
- **Parâmetros livres:** quais etapas aplicar e quais bibliotecas usar.
- **[AMBIGUIDADE: obrigatoriedade de stemming/lematização]** O Req. 1 lista stemming e lematização entre as etapas possíveis; o roadmap marca "Stemming/lematização (opcional)". → **Conservador:** tratar como opcional, mas justificar a escolha com experimento no dev.
- **[AMBIGUIDADE: as consultas devem passar pelo mesmo pipeline?]** Não está dito. → **Conservador:** mesmo pipeline para documentos e consultas no sistema lexical, documentado.
- **Reaproveitável do T1?** Não mencionado no enunciado `[NÃO ESPECIFICADO]` (a permissão textual é só do Req. 3).
- **Dependências:** dataset carregado (seção 2).

### RF-2 — Índice invertido (Req. 2; Roadmap etapa 3)
- **Descrição:** índice invertido próprio (dicionário de termos + posting lists) usado pela recuperação lexical.
- **Restrições específicas:** C1 e C2.
- **Conteúdo mínimo:** as informações exigidas pelo modelo lexical escolhido, incluindo, quando aplicável, **TF**, **DF**, **tamanho dos documentos** e **estatísticas globais**.
- **Estatísticas obrigatórias a registrar:**
  1. nº de termos do vocabulário;
  2. nº total de postings;
  3. tempo de construção do índice;
  4. tempo médio de processamento das consultas.
- **Parâmetros obrigatórios:** nenhum.
- **Parâmetros livres:** estruturas de dados do dicionário e das posting lists.
- **Reaproveitável do T1?** Não mencionado `[NÃO ESPECIFICADO]`. A exigência "do zero" impede reaproveitar mecanismos prontos.
- **Dependências:** RF-1.
- **Status no projeto:** ver `docs/t2_busca_lexical_bm25_contexto.md`, onde o índice foi refeito do zero e o T1 não tinha índice invertido.

### RF-3 — Busca lexical (Req. 3; Roadmap etapa 4)
- **Descrição:** baseline lexical único sobre o índice do RF-2.
- **Restrições específicas:**
  - Escolher **uma** alternativa: Modelo Vetorial (TF-IDF + cosseno) **ou** BM25 (C6).
  - Usar o mesmo modelo em todos os experimentos seguintes.
  - Informar e justificar brevemente a escolha no relatório.
  - Se k1/b forem diferentes dos valores usuais, escolhê-los no dev (C4).
- **Parâmetros obrigatórios:** nenhum valor fixado.
- **Parâmetros livres:** modelo (entre os dois) e seus parâmetros (ajuste só no dev).
- **[NÃO ESPECIFICADO — decisão do grupo]** O que são os "valores usuais" de k1 e b.
- **Reaproveitável do T1?** **Sim** — "Código desenvolvido no Trabalho Prático 1 poderá ser reutilizado e adaptado" (Req. 3).
- **Dependências:** RF-2.
- **Status no projeto:** BM25 escolhido; ver `docs/t2_busca_lexical_bm25_contexto.md`.

### RF-4 — Tolerância a erros / correção ortográfica (Req. 4; Roadmap etapa 5)
- **Descrição:** correção de termos desconhecidos em três passos:
  1. **k-gramas** para gerar candidatos;
  2. **coeficiente de Jaccard** para filtrar;
  3. **distância de edição** para escolher as alternativas mais próximas.
- **Restrições específicas (padronização obrigatória):**
  - (i) Dicionário da correção = termos da coleção **antes** do stemming/lematização.
  - (ii) Cada erro introduzido deve gerar um termo **inexistente** no dicionário; erros que formam outra palavra válida não podem ser usados.
  - (iii) Usar consultas cujo **NDCG@10 original > 0**.
  - (iv) Avaliação obrigatória no sistema **lexical**, com **NDCG@10**.
  - **10 consultas do test**, com erros de tipos diferentes: inserção, remoção, substituição ou troca da ordem de caracteres.
  - Três condições por consulta: **original**, **com erro** e **corrigida**.
  - Experimento separado; não contamina os demais (C8).
- **Fórmula obrigatória (por consulta):**
  - `recuperação = (NDCG@10_corrigida − NDCG@10_erro) / (NDCG@10_original − NDCG@10_erro)`, quando o denominador é **> 0**;
  - caso contrário, registrar que **"o erro não reduziu o desempenho"**.
- **Parâmetros obrigatórios:** 10 consultas; métrica NDCG@10; sistema lexical; três condições; 4 tipos de erro como universo.
- **Parâmetros livres:**
  - k dos k-gramas, limiar de Jaccard e critério de corte da distância de edição — **ajustáveis no dev** ("poderão");
  - quais consultas e quais erros, dentro de (ii) e (iii);
  - aplicar o experimento também ao sistema semântico (**opcional**).
- **[AMBIGUIDADE: nº de erros por consulta]** Não especificado. → **Conservador:** um erro por consulta, controlado, e registrar `tipo_erro` por linha.
- **[AMBIGUIDADE: cobertura dos tipos]** "envolvendo diferentes tipos" não diz se os 4 tipos são obrigatórios. → **Conservador:** cobrir os 4 tipos no conjunto das 10 consultas.
- **[AMBIGUIDADE: variante da distância de edição]** Não especificada. Na Levenshtein, a troca de dois caracteres adjacentes custa 2; na Damerau-Levenshtein, custa 1. → **Conservador:** declarar a variante no relatório e analisar o efeito sobre os erros de troca.
- **[AMBIGUIDADE: "termos da coleção"]** Não diz se inclui termos das consultas. → **Conservador:** apenas termos do corpus (`title` + `text`), com a mesma tokenização/normalização anterior ao stemming.
- **[AMBIGUIDADE: agregação da recuperação]** Não especificada (média das razões, mediana, ganho agregado). A razão pode sair < 0 ou > 1. → **Conservador:** reportar os valores por consulta sem truncar; se houver agregado, declarar a fórmula e excluir os casos com denominador ≤ 0.
- **[AMBIGUIDADE: seleção no test × C3]** Escolher consultas do test usando NDCG@10 > 0 exige os qrels do test, mas o critério (iii) é imposto pelo próprio enunciado. → **Conservador:** usar os qrels do test **só** para o filtro (iii) e escolher as consultas por regra documentada (ex.: sorteio com seed), nunca pelo ganho da correção.
- **Reaproveitável do T1?** Não mencionado `[NÃO ESPECIFICADO]`.
- **Dependências:** RF-1 (vocabulário pré-stemming), RF-3 (sistema lexical), RF-6 (NDCG@10 implementado e validado).

### RF-5 — Recuperação semântica (Req. 5; Roadmap etapa 6)
- **Descrição:** recuperação densa com modelo pré-treinado **bi-encoder**.
  - Embeddings dos documentos **pré-calculados e armazenados**.
  - Para cada consulta: gerar o embedding e calcular a similaridade com os documentos.
- **Restrições específicas:**
  - Não é preciso implementar, treinar ou fazer fine-tuning de modelo neural.
  - Bibliotecas e modelos pré-treinados são permitidos.
  - Busca exaustiva permitida (3.633 documentos); C2 não se aplica aqui.
  - Escolha do modelo no **dev** (C4).
  - Documento = `title` + `text` (C18).
- **Itens obrigatórios no relatório:**
  1. nome e versão do modelo;
  2. biblioteca usada;
  3. dimensionalidade dos embeddings;
  4. função de similaridade;
  5. normalização aplicada (quando houver);
  6. tempo para gerar os embeddings dos documentos;
  7. tempo médio por consulta;
  8. comprimento máximo de entrada (tokens);
  9. se houve truncamento dos documentos.
- **Parâmetros obrigatórios:** arquitetura bi-encoder.
- **Parâmetros livres:** modelo ("A escolha do modelo é livre"), biblioteca, função de similaridade, normalização.
- **Sugestões do professor:** biblioteca Sentence Transformers; exemplos `all-MiniLM-L6-v2` e `multi-qa-MiniLM-L6-cos-v1`.
- **Links de apoio (Req. 5):**
  - Quickstart: https://www.sbert.net/docs/quickstart.html
  - Busca semântica: https://www.sbert.net/examples/sentence_transformer/applications/semantic-search/README.html
  - Modelos pré-treinados: https://www.sbert.net/docs/sentence_transformer/pretrained_models.html
  - Exemplos: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2 e https://huggingface.co/sentence-transformers/multi-qa-MiniLM-L6-cos-v1
- **Reaproveitável do T1?** Não (componente inexistente no T1).
- **Dependências:** dataset (seção 2). Independe de RF-1–RF-3.

### RF-6 — Avaliação da recuperação (Req. 6; Roadmap etapa 7)
- **Descrição:** avaliar os sistemas lexical e semântico sobre **as mesmas consultas do test**.
- **Restrições específicas:**
  - Métricas obrigatórias: **P@10, R@10, MAP@10, NDCG@10**, com as convenções da seção 4.
  - Guardar **todas** as métricas **por consulta**, além das médias globais.
- **Parâmetros obrigatórios:** corte 10; relevância binária "grau > 0" para P/R/MAP; graus originais para o NDCG; ganho linear; ideal calculado a partir de todos os julgamentos.
- **Parâmetros livres:** implementação própria ou biblioteca (cálculo de métricas com biblioteca é permitido na Metodologia).
- **Ferramenta de verificação recomendada:** `ir_measures`.
- **[AMBIGUIDADE: "mesmas consultas do conjunto test"]** → **Conservador:** todas as 323 consultas com julgamento em `qrels/test.tsv`.
- **Reaproveitável do T1?** Não mencionado `[NÃO ESPECIFICADO]`. As convenções do T2 (MAP@10, "grau > 0", ganho linear) diferem das do T1.
- **Dependências:** RF-3, RF-5.

### RF-7 — Comparação lexical × semântica (Req. 7; Roadmap etapa 8)
- **Descrição:** comparação quantitativa global + análise por consulta pela **diferença de NDCG@10**.
- **Restrições específicas:**
  - Identificar as **3** consultas com maior vantagem do **semântico** e as **3** com maior vantagem do **lexical**.
  - Para cada uma das 6, apresentar:
    1. o texto da consulta;
    2. o **Top-5 de cada método**;
    3. quais documentos são relevantes;
    4. os scores ou similaridades;
    5. uma análise das razões da diferença.
  - Buscar evidências de: correspondência lexical, termos técnicos, sinonímia, paráfrase, variação de vocabulário e ambiguidade.
  - Afirmações como "o documento era muito longo", "o termo era mais frequente" ou "o modelo favoreceu…" só com valores observados nos dados (C19).
- **Parâmetros obrigatórios:** métrica de seleção NDCG@10; 3 + 3 consultas; Top-5.
- **Parâmetros livres:** forma de apresentação.
- **[NÃO ESPECIFICADO — decisão do grupo]** Desempate entre consultas com a mesma diferença de NDCG@10.
- **Reaproveitável do T1?** Não mencionado `[NÃO ESPECIFICADO]`.
- **Dependências:** RF-6.

### RF-8 — Busca híbrida (Req. 8; Roadmap etapa 9)
- **Descrição:** fusão dos rankings lexical e semântico por **Reciprocal Rank Fusion (RRF)**.
- **Fórmula obrigatória:** `RRF(d) = Σ_s 1 / (k + rank_s(d))`
  - `rank_s(d)` é a posição de *d* no sistema *s*, **começando em 1**;
  - documentos ausentes de um ranking **não recebem contribuição** daquele sistema.
- **Restrições específicas:**
  - Fundir os **100 primeiros** documentos de cada sistema.
  - Ajustar k no **dev**, avaliando **pelo menos k ∈ {10, 60, 100}**, e escolher o de maior **NDCG@10 médio no dev**.
  - Manter o k escolhido no test (C7).
  - Comparar **Lexical, Semântico e Híbrido** com P@10, R@10, MAP@10 e NDCG@10.
- **Parâmetros obrigatórios:** RRF; top-100 por sistema; grid mínimo {10, 60, 100}; critério NDCG@10 médio no dev.
- **Parâmetros livres:** valores adicionais de k.
- **Motivação declarada:** o RRF evita comparar diretamente scores de modelos com escalas diferentes (Req. 8).
- **Observação:** a união de dois top-100 pode ter até 200 documentos; `results_hybrid.csv` exige os 100 primeiros (Req. 15).
- **[NÃO ESPECIFICADO — decisão do grupo]** Desempate de documentos com o mesmo score RRF, e de valores de k com o mesmo NDCG@10 no dev.
- **Reaproveitável do T1?** Não (componente inexistente no T1).
- **Dependências:** RF-3, RF-5, RF-6 (NDCG@10 no dev).

### RF-9 — Reranking (Req. 9; Roadmap etapa 10)
- **Descrição:** reordenar os **50 primeiros** do ranking híbrido com um modelo que considere consulta e documento em conjunto (preferencialmente **cross-encoder** pré-treinado ou reranking neural equivalente).
- **Restrições específicas:**
  - O recuperador inicial é o sistema **híbrido**.
  - Documentos abaixo da posição 50 **mantêm a ordem relativa** do híbrido, para que exista um ranking completo para as métricas.
  - Escolha do modelo no **dev** (C4), informada e justificada.
  - Comparar **Lexical, Semântico, Híbrido e Híbrido + reranking** (P@10, R@10, MAP@10, NDCG@10).
  - Registrar o **tempo médio das etapas**.
  - Informar o comprimento máximo de entrada do cross-encoder e se houve truncamento.
- **Parâmetros obrigatórios:** profundidade 50; recuperador = híbrido.
- **Parâmetros livres:** modelo de reranking.
- **Sugestões do professor:** `CrossEncoder` do Sentence Transformers; exemplo `cross-encoder/ms-marco-MiniLM-L6-v2`.
- **Links de apoio (Req. 9):**
  - Uso: https://www.sbert.net/docs/cross_encoder/usage/usage.html
  - Modelos: https://www.sbert.net/docs/cross_encoder/pretrained_models.html
  - Exemplo: https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2
- **[AMBIGUIDADE: "tempo médio das etapas"]** → **Conservador:** tempo médio por consulta de cada etapa (lexical, semântica, fusão, reranking). O roadmap fala em "custo/tempo".
- **[NÃO ESPECIFICADO — decisão do grupo]** Que valor de `score` vai para o CSV nas posições > 50 (o score do reranker não existe para elas).
- **Reaproveitável do T1?** Não (componente inexistente no T1).
- **Dependências:** RF-8.

### RF-10 — Análise do reranking (Req. 10; Roadmap etapa 11)
- **Descrição:** identificar quatro casos:
  1. e 2. os **2** com maior ganho de NDCG@10;
  3. **1** com pouca ou nenhuma alteração;
  4. **1** em que o NDCG@10 **piorou**.
- **Restrições específicas:** para cada caso, mostrar os principais documentos que mudaram de posição e discutir a provável razão (C19).
- **Parâmetros obrigatórios:** métrica NDCG@10; 2 + 1 + 1 casos.
- **Parâmetros livres:** critério operacional de "pouca ou nenhuma alteração" `[NÃO ESPECIFICADO]`.
- **Reaproveitável do T1?** Não.
- **Dependências:** RF-9, RF-6.

### RF-11 — Retrieval-Augmented Generation (Req. 11; Roadmap etapa 12)
- **Descrição:** pipeline obrigatório consulta → recuperação híbrida → reranking → seleção de contexto → LLM → resposta.
- **Restrições específicas:**
  - Cada item do `corpus.jsonl` é uma unidade de contexto (C11).
  - Chunking adicional é **opcional**, desde que preserve a relação com o `doc_id`.
- **Itens obrigatórios no relatório:**
  - modelo e versão;
  - forma de execução (local ou serviço externo);
  - temperatura;
  - nº máximo de tokens;
  - demais parâmetros relevantes;
  - **custo aproximado**, se houver cobrança por API;
  - **hardware**, se a execução for local.
- **Parâmetros livres:** LLM (aberta local/Colab — ex.: Qwen, Llama, Gemma, Mistral — ou comercial via API). Serviço pago **não** é obrigatório, e modelo maior ou comercial **não** garante nota melhor.
- **Reaproveitável do T1?** Não.
- **Dependências:** RF-9.

### RF-12 — Experimento sem RAG × com RAG (Req. 12; Roadmap etapa 13)
- **Descrição:** 10 consultas fixas (seção 2.4) × 2 condições:
  - **(A) sem RAG:** a LLM recebe **somente a pergunta**, sem acesso ao NFCorpus;
  - **(B) com RAG:** a LLM recebe a pergunta + os **Top-5 após o reranking**, com seus identificadores.
- **Restrições específicas:**
  - O prompt do RAG deve instruir o modelo a:
    1. responder com base no contexto fornecido;
    2. indicar as fontes usadas;
    3. informar quando não houver evidência suficiente.
  - Mesma LLM e mesmos parâmetros nas duas condições (C10).
  - Mesmo idioma nas duas condições, e o idioma deve ser informado.
- **Parâmetros obrigatórios:** 10 consultas fixas; Top-5 pós-reranking; condições A/B.
- **Parâmetros recomendados (não obrigatórios):** temperatura **0** (ou a menor possível); **seed fixa**, quando disponível.
- **Parâmetros livres:** idioma; texto exato dos prompts; nº máximo de tokens.
- **[NÃO ESPECIFICADO — decisão do grupo]** Prompt da condição A além de "somente a pergunta"; formato dos identificadores no contexto; truncamento do texto dos documentos no prompt.
- **Reaproveitável do T1?** Não.
- **Dependências:** RF-11.

### RF-13 — Avaliação do RAG (Req. 13; Roadmap etapa 14)
- **Descrição:** avaliação em duas partes.
  - **Recuperação:** Recall@5, Precision@5 e Hit@5 nas 10 consultas (seção 4).
  - **Geração:** avaliação humana estruturada com notas 0/1/2 em 4 critérios, cada nota com justificativa (seção 5.3).
- **Restrições específicas:**
  - Na condição A, registrar explicitamente que não houve contexto externo.
  - Fidelidade em A = comparar com os **mesmos Top-5** usados em B para aquela consulta.
  - Rastreabilidade em A = **"N/A"**, excluída da média de A; a ausência de fontes deve ser discutida qualitativamente.
  - Apresentar e discutir as médias das duas condições.
  - Objetivo declarado: investigar o efeito da recuperação na qualidade, na fidelidade e na disponibilidade de evidências, e **não** provar que o RAG é sempre melhor.
  - Recall@5 baixo é esperado (≈38 relevantes por consulta) e, sozinho, não indica falha.
- **Parâmetros recomendados:** avaliação **cega** (sem saber se a resposta é A ou B, ex.: embaralhando); dois avaliadores independentes e concordância, **quando possível**.
- **[AMBIGUIDADE: grupo de 1 pessoa]** A dupla avaliação é "quando possível". → **Conservador:** declarar a impossibilidade e manter o cegamento.
- **[NÃO ESPECIFICADO — decisão do grupo]** Medida de concordância (ex.: % de acordo, kappa de Cohen).
- **Divergência texto × roadmap:** o roadmap (etapa 14) cita só "Recall@5"; o texto exige **Recall@5, Precision@5 e Hit@5**. → Seguir o texto.
- **Reaproveitável do T1?** Não.
- **Dependências:** RF-12; RF-6 (convenções de relevância).

### RF-14 — Análise de falhas ponta a ponta (Req. 14; Roadmap etapa 15)
- **Descrição:** análise detalhada de **3** das 10 consultas do RAG:
  1. uma em que o RAG **melhorou claramente** a resposta;
  2. uma com **pouca diferença**;
  3. uma em que o RAG **não ajudou ou prejudicou**.
- **Restrições específicas:**
  - No caso 3, localizar o estágio do problema: recuperação lexical, recuperação semântica, fusão, reranking, seleção de contexto ou geração.
  - Verificar se a informação necessária **chegou ou não ao contexto**; "a LLM errou" não basta.
- **Parâmetros obrigatórios:** 3 casos entre as 10 consultas.
- **Parâmetros livres:** critério operacional de "melhorou claramente" e "pouca diferença" `[NÃO ESPECIFICADO]`.
- **Reaproveitável do T1?** Não.
- **Dependências:** RF-13 e os rankings intermediários de RF-3, RF-5, RF-8 e RF-9 para essas consultas.

### RF-15 — Arquivos padronizados de resultados (Req. 15; Roadmap etapa 16)
- **Descrição:** CSVs independentes dos notebooks (detalhes na seção 6).
- **Restrições específicas:** C12; os rankings trazem os **100 primeiros** documentos por consulta; a média de `AP10` deve bater com o MAP@10 reportado.
- **Parâmetros obrigatórios:** nomes e colunas mínimas (seção 6).
- **Parâmetros livres:** colunas adicionais.
- **Reaproveitável do T1?** Não mencionado.
- **Dependências:** RF-3, RF-4, RF-5, RF-6, RF-8, RF-9, RF-12, RF-13.

### 3.16 Grafo de dependências (resumo)

```
Dataset ─► RF-1 ─► RF-2 ─► RF-3 ─┬─► RF-6 ─► RF-7
   │                             │     │
   └──────────────► RF-5 ────────┘     ├─► RF-4 (também RF-1, RF-3)
                                       └─► RF-8 ─► RF-9 ─► RF-10
                                                     └─► RF-11 ─► RF-12 ─► RF-13 ─► RF-14
RF-15 ◄── RF-3, RF-4, RF-5, RF-6, RF-8, RF-9, RF-12, RF-13
```

---

## 4. Métricas de Avaliação

Notação: `rel(q)` = conjunto de documentos com grau **> 0** em qrels para a consulta *q*; `R = |rel(q)|`; `g(d)` = grau do documento nos qrels (0 se não julgado); `r_i` = documento na posição *i*.

| Métrica | Fórmula (conforme enunciado) | Relevância | Consultas | Origem |
|---|---|---|---|---|
| **P@10** | `|{r_1..r_10} ∩ rel(q)| / 10` | Binária: grau > 0 | 323 do test (RF-6/8/9); 10 do Req. 4 só NDCG | Req. 6 |
| **R@10** | `|{r_1..r_10} ∩ rel(q)| / R` — denominador = **total de relevantes nos qrels** | Binária: grau > 0 | Idem | Req. 6 |
| **AP@10** | `( Σ_{k≤10, r_k ∈ rel(q)} P@k ) / R` — denominador = **total de relevantes nos qrels** (convenção trec_eval/ir_measures) | Binária: grau > 0 | Idem | Req. 6 |
| **MAP@10** | Média de AP@10 sobre as consultas | Binária | Idem | Req. 6 |
| **NDCG@10** | `DCG@10 / IDCG@10`, com **ganho linear** (ganho = grau) e **IDCG calculado a partir de todos os julgamentos da consulta**, não só dos recuperados | **Graduada** (graus preservados) | Idem; dev para o k do RRF e limiares; 10 consultas do Req. 4 | Req. 6, 4, 8 |
| **Recall@5** | `|Top-5 ∩ rel(q)| / R` | Ver ambiguidade abaixo | As **10** consultas do RAG | Req. 13 |
| **Precision@5** | `|Top-5 ∩ rel(q)| / 5` | Ver ambiguidade abaixo | As **10** consultas do RAG | Req. 13 |
| **Hit@5** | 1 se ao menos um relevante foi **enviado à LLM**; 0 caso contrário | Ver ambiguidade abaixo | As **10** consultas do RAG | Req. 13 |

**Ferramenta de verificação indicada:** `ir_measures` (https://ir-measur.es/en/latest/) — Req. 6, Referências.

Observações e lacunas:
- **[NÃO ESPECIFICADO]** Base do logaritmo e forma do desconto do DCG. → **Conservador:** a convenção da ferramenta indicada (`ir_measures`/trec_eval: desconto `1/log2(i+1)`), declarada no relatório.
- **[AMBIGUIDADE: P@10 com menos de 10 documentos recuperados]** Um modelo lexical que só ranqueia documentos com algum termo da consulta pode devolver menos de 10. → **Conservador:** denominador fixo em 10 (convenção trec_eval).
- **[AMBIGUIDADE: relevância no Recall@5/P@5/Hit@5]** O Req. 13 não repete a regra. → **Conservador:** a mesma do Req. 6 (grau > 0) e o mesmo denominador `R` do R@10.
- **[AMBIGUIDADE: consulta com R = 0]** R@10 e AP@10 ficam indefinidos. → **Conservador:** seguir a convenção de `ir_measures` e verificar se esse caso existe no test.
- **[NÃO ESPECIFICADO]** Testes de significância estatística: não exigidos.
- **Consistência obrigatória:** `mean(AP10)` em `metrics_per_query.csv` = MAP@10 reportado (Req. 15).

---

## 5. Protocolo Experimental

### 5.1 Regras gerais
| Regra | Origem |
|---|---|
| dev para ajuste; test só para avaliação final (C3–C5) | Metodologia |
| Variar apenas o componente investigado em cada comparação (C21) | Análise crítica |
| Não comparar scores de modelos diferentes sem justificativa/normalização (C20) | Análise crítica |
| Toda afirmação analítica sustentada por dados observados (C19) | Req. 7; Análise crítica |
| Análise crítica é "parte central": exemplos concretos, ligação com os conceitos, análises por consulta para explicar comportamentos | Análise crítica |

### 5.2 Tolerância a erros (Req. 4)
- **Estrutura:** 10 consultas do test × 3 condições (original, com erro, corrigida), avaliadas no sistema **lexical** com **NDCG@10**.
- **Tipos de erro:** inserção, remoção, substituição, troca da ordem de caracteres.
- **Restrições de seleção:**
  - (iii) NDCG@10 original > 0;
  - (ii) o termo com erro não pode existir no dicionário (pré-stemming).
- **Fórmula de recuperação:** `(NDCG_corr − NDCG_erro) / (NDCG_orig − NDCG_erro)` se o denominador > 0; senão, registrar "o erro não reduziu o desempenho".
- **Ajuste:** limiares e parâmetros da correção **no dev**.
- **Opcional:** aplicar as mesmas consultas com erro ao sistema semântico.
- **Isolamento:** os demais experimentos usam as consultas originais (C8).

### 5.3 RAG (Req. 11–13)
| Item | Condição A | Condição B |
|---|---|---|
| Entrada da LLM | Só a pergunta | Pergunta + Top-5 pós-reranking com identificadores |
| Acesso ao NFCorpus | Não | Sim (via Top-5) |
| LLM / parâmetros | Mesmos | Mesmos |
| Idioma | Igual em A e B (informar) | Igual em A e B (informar) |
| Instruções do prompt | `[NÃO ESPECIFICADO]` | Responder com base no contexto; citar as fontes; avisar se faltar evidência |
| Temperatura | 0 ou a menor possível (recomendado) | Idem |
| Seed | Fixa quando disponível (recomendado) | Idem |
| Fidelidade avaliada contra | Os **mesmos Top-5 de B** | Os Top-5 enviados |
| Rastreabilidade | **N/A** (fora da média) | 0/1/2 |

**Critérios da avaliação humana (cada nota com justificativa breve):**

| Critério | 0 | 1 | 2 |
|---|---|---|---|
| Fidelidade | Afirmações importantes não são sustentadas pelo contexto ou o contradizem | Resposta parcialmente sustentada | As afirmações principais são sustentadas pelo contexto |
| Completude | Não responde aos aspectos principais | Responde parcialmente | Cobre adequadamente os aspectos que podem ser respondidos |
| Utilidade | Não responde à necessidade da consulta | Parcialmente útil | Responde diretamente à necessidade apresentada |
| Rastreabilidade | Não é possível identificar as fontes | Só parte das afirmações pode ser associada às fontes | As afirmações principais podem ser associadas aos `doc_ids` apresentados |

**Protocolo da avaliação (recomendações do Req. 13):**
- avaliar às cegas quanto à condição (ex.: embaralhar as respostas);
- quando possível, dois integrantes avaliam de forma independente e reportam a concordância.

---

## 6. Artefatos de Entrega

### 6.1 CSVs padronizados (Req. 15; Entrega 3)

| Arquivo | Colunas mínimas | Restrições | Gerado por |
|---|---|---|---|
| `results_lexical.csv` | `query_id, doc_id, rank, score, relevance` | Top-100 por consulta | RF-3 |
| `results_semantic.csv` | Idem | Top-100 por consulta | RF-5 |
| `results_hybrid.csv` | Idem | Top-100 por consulta | RF-8 |
| `results_reranking.csv` | Idem | Top-100 por consulta | RF-9 |
| `metrics_per_query.csv` | `query_id, model, P10, R10, AP10, NDCG10` | `mean(AP10)` = MAP@10 reportado | RF-6, RF-8, RF-9 |
| `typo_experiment.csv` | `query_id, query_original, query_erro, query_corrigida, tipo_erro, NDCG10_original, NDCG10_erro, NDCG10_corrigida` | 10 linhas (uma por consulta) | RF-4 |
| `rag_experiment.csv` | `query_id, query, condition (A/B), context_doc_ids, prompt, answer`, as **4 notas** (fidelidade, completude, utilidade, rastreabilidade) e as **4 justificativas** | 20 linhas (10 × 2); rastreabilidade "N/A" em A | RF-12, RF-13 |

Lacunas dos CSVs:
- **[AMBIGUIDADE: split dos rankings/métricas]** Não dito. → **Conservador:** as 323 consultas do test (avaliação final). Resultados do dev, se entregues, em arquivos separados.
- **[AMBIGUIDADE: coluna `relevance`]** Não diz se é binária ou graduada. → **Conservador:** grau original dos qrels, com 0 para não julgados (o valor binário pode ser derivado dele).
- **[AMBIGUIDADE: menos de 100 candidatos]** "os 100 primeiros documentos de cada consulta" assume que existem 100. O lexical por índice invertido pode retornar menos. → **Conservador:** gravar todos os candidatos existentes (até 100) e documentar quantas consultas ficaram com menos de 100.
- **[NÃO ESPECIFICADO]** Valores da coluna `model` em `metrics_per_query.csv`. → **Conservador:** os 4 sistemas comparados no Req. 9 (lexical, semântico, híbrido, híbrido + reranking).
- **[NÃO ESPECIFICADO]** Formato de `context_doc_ids` (separador, ordem) e de `tipo_erro` (vocabulário).
- **[NÃO ESPECIFICADO]** Colunas para a concordância entre avaliadores.

### 6.2 Código-fonte (Entrega 1)
- Todos os scripts, notebooks e módulos necessários para reproduzir os experimentos.
- **Sem** arquivos grandes regeneráveis; o README diz como gerá-los (C14).
- Entregue **no Moodle**; link não substitui (C13).

### 6.3 README (Entrega 2) — conteúdo mínimo
1. Integrantes (nome completo e Nº USP)
2. Instruções de instalação e execução
3. Versão da linguagem
4. Bibliotecas e versões principais
5. Modelo lexical escolhido
6. Modelo de embeddings
7. Modelo de reranking
8. LLM utilizada
9. Hardware empregado
10. Ordem recomendada de execução dos scripts/notebooks
11. Como gerar os arquivos grandes não incluídos (Entrega 1)

### 6.4 Relatório PDF (Entrega 4)
- **Limite:** até 20 páginas (só as referências ficam fora); sem anexos para contornar; Português ou Inglês.
- **Seções obrigatórias:**

| Seção | Conteúdo mínimo |
|---|---|
| a. Cabeçalho | Título, autores (nome completo e Nº USP), filiação, email |
| b. Introdução | Contextualização, motivação, objetivo |
| c. Metodologia | Pré-processamento, índice invertido, busca lexical, tolerância a erros, recuperação semântica, busca híbrida, reranking, RAG |
| d. Avaliação | Dataset, splits, qrels, métricas, modelos, parâmetros, ambiente computacional, protocolo experimental |
| e. Resultados e Análise | Lexical × semântico, tolerância a erros, híbrido, reranking, tempos, RAG, análise de falhas |
| f. Considerações finais | Principais conclusões tiradas dos experimentos |
| g. Uso de ferramentas de IA | Principais atividades em que a IA generativa foi usada como apoio (sem o histórico completo) |

- **Itens que o relatório deve declarar, consolidados dos requisitos:**

| Requisito | O que declarar |
|---|---|
| Req. 1 | Etapas, bibliotecas e decisões de pré-processamento |
| Req. 2 | Vocabulário, nº de postings, tempo de construção, tempo médio por consulta |
| Req. 3 | Modelo lexical escolhido e justificativa |
| Req. 5 | Os 9 itens do modelo bi-encoder (RF-5) |
| Req. 8 | k escolhido e evidência no dev |
| Req. 9 | Modelo de reranking e justificativa; tempos das etapas; comprimento máximo e truncamento |
| Req. 11 | Itens da LLM (RF-11) |
| Req. 12 | Idioma das respostas |
| Metodologia | Justificativa, caso não se concatene `title` + `text` |
| Uso de IA | Modelos de embeddings, cross-encoders e LLMs documentados como **componentes do experimento** (na Metodologia), separados do uso de IA como apoio (seção g) |

---

## 7. Algoritmos, ML e LLM

> Inventário das etapas que exigem um algoritmo, modelo de ML ou LLM, com as opções existentes. **Não é recomendação**: a escolha final de modelos deve ser feita no dev (C4).

### 7.1 Visão por etapa

| Etapa | Algoritmo / modelo exigido | Tipo | Roda localmente? | Sugestão do professor |
|---|---|---|---|---|
| RF-1 Pré-processamento | Tokenização, normalização, stopwords, stemming (ex.: Porter/Snowball) ou lematização | Algoritmo padrão / heurística linguística | Sim (CPU) | Não; bibliotecas externas permitidas |
| RF-2 Índice invertido | Dicionário + posting lists | Algoritmo padrão (estrutura de dados), **implementação própria** | Sim (CPU) | Não |
| RF-3 Busca lexical | BM25 **ou** TF-IDF + cosseno | Algoritmo padrão (modelo probabilístico / vetorial) | Sim (CPU) | Escolha entre os dois |
| RF-4 Geração de candidatos | Índice de k-gramas | Algoritmo padrão | Sim | Obrigatório (método fixado) |
| RF-4 Filtragem | Coeficiente de Jaccard sobre conjuntos de k-gramas | Algoritmo padrão + **limiar heurístico** (ajuste no dev) | Sim | Obrigatório |
| RF-4 Seleção | Distância de edição (variante `[NÃO ESPECIFICADA]`: Levenshtein ou Damerau-Levenshtein) | Algoritmo padrão (programação dinâmica) | Sim | Obrigatório |
| RF-5 Recuperação semântica | Bi-encoder pré-treinado + similaridade (cosseno/produto interno), busca exaustiva | **ML** (modelo pré-treinado; inferência apenas) | Sim (CPU viável para 3.633 docs; GPU acelera) | Sentence Transformers; `all-MiniLM-L6-v2`, `multi-qa-MiniLM-L6-cos-v1` |
| RF-8 Busca híbrida | Reciprocal Rank Fusion | Algoritmo padrão (fusão por rank, com hiperparâmetro k) | Sim | RRF obrigatório |
| RF-9 Reranking | Cross-encoder (ou reranking neural equivalente) | **ML** (pré-treinado; inferência) | Sim (CPU viável para 323 × 50 pares; GPU acelera) | `CrossEncoder` do Sentence Transformers; `cross-encoder/ms-marco-MiniLM-L6-v2` |
| RF-11/12 Geração | LLM | **LLM** | Sim (local/Colab) ou API | Famílias Qwen, Llama, Gemma, Mistral; API comercial permitida, não obrigatória |
| RF-12 Sorteio | `random.Random(42).sample` (Mersenne Twister do Python) | Algoritmo padrão | Sim | Código fornecido |
| RF-13 Concordância | Ex.: % de acordo, kappa de Cohen `[NÃO ESPECIFICADO]` | Estatística | Sim | Não |

### 7.2 Modelos de embeddings (bi-encoder) executáveis localmente `[EXTERNO]`

Todos rodam com `sentence-transformers` em CPU ou GPU (Colab T4 basta). Especificações tiradas dos model cards no Hugging Face; **conferir no card antes de reportar** (Req. 5 exige dimensão, comprimento máximo e normalização).

| Modelo | Parâm. (aprox.) | Dim. | Comprimento máx. (tokens) | Observações |
|---|---|---|---|---|
| `sentence-transformers/all-MiniLM-L6-v2` | 22M | 384 | 256 | **Sugerido pelo professor**; uso geral |
| `sentence-transformers/multi-qa-MiniLM-L6-cos-v1` | 22M | 384 | 512 | **Sugerido pelo professor**; treinado para QA/busca; embeddings normalizados (cosseno) |
| `sentence-transformers/multi-qa-mpnet-base-cos-v1` | 110M | 768 | 512 | Mesma família, maior |
| `BAAI/bge-small-en-v1.5` | 33M | 384 | 512 | Instrução opcional na consulta |
| `BAAI/bge-base-en-v1.5` | 110M | 768 | 512 | Idem |
| `intfloat/e5-base-v2` | 110M | 768 | 512 | Exige prefixos `query: ` / `passage: ` |
| `nomic-ai/nomic-embed-text-v1.5` | 137M | 768 | 8192 | Prefixos `search_query: ` / `search_document: `; `trust_remote_code` |
| `Qwen/Qwen3-Embedding-0.6B` | 0.6B | 1024 | longo (conferir) | Exemplo atual na documentação do Sentence Transformers; mais pesado |

> NFCorpus é biomédico/nutrição (documentos ≈ resumos científicos). A adequação ao domínio deve ser verificada **no dev**, não presumida.

### 7.3 Cross-encoders (reranking) executáveis localmente `[EXTERNO]`

| Modelo | Parâm. (aprox.) | Comprimento máx. | Observações |
|---|---|---|---|
| `cross-encoder/ms-marco-MiniLM-L6-v2` | 22M | 512 | **Sugerido pelo professor** |
| `cross-encoder/ms-marco-MiniLM-L12-v2` | 33M | 512 | Mesma família, mais profundo |
| `cross-encoder/ms-marco-TinyBERT-L2-v2` | 4M | 512 | O mais rápido |
| `BAAI/bge-reranker-base` | 278M | 512 | Mais pesado |
| `BAAI/bge-reranker-v2-m3` | 568M | 8192 | Multilíngue; GPU recomendada |
| `mixedbread-ai/mxbai-rerank-xsmall-v1` / `-base-v1` | 70M / 184M | 512 | Alternativa leve |

### 7.4 LLM — opções locais `[EXTERNO]`

O enunciado cita as famílias **Qwen, Llama, Gemma e Mistral**, executadas localmente ou no Google Colab. Para 20 gerações (10 consultas × 2 condições), modelos de 3–8B quantizados em 4 bits cabem numa GPU T4 (16 GB) do Colab ou num notebook com 8–16 GB de RAM.

| Família | Exemplos de checkpoints instruct | Execução típica | Reprodutibilidade |
|---|---|---|---|
| Qwen | Qwen3-4B, Qwen3-8B (Apache 2.0) | Ollama, llama.cpp, HF `transformers`, vLLM | Ollama aceita `temperature: 0` e `seed`; `transformers` com `do_sample=False` (greedy) |
| Llama | Llama 3.1 8B Instruct, Llama 3.2 3B Instruct | Idem | Idem |
| Gemma | Gemma 3 4B, Gemma 4 (variantes pequenas) | Idem; Gemma também no free tier da Gemini API | Idem |
| Mistral | Mistral 7B Instruct, Mistral Small | Idem | Idem |

Itens a reportar se o modelo for local (Req. 11): hardware (GPU/CPU, VRAM/RAM), quantização, versão/tag exata do checkpoint e runtime.

### 7.5 LLM — opções por API (preços de lista, USD por 1M tokens; consultados em 11/11/2026) `[EXTERNO]`

| Provedor | Modelo (ID) | Entrada | Saída | Free tier | Fonte |
|---|---|---|---|---|---|
| Anthropic | `claude-haiku-5-5` (prompt ≤ 100K) | 0,10 | 0,50 | Não informado | platform.claude.com/docs/en/about-docs/pricing |
| Anthropic | `claude-haiku-4-5` | 1,00 | 5,00 | Não informado | Idem |
| Anthropic | `claude-sonnet-5-5` | 2,00 | 10,00 | Não informado | Idem |
| OpenAI | `gpt-6-luna` | 0,10 | 0,50 | Não informado | developers.openai.com/api/docs/pricing |
| OpenAI | `gpt-6.1-terra` | 1,00 | 5,00 | Não informado | Idem |
| Google | `gemini-2.5-flash-lite` | 0,10 | 0,40 | Sim | ai.google.dev/gemini-api/docs/pricing |
| Google | `gemini-3.1-flash-lite` | 0,25 | 1,50 | Sim | Idem |
| Google | `gemini-2.5-flash` | 0,30 | 2,50 | Sim | Idem |
| Google | `gemini-3.5-flash` | 1,50 | 9,00 | Sim | Idem |
| Google | Gemma (via Gemini API) | — | — | Sim (sem plano pago) | Idem |
| Groq | Modelos abertos (Llama, Qwen, GPT-OSS…) | varia | varia | Sim (com limites de taxa) | Catálogo mudou em ago–out/2026; conferir em console.groq.com |

Ressalvas:
- No **free tier** da Gemini API, o conteúdo **é usado para melhorar os produtos** do Google (página de preços). Isso é relevante se o prompt tiver dados sensíveis (não é o caso do NFCorpus, mas convém registrar).
- O suporte a **seed** varia por provedor e por modelo: verificar na documentação antes de declarar reprodutibilidade (Req. 12 pede seed "quando disponível").
- A página da Gemini mostrou preços diferentes conforme a variante consultada (há modelos com mudança de preço datada). **Reconferir no dia.**

**Estimativa de custo do experimento obrigatório (Req. 11 pede "custo aproximado")**
- **Premissas** (a medir com o tokenizer real):
  - 10 chamadas A com ≈ 100 tokens de entrada cada;
  - 10 chamadas B com ≈ 3.000 tokens de entrada cada (pergunta + 5 documentos);
  - `max_tokens` = 512 por resposta.
- **Total:** ≈ 31K tokens de entrada + ≤ 10,2K de saída.

| Modelo | Custo estimado (20 chamadas) |
|---|---|
| `gemini-2.5-flash-lite` | ≈ US$ 0,007 |
| `claude-haiku-5-5` / `gpt-6-luna` | ≈ US$ 0,008 |
| `gemini-3.1-flash-lite` | ≈ US$ 0,023 |
| `gemini-2.5-flash` | ≈ US$ 0,035 |
| `claude-haiku-4-5` / `gpt-6.1-terra` | ≈ US$ 0,082 |
| `gemini-3.5-flash` | ≈ US$ 0,14 |
| `claude-sonnet-5-5` | ≈ US$ 0,16 |

Reexecuções multiplicam esses valores linearmente. Modelos com "thinking" cobram os tokens de raciocínio como saída.

---

## 8. Pontos de Atenção Críticos (5 maiores riscos de perda de nota)

**1. Índice invertido "não do zero" ou varredura completa por consulta** (≈10% + critério 1 das Observações finais)
> "Não será permitido percorrer integralmente todos os documentos da coleção a cada consulta para calcular o ranking." — Req. 2
> "não mecanismos prontos que construam e consultem automaticamente o índice textual." — Metodologia

Risco adicional: deixar de registrar as quatro estatísticas obrigatórias (vocabulário, postings, tempo de construção, tempo médio por consulta).

**2. Vazamento do test (tuning ou seleção com qrels do test)** (afeta todos os componentes e a reprodutibilidade, 20%)
> "deverá ser feita com base no conjunto dev, e não pela comparação de alternativas no conjunto test." — Metodologia

Pontos sensíveis:
- escolha do modelo de embeddings e do reranker;
- k do RRF;
- limiares da correção;
- seleção das 10 consultas do Req. 4, onde os qrels do test devem servir só ao filtro (iii).

**3. Convenções de métricas erradas ou não reproduzíveis** (20%: avaliação quantitativa, análise crítica e reprodutibilidade)
> "dividida pelo número total de documentos relevantes da consulta nos qrels (convenção do trec_eval e do ir_measures)" — Req. 6
> "A média de AP10 deverá corresponder ao MAP@10 reportado." — Req. 15
> "Resultados existentes apenas como saída de células de notebook não serão considerados como arquivos de resultados entregues." — Req. 15

Pontos sensíveis:
- denominador do AP@10 e do R@10;
- IDCG calculado a partir de **todos** os julgamentos;
- ganho linear;
- P@10 com menos de 10 recuperados;
- conferência com `ir_measures`;
- CSVs incluídos no pacote entregue.

**4. Protocolo do RAG fora da especificação** (≈20%, o maior bloco)
> "todos os grupos utilizarão as mesmas consultas, sorteadas com seed 42" — Req. 12
> "Não será suficiente afirmar simplesmente que 'a LLM errou'; deverá ser verificado se a informação necessária chegou ou não ao contexto." — Req. 14

Pontos sensíveis:
- lista de consultas diferente da oficial;
- parâmetros diferentes entre A e B;
- prompt de B sem as três instruções;
- Fidelidade de A não comparada com os Top-5 de B;
- Rastreabilidade de A diferente de "N/A";
- notas sem justificativa;
- Precision@5 e Hit@5 ausentes (o roadmap só lista Recall@5).

**5. Análises sem evidência dos próprios dados** (parte dos 20% + critério 5)
> "somente deverão ser feitas quando sustentadas por valores efetivamente observados nos próprios dados." — Req. 7
> "Valores de scores produzidos por modelos diferentes não deverão ser comparados diretamente sem justificativa ou normalização adequada." — Análise crítica

Afeta os Req. 7, 10 e 14: cada hipótese (tamanho do documento, frequência de termo, sinonímia etc.) precisa de um número observado. Os scores do BM25 e do bi-encoder não são comparáveis entre si.

---

## Apêndice A — Critérios de correção (Observações finais)

| Componente | Peso aprox. |
|---|---|
| Indexação e recuperação lexical | 10% |
| Tolerância a erros | 10% |
| Recuperação semântica | 10% |
| Busca híbrida | 10% |
| Reranking | 10% |
| Implementação e avaliação do RAG | 20% |
| Avaliação quantitativa, análise crítica e reprodutibilidade | 20% |
| Compreensão individual / defesa | 10% |

Perguntas de correção:
1. Índice invertido e busca lexical corretos e conformes?
2. Tolerância a erros, semântica, híbrida e reranking corretos?
3. RAG implementado e avaliado conforme o protocolo?
4. Avaliação quantitativa correta e reprodutível?
5. Análises com exemplos concretos e evidências dos experimentos?
6. Relatório bem apresentado, organizado e tecnicamente sólido?
7. Integrantes demonstram compreensão na defesa?

## Apêndice B — Defesa individual (Defesa do projeto)

Qualquer integrante pode ser solicitado a:
- explicar parte do código;
- mostrar como o índice invertido foi construído;
- explicar um cálculo do RRF;
- interpretar um resultado por consulta;
- justificar mudanças de posição após o reranking;
- mostrar quais documentos foram enviados à LLM;
- explicar por que uma resposta foi considerada fiel ou não;
- alterar um parâmetro e prever o efeito;
- fazer uma pequena modificação no código ou na configuração experimental.

## Apêndice C — Divergências entre texto e roadmap (p. 9)

| Ponto | Texto | Roadmap | Seguir |
|---|---|---|---|
| Stemming/lematização | Listados entre as etapas possíveis | "(opcional)" | Opcional, com justificativa |
| Métricas do RAG (etapa 14) | Recall@5, Precision@5, Hit@5 | Só "Recall@5" | Texto |
| Reranking (etapa 10) | "tempo médio das etapas" | "custo/tempo das etapas" | Reportar tempo; custo quando aplicável |
| Busca lexical (etapa 4) | Req. 2 pede tempo médio por consulta | Repete "Medir tempo médio das consultas" | Consistentes |

## Apêndice D — Uso de IA (seção do enunciado)
- **IA como componente** (embeddings, cross-encoders, LLMs): documentar na **Metodologia**.
- **IA como apoio** (programação, depuração, explicação de conceitos, análises, revisão textual): seção **"Uso de ferramentas de IA"** com as principais atividades, sem o histórico completo de interações.
- Os integrantes continuam responsáveis por entender integralmente o código, a metodologia e os resultados.

## Apêndice E — Pendências de verificação deste documento
1. Rodar o snippet do Req. 12 e confirmar as 42 perguntas e as 10 consultas (seção 2.4).
2. Contar as consultas por split (`[EXTERNO]` 2.590 / 324 / 323) e a distribuição dos graus em `qrels/test.tsv`.
3. Checar se há consulta do test com R = 0 (seção 4).
4. Conferir nos model cards a dimensão, o comprimento máximo e a normalização dos modelos candidatos (seções 7.2 e 7.3).
5. Reconferir os preços da API no dia da execução (seção 7.5).

## Apêndice F — Referências e materiais de apoio citados no enunciado
- MANNING, C. D.; RAGHAVAN, P.; SCHÜTZE, H. *Introduction to Information Retrieval*. Cambridge University Press, 2008. https://nlp.stanford.edu/IR-book/information-retrieval-book.html
- BEIR: https://github.com/beir-cellar/beir ; lista de datasets: https://github.com/beir-cellar/beir/wiki/Datasets-available
- NFCorpus no Hugging Face: https://huggingface.co/datasets/BeIR/nfcorpus
- Sentence Transformers: https://www.sbert.net/ (busca semântica e Cross-Encoders: links no RF-5 e no RF-9)
- ir_measures: https://ir-measur.es/en/latest/
- O enunciado ressalta que a implementação deve respeitar a especificação, principalmente a construção própria do índice invertido e a compreensão do código.

## Fontes externas (seção 7 e marcações `[EXTERNO]`)
- Anthropic — preços: https://platform.claude.com/docs/en/about-docs/pricing (consultado em 11/11/2026)
- OpenAI — preços: https://developers.openai.com/api/docs/pricing (consultado em 11/11/2026)
- Google — preços da Gemini API: https://ai.google.dev/gemini-api/docs/pricing (consultado em 11/11/2026)
- Groq — free tier e mudanças de catálogo: https://www.developersdigest.tech/models/free-api/groq ; https://www.usagepricing.com/blueprint/activity/groq-2026-08-26-llama-enterprise-only
- Sentence Transformers — modelos pré-treinados: https://www.sbert.net/docs/sentence_transformer/pretrained_models.html
- BEIR — formato e datasets: https://github.com/beir-cellar/beir
