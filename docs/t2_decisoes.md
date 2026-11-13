# Registro de Decisões — T2 (SCC0282, RI 2026)

> Log das decisões de design do T2. Cada entrada aponta o requisito afetado (ver `docs/t2_requisitos.md`) e o que ainda falta fixar.
> Formato: ID · data · decisão · requisito · justificativa · pendências.

| ID | Data | Decisão | Requisito | Detalhes |
|---|---|---|---|---|
| D1 | — | Baseline lexical = **BM25** (reaproveitado do T1, índice invertido refeito do zero) | Req. 2, Req. 3 | `docs/t2_busca_lexical_bm25_contexto.md` |
| D2 | 11/11/2026 | LLM do RAG = **Gemini via API, free tier**; modelo `gemini-3.1-flash-lite` (proposto em `t2_arquitetura.md` §3.3) | Req. 11, Req. 12 | Abaixo |
| D3 | 11/11/2026 | Modelos locais restritos aos **indicados no enunciado ou na documentação do Sentence Transformers**, executáveis em **CPU local** (i5, 4 threads, 20 GB RAM); **sem comparação entre modelos** | Req. 5, Req. 9; Metodologia (restrição C4) | Abaixo |
| D4 | 11/11/2026 | Bi-encoder = **`sentence-transformers/multi-qa-MiniLM-L6-cos-v1`** | Req. 5 (RF-5) | `t2_arquitetura.md` §3.1 |
| D5 | 11/11/2026 | Cross-encoder = **`cross-encoder/ms-marco-MiniLM-L2-v2`** | Req. 9 (RF-9) | `t2_arquitetura.md` §3.2 |
| D6 | 13/11/2026 | Incluir a **extensão semântica opcional do RF-4**: mesmas consultas (original, com erro, corrigida) também no bi-encoder | Req. 4 (RF-4) | Abaixo; `t2_tds.md` TDS-4 (TDS-DEC-12) |
| D7 | 13/11/2026 | Lexical com menos de 100 candidatos: **não completar**; explicitar sempre que ocorrer | Req. 15 (RF-15) | `t2_tds.md` TDS-3 (TDS-DEC-4) |
| D8 | 13/11/2026 | Tokenização **sem pontuação** (regex `[a-z0-9]+` em minúsculas) | Req. 1 (RF-1) | `t2_tds.md` TDS-1 (TDS-DEC-13) |
| D9 | 13/11/2026 | **Sem atribuição de IA** nos commits: `attribution` vazio no Claude Code, hook `commit-msg` e GitHub Action | Entrega (autoria) | `AGENTS.md`, `.githooks/`, `.github/workflows/` |
| D10 | 13/11/2026 | Documentos de apoio à IA (`docs/t2_*.md`, `AGENTS.md`, `CLAUDE.md`, `.claude/`) são **removidos após o código ficar pronto**, antes da entrega | Entrega | `t2_tds.md` §5 |

## D2 — LLM do RAG: Gemini (free tier)

**Justificativa (para o relatório):**
- Custo zero; o enunciado não exige serviço pago (Req. 11).
- O volume é pequeno: 20 chamadas obrigatórias (10 consultas × condições A/B, Req. 12), compatível com os limites do free tier.
- Sem necessidade de GPU ou hardware local.

**Implicações a registrar:**
- **Forma de execução:** serviço externo (API), free tier; custo = US$ 0.
- **Uso dos dados:** no free tier, o Google usa entradas e saídas para melhorar produtos, e pode haver revisão humana. Aceitável porque o NFCorpus é público; nenhum dado pessoal vai nos prompts.
- **Chave de API:** fora da entrega (C15). Carregar por variável de ambiente e documentar no README.
- **Limites:** RPM/TPM/RPD por projeto; RPD zera à meia-noite do horário do Pacífico. Ver o limite real no AI Studio antes de rodar e tratar 429 com retry/backoff.
- **Reprodutibilidade:** fixar o ID exato do modelo (sem alias `-latest`) e registrar por execução a versão do modelo devolvida pela API, a temperatura, `max_output_tokens`, a seed e a configuração de "thinking", se houver.

**Pendências (fixar antes de rodar o Req. 12):**
- [ ] Confirmar `gemini-3.1-flash-lite`: stable, free tier, sem roteamento automático. Evitar a família 2.5 (acesso restrito a quem já usava) e o `gemini-3.5-flash` (roteado automaticamente para o 3.6). Conferir o RPD no AI Studio.
- [ ] Temperatura (recomendado pelo enunciado: 0).
- [ ] Seed fixa: testar 2 chamadas idênticas e comparar.
- [ ] `max_output_tokens`.
- [ ] Configuração de thinking (desligado / nível fixo), se aplicável.
- [ ] Idioma das respostas (igual em A e B; informar no relatório).
- [ ] Prompts das condições A e B (B: responder com base no contexto, citar fontes, avisar quando faltar evidência).

## D3 — Restrição de modelos locais e escolha sem comparação

**Decisão:**
- Usar apenas modelos citados no enunciado ou na documentação do Sentence Transformers que rodem em CPU local.
- **Um** bi-encoder e **um** cross-encoder, escolhidos a priori.

**Base textual:**
- O enunciado não exige comparar modelos ("A escolha do modelo é livre", Req. 5; reranker "informado e justificado", Req. 9).
- As comparações obrigatórias são entre sistemas (lexical, semântico, híbrido, reranking).

**Atendimento à restrição C4** (escolha "com base no dev, e não pela comparação de alternativas no test"):
- O critério de escolha é viabilidade e indicação; não usa nenhum resultado de avaliação.
- Cada modelo é **validado no dev** (NDCG@10 reportado) antes da execução única no test.
- Rankings públicos do NFCorpus (medidos no test) não são usados nem citados como justificativa.

**Ambiguidade registrada:** "com base no dev" pode ser lido como "comparar alternativas no dev". O relatório deve declarar a leitura adotada.

**Consequência:** nenhum modelo biomédico. Material para a análise do RF-7 (termos técnicos/médicos, lexical × semântico).

## D4 — Bi-encoder: multi-qa-MiniLM-L6-cos-v1

**Justificativa (para o relatório):**
1. É o que roda na máquina disponível em tempo compatível com o prazo.
2. Indicado pelo professor.
3. Treinado para busca (~215M pares QA, inclui MS MARCO).
4. Limite de 512 tokens (o `all-MiniLM-L6-v2` trunca em 256).
5. Embeddings normalizados (cosseno = produto interno).

**Limitações a declarar:** treinado com até 250 word pieces; não especializado em domínio biomédico.

**Pendências:**
- [ ] Fixar a revisão do modelo no HF.
- [ ] Medir tempo de codificação, tempo por consulta e % de documentos com mais de 512 tokens.
- [ ] Reportar o NDCG@10 no dev.

## D5 — Cross-encoder: ms-marco-MiniLM-L2-v2

**Justificativa (para o relatório):**
- Reranker MiniLM mais rápido da família indicada pelo professor (4.100 docs/s na referência V100 da doc. ST, ~2,3× o L6).
- Viável em CPU para ~32 mil pares (dev + test).

**Trade-off declarado:** NDCG@10 TREC DL19 de 71,01 contra 74,30 do L6 (−3,3 pontos, fora do domínio). Escolha por custo, não por qualidade. Não presumir ganho sobre o híbrido; casos de piora alimentam o RF-10.

**Pendências:**
- [ ] Confirmar o `max_length` no config do tokenizer e fixar 512 explicitamente.
- [ ] Medir a % de pares truncados e o tempo médio por consulta.
- [ ] Reportar o NDCG@10 do híbrido + reranking no dev.

## D6 — Extensão semântica do experimento de erros ortográficos

**Decisão:** além do BM25 (obrigatório), rodar no bi-encoder (D4) as mesmas 10 consultas do test nas três versões (original, com erro, corrigida), com texto cru, e calcular o NDCG@10 de cada uma.

**Base textual:** "Aplicar as mesmas consultas com erro ao sistema semântico é opcional e pode enriquecer a discussão sobre a robustez de embeddings a erros ortográficos" (Req. 4).

**Justificativa:** custo quase nulo, pois os embeddings dos documentos já estão calculados e são só 20 consultas curtas a mais. Permite comparar, consulta a consulta, a queda causada pelo erro no lexical e no semântico.

**O que não muda:**
- A seleção das 10 consultas (NDCG@10 > 0) e o ajuste dos limiares usam só o BM25 (Req. 4 iii, iv).
- A consulta corrigida vem do mesmo corretor (dicionário lexical).
- Os demais experimentos continuam usando as consultas originais.

**Saídas:**
- Colunas `NDCG10_sem_original`, `NDCG10_sem_erro`, `NDCG10_sem_corrigida`, `recuperacao_sem` e `status_sem` em `typo_experiment.csv`.
- Resumo em `artifacts/typo/robustez_lex_sem.csv`.

## D7 — BM25 com menos de 100 candidatos

**Decisão:** o ranking lexical contém só os documentos com score > 0 (ao menos um termo da consulta). Se uma consulta tiver menos de 100 candidatos, o `results_lexical.csv` terá menos de 100 linhas para ela, **sem completar** com documentos de score 0.

**Explicitação obrigatória:**
- aviso no console ao gerar a run;
- registro em `artifacts/lexical_candidates_{split}.csv`;
- menção no relatório, listando as consultas afetadas ou dizendo "nenhuma consulta".

**Justificativa:** documentos sem nenhum termo da consulta não têm score BM25 definido de forma significativa, e ordená-los seria arbitrário. A métrica P@10 mantém o denominador 10.

## D8 — Tokenização sem pontuação

**Decisão:** `tokenize(text) = re.findall(r"[a-z0-9]+", text.lower())`. Vale para documentos, consultas e o dicionário do corretor (RF-4).

**Motivo:** no T1, pontuação colada aos tokens ("cancer," ≠ "cancer") dividiu o vocabulário. Efeitos a declarar: "omega-3" → `omega`, `3`; "1.5" → `1`, `5`.

## D9 — Sem atribuição de IA nos commits

**Decisão:** nenhum commit ou PR pode ter IA como autora ou coautora.

**Travas:**
1. **Claude Code:** `.claude/settings.json` com `"attribution": {"commit": "", "pr": ""}`. Outras ferramentas: desativar a opção equivalente.
2. **Hook local** `.githooks/commit-msg`, ativado uma vez por clone com `git config core.hooksPath .githooks`.
3. **GitHub Action** `.github/workflows/no-ai-attribution.yml`, que verifica todos os commits de pushes e PRs.

## D10 — Limpeza dos documentos de apoio à IA

**Decisão:** quando o código estiver pronto e os resultados gerados, remover `docs/t2_*.md`, `AGENTS.md`, `CLAUDE.md` e `.claude/`. Antes, transcrever para o relatório as justificativas necessárias (D1–D9, TDS-DEC-n) e revisar o README. Lista exata no `t2_tds.md` §5.
