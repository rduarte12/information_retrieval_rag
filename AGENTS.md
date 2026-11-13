# AGENTS.md — Instruções para assistentes de IA

Vale para qualquer assistente (Claude Code, Codex, Copilot, Cursor, Gemini etc.) que leia ou altere este repositório. Humanos também seguem.

## Antes de começar

1. Leia o `README.md`, depois o `docs/t2_plano_execucao.md` para achar a sprint atual, e o `docs/t2_tds.md` na seção do componente correspondente.
2. Dúvida sobre obrigatoriedade: `docs/t2_requisitos.md`. Escolhas já feitas: `docs/t2_decisoes.md`.
3. Precedência: enunciado (`docs/enunciado/`) > requisitos > decisões > arquitetura > TDS > plano.

## Configuração obrigatória da ferramenta

Nenhum commit ou PR pode ter IA como autora ou coautora (D9).

- **Claude Code:** o repositório já traz `.claude/settings.json` com:
  ```json
  { "attribution": { "commit": "", "pr": "" } }
  ```
  Não remova nem sobrescreva esse arquivo. Se você usa um `settings.local.json`, não reative a atribuição nele.
- **Outras ferramentas:** desative a opção equivalente ("co-authored-by", "sign commits as agent", assinatura em PR).
- O hook `.githooks/commit-msg` e a Action `no-ai-attribution` recusam commits com atribuição de IA. Não tente contorná-los (`--no-verify`, editar o hook ou a Action).

## Regras invioláveis

1. **Git:** não faça `commit`, `push`, `merge`, `rebase`, `reset` nem crie tags sem autorização explícita de um integrante na conversa atual.
2. **Segredos:** nunca escreva chaves de API em arquivos versionados. Use `.env` (ignorado) e `.env.example`.
3. **Split test:** nunca use `qrels/test.tsv` para ajustar parâmetros, escolher modelos, selecionar exemplos ou "melhorar" resultados. Ajustes só no dev (restrições C3–C5, C7). O test roda uma vez (TDS-14).
4. **Índice invertido:** `src/inverted_index.py` e `src/bm25.py` não podem usar bibliotecas que construam ou consultem índices textuais, e o BM25 nunca percorre todos os documentos por consulta (C1, C2).
5. **Arquivos protegidos:** não altere `docs/`, `src/inverted_index.py`, `src/bm25.py`, `tests/reference/`, `.githooks/`, `.github/` nem `.claude/settings.json` sem pedido explícito. Proponha a mudança e registre em `docs/t2_decisoes.md`.
6. **Testes-contrato:** não apague, enfraqueça nem marque `skip` em testes para fazê-los passar. Valores esperados vêm do TDS ou de cálculo à mão.
7. **Modelos e parâmetros:** use exatamente os de `docs/t2_decisoes.md` (D2, D4, D5). Não troque de modelo por conta própria.
8. **Tokenização:** sem pontuação (D8): `re.findall(r"[a-z0-9]+", text.lower())`, a mesma para documentos, consultas e o dicionário do corretor.
9. **BM25 com menos de 100 candidatos:** não completar; registrar e avisar (D7).

## Como trabalhar

- Um componente por vez, na ordem do plano. Escreva os testes da sprint **antes** do código.
- Um componente só está pronto quando o critério de conclusão do TDS é verificado por teste em `tests/` ou por comando reproduzível.
- Mantenha nomes de arquivos, colunas e caminhos exatamente como no TDS (§0.4 e §2).
- Decisão de implementação não prevista: escolha a opção mais simples, registre em `docs/t2_decisoes.md` e avise o humano.
- Ao terminar uma sessão: atualize a tabela "Status dos componentes" do `README.md` e marque o checklist da sprint no plano.

## Comandos

```bash
pytest                       # todos os testes (sem dados reais)
pytest -m data               # testes que precisam do NFCorpus em data/nfcorpus/
pytest tests/test_bm25.py    # um componente
```

## Ambiente

- Python `<versão a preencher na S0>`; dependências em `requirements.txt`, com versões fixadas na S0.
- Só CPU (Intel i5, 4 threads, 20 GB RAM): `torch.set_num_threads(4)`; nada pode exigir GPU.
- Dataset em `data/nfcorpus/` (não versionado).
