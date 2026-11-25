# Trabalho Prático 2: Busca Lexical, Semântica, Híbrida e RAG

SCC0282 Recuperação de Informação, ICMC-USP, 2º semestre de 2026.

| Integrante | Nº USP |
|---|---|
| Rafael Mendonça Duarte | 16817608 |
| João Pedro Correia Caetano | 16987067 |

Este repositório contém um sistema de recuperação de informação sobre o NFCorpus (versão do BEIR). O pipeline tem cinco etapas:

1. busca lexical com BM25 sobre um índice invertido implementado pelo grupo;
2. busca semântica com um bi-encoder;
3. fusão dos dois rankings com RRF;
4. reranking do top 50 híbrido com um cross-encoder;
5. geração de respostas com RAG usando o Gemini.

A avaliação usa P@10, R@10, MAP@10 e NDCG@10 no split de teste. O experimento de RAG compara respostas sem contexto (condição A) e com os 5 primeiros documentos recuperados (condição B).

## Configuração

| Item | Escolha |
|---|---|
| Pré-processamento | minúsculas, tokenização por regex `[a-z0-9]+`, remoção de stopwords (NLTK, inglês) e stemming Snowball, configuração escolhida no dev |
| Busca lexical | BM25 com k1 = 1,2 e b = 0,75 sobre índice invertido próprio |
| Bi-encoder | `sentence-transformers/multi-qa-MiniLM-L6-cos-v1` (revisão `b207367332321f8e44f96e224ef15bc607f4dbf0`), similaridade do cosseno, busca exaustiva, texto sem pré-processamento |
| Fusão | RRF com k = 10, escolhido no dev entre {10, 60, 100} |
| Cross-encoder | `cross-encoder/ms-marco-MiniLM-L2-v2` (revisão `1b5cd67b15209f24824c50370e0397743aa9b787`), aplicado ao top 50 do híbrido |
| LLM | `gemini-3.1-flash-lite` pela Gemini API (free tier), temperature 0.0, seed 42, max_output_tokens 1024, respostas em português |
| Correção ortográfica | k-gramas (k = 2), filtro de Jaccard (limiar 0,2) e distância de edição (máximo 1) |
| Hardware de referência | CPU Intel i5 (4 threads), 20 GB de RAM, sem GPU |
| Linguagem e bibliotecas | Python 3.14.10; numpy 2.5.4, pandas 3.1.1, nltk 3.10.4, torch 2.15.0 (CPU), transformers 5.23.0, sentence-transformers 6.3.0, ir_measures 0.4.3, google-genai 3.3.0, pytest 9.2.0 (lista completa em `requirements.txt`) |

O split dev foi usado para todas as escolhas de parâmetros. O split test foi executado uma única vez, com a configuração já fixada.

## Instalação

```bash
git clone https://github.com/rduarte12/information_retrieval_rag.git
cd information_retrieval_rag

python -m venv .venv
source .venv/bin/activate            # no Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -c "import nltk; nltk.download('stopwords')"
```

## Base de dados

Usamos o NFCorpus distribuído pelo BEIR (`corpus.jsonl`, `queries.jsonl` e `qrels/{train,dev,test}.tsv`). O script abaixo baixa a coleção do Hugging Face (`BeIR/nfcorpus` e `BeIR/nfcorpus-qrels`), grava os arquivos em `data/nfcorpus/` no formato do BEIR e confere as contagens (3633 documentos, 3237 consultas, 324 consultas de dev e 323 de teste com julgamentos):

```bash
python scripts/baixar_nfcorpus_hf.py
```

O mesmo conteúdo pode ser obtido no pacote original do BEIR: https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/nfcorpus.zip (extrair em `data/nfcorpus/`).

## Testes

```bash
pytest -m "not data"     # 79 testes que não dependem do dataset
pytest -m data           # 17 testes que leem data/nfcorpus/
pytest                   # os 96 testes
```

## Execução

Os comandos abaixo reproduzem os experimentos na ordem em que dependem um do outro. Não é preciso configurar `PYTHONPATH`.

```bash
python scripts/run_s2_dev.py          # índice, BM25 e comparação de pré-processamento no dev
python -m src.dense --split dev       # busca semântica no dev
python -m src.fusion --split dev      # escolha do k do RRF no dev
python -m src.rerank --split dev      # reranking no dev
python scripts/run_test.py            # execução única no test, gera os CSVs de results/
python -m src.spell                   # experimento de erros ortográficos, gera results/typo_experiment.csv
python -m src.rag_context             # seleção do contexto top 5 e métricas de recuperação do RAG
python -m src.llm                     # geração das respostas A e B com o Gemini
python -m src.analysis                # análises por consulta (lexical x semântico, reranking, falhas do RAG)
python scripts/medir_tempo_semantico.py   # tempo médio por consulta da busca semântica
python scripts/gerar_figuras.py           # figuras do relatório em results/figuras/
python scripts/verificar_resultados.py
python scripts/export_evidencias.py
```

Só o `src.llm` precisa de chave de API. Copie `.env.example` para `.env` e preencha `GEMINI_API_KEY`. O arquivo `.env` não é versionado. As respostas avaliadas estão em `results/rag_experiment.csv`. Uma nova execução do `src.llm` gera respostas novas, que podem não ser idênticas às avaliadas, mesmo com temperature 0.

Os experimentos foram executados em partes, em duas máquinas diferentes. Por isso, alguns valores intermediários podem diferir por arredondamento de ponto flutuante, como a ordem de documentos com scores praticamente empatados abaixo do top 10. As métricas reportadas são as mesmas nas duas máquinas.

## Resultados

Os arquivos pedidos no enunciado estão em `results/`:

| Arquivo | Conteúdo |
|---|---|
| `results_lexical.csv` | rankings do BM25 no test |
| `results_semantic.csv` | rankings do bi-encoder no test |
| `results_hybrid.csv` | rankings do RRF no test |
| `results_reranking.csv` | rankings do cross-encoder no test |
| `metrics_per_query.csv` | P@10, R@10, MAP@10 e NDCG@10 por consulta e por sistema |
| `typo_experiment.csv` | consultas original, com erro e corrigida, com o NDCG@10 das buscas lexical e semântica em cada uma |
| `rag_experiment.csv` | respostas das condições A e B, notas e justificativas |

A pasta `results/figuras/` tem as figuras do relatório, geradas por `scripts/gerar_figuras.py` a partir dos CSVs acima. A pasta `results/evidencias/` reúne os arquivos usados nas tabelas e análises do relatório:

| Pasta | Conteúdo |
|---|---|
| `indice/` | estatísticas do índice e número de candidatos lexicais por consulta |
| `dev/` | comparação de pré-processamento, escolha do k do RRF, grade da correção ortográfica e validação no dev |
| `modelos/` | modelos, revisões, truncamento e tempos do bi-encoder e do cross-encoder |
| `typo/` | erros introduzidos e robustez das buscas lexical e semântica |
| `rag/` | contexto enviado à LLM, métricas de recuperação, notas de cada avaliador, concordância e resumo da avaliação |
| `analise/` | casos por consulta: lexical x semântico, mudanças de posição no reranking e rastreio das falhas do RAG |

As respostas do RAG foram avaliadas pelos dois integrantes, de forma independente e às cegas (sem saber a condição de cada resposta), nos critérios fidelidade, completude, utilidade e rastreabilidade (notas de 0 a 2). A concordância foi medida com o kappa de Cohen, e as divergências foram resolvidas por consenso. As notas finais são as de consenso.

## Observações

- O índice invertido foi implementado do zero, e a busca lexical consulta apenas as listas de postings dos termos da consulta, sem percorrer a coleção.
- Quando o BM25 encontra menos de 100 documentos com algum termo da consulta, o ranking lexical não é completado artificialmente. No test, 127 das 323 consultas têm menos de 100 candidatos, e 15 não têm nenhum.
- Não são versionados o dataset, os embeddings, os modelos, os índices serializados, a pasta `artifacts/` e o `.env`.
