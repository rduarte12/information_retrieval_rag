"""
Baixa o NFCorpus (BEIR) do Hugging Face e grava no formato esperado pelo projeto:

    data/nfcorpus/corpus.jsonl
    data/nfcorpus/queries.jsonl
    data/nfcorpus/qrels/{train,dev,test}.tsv

Alternativa ao zip da UKP quando o servidor dela não responde. O enunciado cita o
Hugging Face (BeIR/nfcorpus e BeIR/nfcorpus-qrels) como fonte válida.

Uso, na raiz do repositório (com o .venv ativo, que já tem pandas e pyarrow):
    python scripts/baixar_nfcorpus_hf.py
"""
import json
import urllib.request
from pathlib import Path

import pandas as pd

BASE = "https://huggingface.co/datasets/BeIR"
ARQUIVOS = {
    "corpus.parquet": f"{BASE}/nfcorpus/resolve/main/corpus/corpus-00000-of-00001.parquet",
    "queries.parquet": f"{BASE}/nfcorpus/resolve/main/queries/queries-00000-of-00001.parquet",
    "qrels/train.tsv": f"{BASE}/nfcorpus-qrels/resolve/main/train.tsv",
    "qrels/dev.tsv": f"{BASE}/nfcorpus-qrels/resolve/main/dev.tsv",
    "qrels/test.tsv": f"{BASE}/nfcorpus-qrels/resolve/main/test.tsv",
}
OUT = Path("data") / "nfcorpus"


def txt(v):
    return "" if v is None or (isinstance(v, float) and pd.isna(v)) else str(v)


def parquet_para_jsonl(src: Path, dst: Path, campos):
    df = pd.read_parquet(src)
    faltando = [c for c in campos if c not in df.columns]
    if faltando:
        raise SystemExit(f"{src.name}: colunas {faltando} ausentes; colunas encontradas: {list(df.columns)}")
    with dst.open("w", encoding="utf-8", newline="\n") as f:
        for r in df[campos].to_dict("records"):  # itertuples renomearia "_id"
            f.write(json.dumps({c: txt(r[c]) for c in campos}, ensure_ascii=False) + "\n")
    return len(df)


def main():
    (OUT / "qrels").mkdir(parents=True, exist_ok=True)
    for nome, url in ARQUIVOS.items():
        destino = OUT / nome
        print(f"baixando {url}")
        urllib.request.urlretrieve(url, destino)

    n_docs = parquet_para_jsonl(OUT / "corpus.parquet", OUT / "corpus.jsonl", ["_id", "title", "text"])
    n_q = parquet_para_jsonl(OUT / "queries.parquet", OUT / "queries.jsonl", ["_id", "text"])
    (OUT / "corpus.parquet").unlink()
    (OUT / "queries.parquet").unlink()

    def n_consultas(split):
        df = pd.read_csv(OUT / "qrels" / f"{split}.tsv", sep="\t", dtype=str)
        return df.iloc[:, 0].nunique()

    contagens = {"docs": n_docs, "queries": n_q, "dev": n_consultas("dev"), "test": n_consultas("test")}
    esperado = {"docs": 3633, "queries": 3237, "dev": 324, "test": 323}
    print(contagens)
    if contagens != esperado:
        raise SystemExit(f"ATENÇÃO: contagens diferentes do esperado {esperado}")
    print("OK: NFCorpus em data/nfcorpus/ com as contagens do enunciado.")


if __name__ == "__main__":
    main()
