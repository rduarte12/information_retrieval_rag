from __future__ import annotations

import csv
import hashlib
import json
import random
import string
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.bm25 import BM25
from src.data_loader import doc_text, load_dataset
from src.dense import DenseRetriever, model_slug
from src.evaluation import score_query
from src.inverted_index import InvertedIndex
from src.orchestrator import (
    freeze_config,
    guard_stage,
    load_config,
    record_stage_result,
    save_config,
)
from src.preprocessing import STOPWORDS_EN, preprocess, surface_vocab, tokenize

ERROR_TYPES = ("insercao", "remocao", "substituicao", "transposicao")


def kgrams(term: str, k: int) -> set[str]:
    """Return the set of character k-grams of `term` padded with '$' boundaries."""
    padded = f"${term}$"
    if len(padded) < k:
        return {padded}
    return {padded[i : i + k] for i in range(len(padded) - k + 1)}


def jaccard(a: set[str], b: set[str]) -> float:
    """Compute Jaccard similarity between two sets."""
    if not a and not b:
        return 0.0
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def osa_distance(a: str, b: str) -> int:
    """Compute Optimal String Alignment (restricted Damerau-Levenshtein) distance."""
    la, lb = len(a), len(b)
    if la == 0:
        return lb
    if lb == 0:
        return la

    dp = [[0] * (lb + 1) for _ in range(la + 1)]
    for i in range(la + 1):
        dp[i][0] = i
    for j in range(lb + 1):
        dp[0][j] = j

    for i in range(1, la + 1):
        for j in range(1, lb + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            val = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )
            if i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                val = min(val, dp[i - 2][j - 2] + 1)
            dp[i][j] = val
    return dp[la][lb]


def build_kgram_index(dictionary: Mapping[str, int], k: int) -> dict[str, set[str]]:
    """Build an inverted index from character k-grams to dictionary terms."""
    index: dict[str, set[str]] = defaultdict(set)
    for term in dictionary:
        for kg in kgrams(term, k):
            index[kg].add(term)
    return dict(index)


def correct_query(
    query: str,
    dictionary: Mapping[str, int],
    kgram_index: Mapping[str, set[str]],
    k: int,
    jaccard_min: float,
    max_edit: int,
) -> tuple[str, list[dict[str, Any]]]:
    """Correct unknown tokens in `query` using k-gram + Jaccard + OSA distance."""
    tokens = tokenize(query, lowercase=True)
    corrected_tokens: list[str] = []
    log: list[dict[str, Any]] = []

    for term in tokens:
        if term in dictionary:
            corrected_tokens.append(term)
            continue

        term_kgs = kgrams(term, k)
        candidates: set[str] = set()
        for kg in term_kgs:
            candidates.update(kgram_index.get(kg, ()))

        after_jaccard = [
            cand
            for cand in candidates
            if jaccard(term_kgs, kgrams(cand, k)) >= jaccard_min
        ]

        within_edit: list[tuple[int, int, str]] = []
        for cand in after_jaccard:
            dist = osa_distance(term, cand)
            if dist <= max_edit:
                within_edit.append((dist, -int(dictionary[cand]), cand))

        within_edit.sort()
        if within_edit:
            best_dist, _, chosen = within_edit[0]
            corrected_tokens.append(chosen)
            log.append(
                {
                    "term": term,
                    "n_kgram_candidates": len(candidates),
                    "n_after_jaccard": len(after_jaccard),
                    "chosen": chosen,
                    "edit_distance": best_dist,
                    "reason": "corrected",
                }
            )
        else:
            corrected_tokens.append(term)
            log.append(
                {
                    "term": term,
                    "n_kgram_candidates": len(candidates),
                    "n_after_jaccard": len(after_jaccard),
                    "chosen": None,
                    "edit_distance": None,
                    "reason": "no_candidate_within_max_edit",
                }
            )

    return " ".join(corrected_tokens), log


def _stable_seed(query_id: str) -> int:
    return int(hashlib.sha256(query_id.encode("utf-8")).hexdigest(), 16)


def _mutate_token(token: str, error_type: str, rng: random.Random) -> str | None:
    letters = string.ascii_lowercase
    if error_type == "insercao":
        pos = rng.randint(0, len(token))
        ch = rng.choice(letters)
        return token[:pos] + ch + token[pos:]
    if error_type == "remocao":
        if len(token) <= 1:
            return None
        pos = rng.randrange(len(token))
        return token[:pos] + token[pos + 1 :]
    if error_type == "substituicao":
        pos = rng.randrange(len(token))
        choices = [c for c in letters if c != token[pos]]
        ch = rng.choice(choices)
        return token[:pos] + ch + token[pos + 1 :]
    if error_type == "transposicao":
        swappable = [i for i in range(len(token) - 1) if token[i] != token[i + 1]]
        if not swappable:
            return None
        pos = rng.choice(swappable)
        return token[:pos] + token[pos + 1] + token[pos] + token[pos + 2 :]
    raise ValueError(f"Unknown error_type: {error_type!r}")


def introduce_typo(
    query_id: str,
    query_text: str,
    error_type: str,
    dictionary: Mapping[str, int],
    max_attempts: int = 50,
) -> tuple[str, str, str] | None:
    """Deterministically introduce one out-of-vocabulary typo of `error_type`."""
    tokens = tokenize(query_text, lowercase=True)
    candidates = [
        (idx, tok)
        for idx, tok in enumerate(tokens)
        if len(tok) >= 4 and tok not in STOPWORDS_EN and tok in dictionary
    ]
    candidates.sort(key=lambda item: (-len(item[1]), item[0]))

    rng = random.Random(_stable_seed(query_id))
    for idx, orig_term in candidates:
        for _ in range(max_attempts):
            mutated = _mutate_token(orig_term, error_type, rng)
            if mutated and mutated != orig_term and mutated not in dictionary:
                new_tokens = list(tokens)
                new_tokens[idx] = mutated
                return " ".join(new_tokens), orig_term, mutated
    return None


def _build_bm25(docs: Mapping[str, Mapping[str, str]], cfg: Mapping[str, Any]) -> BM25:
    pp = cfg["preprocessing"]
    processed = {
        doc_id: preprocess(
            doc_text(doc),
            lowercase=bool(pp["lowercase"]),
            remove_stopwords=bool(pp["remove_stopwords"]),
            stemmer=pp["stemmer"],
        )
        for doc_id, doc in docs.items()
    }
    index = InvertedIndex(processed)
    return BM25(index, k1=float(cfg["bm25"]["k1"]), b=float(cfg["bm25"]["b"]))


def _bm25_ndcg10(bm25: BM25, query_text: str, qrels_q: Mapping[str, int], cfg: Mapping[str, Any]) -> float:
    pp = cfg["preprocessing"]
    q_tokens = preprocess(
        query_text,
        lowercase=bool(pp["lowercase"]),
        remove_stopwords=bool(pp["remove_stopwords"]),
        stemmer=pp["stemmer"],
    )
    ranking = bm25.search(q_tokens, k=10)
    return score_query(ranking, qrels_q)["NDCG10"]


def _recovery_stats(orig: float, err: float, corr: float) -> tuple[str, str]:
    denom = orig - err
    if denom > 0.0:
        return f"{(corr - err) / denom:.17g}", "ok"
    return "", "erro_nao_reduziu"


def run_spell_dev_and_test_experiment(
    config_path: Path | str = Path("config/config.yaml"),
    artifacts_dir: Path | str = Path("artifacts"),
    results_dir: Path | str = Path("results"),
    data_root: Path | str = Path("data/nfcorpus"),
) -> dict[str, Any]:
    """Tune spell corrector on dev, freeze config, and run 10-query test typo experiment."""
    cfg_path = Path(config_path)
    art_dir = Path(artifacts_dir)
    res_dir = Path(results_dir)
    config = load_config(cfg_path)
    guard_stage("spell", split="dev", config=config, artifacts_dir=art_dir)

    dataset = load_dataset(data_root)
    dictionary = surface_vocab(dataset.docs)
    bm25 = _build_bm25(dataset.docs, config)

    # 1. Dev grid search
    dev_cases: list[tuple[str, str, str, str, str]] = []
    for idx, qid in enumerate(sorted(dataset.query_ids["dev"])):
        q_text = dataset.queries[qid]
        if _bm25_ndcg10(bm25, q_text, dataset.qrels["dev"][qid], config) <= 0.0:
            continue
        err_type = ERROR_TYPES[len(dev_cases) % len(ERROR_TYPES)]
        typo = introduce_typo(qid, q_text, err_type, dictionary)
        if typo is not None:
            q_err, orig_term, err_term = typo
            dev_cases.append((qid, q_text, q_err, orig_term, err_term))

    kgram_indexes = {k: build_kgram_index(dictionary, k=k) for k in (2, 3)}
    grid_rows: list[dict[str, Any]] = []
    for k in (2, 3):
        idx_k = kgram_indexes[k]
        for j_min in (0.2, 0.3, 0.4, 0.5):
            for max_ed in (1, 2):
                n_correct = 0
                ndcg_sum = 0.0
                for qid, _, q_err, orig_term, _ in dev_cases:
                    q_corr, log = correct_query(
                        q_err,
                        dictionary=dictionary,
                        kgram_index=idx_k,
                        k=k,
                        jaccard_min=j_min,
                        max_edit=max_ed,
                    )
                    if log and log[0]["chosen"] == orig_term:
                        n_correct += 1
                    ndcg_sum += _bm25_ndcg10(bm25, q_corr, dataset.qrels["dev"][qid], config)

                acc = n_correct / len(dev_cases)
                ndcg_mean = ndcg_sum / len(dev_cases)
                grid_rows.append(
                    {
                        "k": k,
                        "jaccard_min": j_min,
                        "max_edit": max_ed,
                        "correction_accuracy": f"{acc:.6f}",
                        "ndcg10_corr_mean": f"{ndcg_mean:.6f}",
                        "n_queries": len(dev_cases),
                    }
                )

    grid_csv = art_dir / "dev" / "spell_grid_dev.csv"
    grid_csv.parent.mkdir(parents=True, exist_ok=True)
    with grid_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "k",
                "jaccard_min",
                "max_edit",
                "correction_accuracy",
                "ndcg10_corr_mean",
                "n_queries",
            ],
        )
        writer.writeheader()
        writer.writerows(grid_rows)

    best = min(
        grid_rows,
        key=lambda r: (
            -float(r["correction_accuracy"]),
            -float(r["ndcg10_corr_mean"]),
            int(r["k"]),
            float(r["jaccard_min"]),
            int(r["max_edit"]),
        ),
    )
    best_k = int(best["k"])
    best_j = float(best["jaccard_min"])
    best_ed = int(best["max_edit"])

    config["spell"]["k"] = best_k
    config["spell"]["jaccard_min"] = best_j
    config["spell"]["max_edit"] = best_ed
    config["spell"]["distance"] = "osa"
    config["frozen"] = True
    save_config(config, cfg_path)
    freeze_config(config, artifacts_dir=art_dir)
    record_stage_result("spell", grid_csv, config=config, artifacts_dir=art_dir)

    # 2. Test typo experiment (10 queries, lexical + semantic)
    guard_stage("typo", split="test", config=config, artifacts_dir=art_dir)
    eligible_test = [
        qid
        for qid in sorted(dataset.query_ids["test"])
        if _bm25_ndcg10(bm25, dataset.queries[qid], dataset.qrels["test"][qid], config) > 0.0
        and any(
            len(tok) >= 4 and tok not in STOPWORDS_EN and tok in dictionary
            for tok in tokenize(dataset.queries[qid], lowercase=True)
        )
    ]
    sampled_test = random.Random(42).sample(sorted(eligible_test), 10)

    dense_cfg = config["dense"]
    emb_dir = art_dir / "embeddings" / model_slug(dense_cfg["model_id"])
    dense = DenseRetriever.build_or_load(
        dataset.docs,
        embeddings_dir=emb_dir,
        model_id=dense_cfg["model_id"],
        revision=dense_cfg.get("revision"),
        max_seq_length=int(dense_cfg["max_seq_length"]),
        batch_size=int(dense_cfg["batch_size"]),
        torch_threads=int(config["runtime"]["torch_threads"]),
    )

    typo_rows: list[dict[str, Any]] = []
    typo_logs: list[dict[str, Any]] = []
    idx_best = kgram_indexes[best_k]

    for i, qid in enumerate(sampled_test):
        err_type = ERROR_TYPES[i % len(ERROR_TYPES)]
        q_orig = dataset.queries[qid]
        typo = introduce_typo(qid, q_orig, err_type, dictionary)
        if typo is None:
            raise RuntimeError(f"Could not introduce typo for test query {qid!r}")
        q_err, orig_term, err_term = typo
        q_corr, log = correct_query(
            q_err,
            dictionary=dictionary,
            kgram_index=idx_best,
            k=best_k,
            jaccard_min=best_j,
            max_edit=best_ed,
        )
        typo_logs.append({"query_id": qid, "tipo_erro": err_type, "log": log})

        qrels_q = dataset.qrels["test"][qid]
        lex_orig = _bm25_ndcg10(bm25, q_orig, qrels_q, config)
        lex_err = _bm25_ndcg10(bm25, q_err, qrels_q, config)
        lex_corr = _bm25_ndcg10(bm25, q_corr, qrels_q, config)
        rec_lex, status_lex = _recovery_stats(lex_orig, lex_err, lex_corr)

        sem_orig = score_query(dense.search(q_orig, k=100), qrels_q)["NDCG10"]
        sem_err = score_query(dense.search(q_err, k=100), qrels_q)["NDCG10"]
        sem_corr = score_query(dense.search(q_corr, k=100), qrels_q)["NDCG10"]
        rec_sem, status_sem = _recovery_stats(sem_orig, sem_err, sem_corr)

        corr_term = log[0]["chosen"] if log else None
        typo_rows.append(
            {
                "query_id": qid,
                "query_original": q_orig,
                "query_erro": q_err,
                "query_corrigida": q_corr,
                "tipo_erro": err_type,
                "NDCG10_original": f"{lex_orig:.17g}",
                "NDCG10_erro": f"{lex_err:.17g}",
                "NDCG10_corrigida": f"{lex_corr:.17g}",
                "termo_original": orig_term,
                "termo_erro": err_term,
                "termo_corrigido": corr_term or "",
                "recuperacao": rec_lex,
                "status": status_lex,
                "NDCG10_sem_original": f"{sem_orig:.17g}",
                "NDCG10_sem_erro": f"{sem_err:.17g}",
                "NDCG10_sem_corrigida": f"{sem_corr:.17g}",
                "recuperacao_sem": rec_sem,
                "status_sem": status_sem,
            }
        )

    res_dir.mkdir(parents=True, exist_ok=True)
    typo_csv = res_dir / "typo_experiment.csv"
    fieldnames = [
        "query_id",
        "query_original",
        "query_erro",
        "query_corrigida",
        "tipo_erro",
        "NDCG10_original",
        "NDCG10_erro",
        "NDCG10_corrigida",
        "termo_original",
        "termo_erro",
        "termo_corrigido",
        "recuperacao",
        "status",
        "NDCG10_sem_original",
        "NDCG10_sem_erro",
        "NDCG10_sem_corrigida",
        "recuperacao_sem",
        "status_sem",
    ]
    with typo_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(typo_rows)

    log_json = art_dir / "typo" / "log.json"
    log_json.parent.mkdir(parents=True, exist_ok=True)
    log_json.write_text(json.dumps(typo_logs, indent=2), encoding="utf-8")

    # Robustness summary
    rob_rows = []
    for sys_name, orig_col, err_col, rec_col, st_col in (
        ("lexical", "NDCG10_original", "NDCG10_erro", "recuperacao", "status"),
        ("semantic", "NDCG10_sem_original", "NDCG10_sem_erro", "recuperacao_sem", "status_sem"),
    ):
        drops = [float(r[orig_col]) - float(r[err_col]) for r in typo_rows]
        ok_recs = [float(r[rec_col]) for r in typo_rows if r[st_col] == "ok"]
        n_ok = len(ok_recs)
        n_no_drop = len(typo_rows) - n_ok
        rob_rows.append(
            {
                "system": sys_name,
                "queda_media": f"{(sum(drops) / len(drops)):.6f}",
                "recuperacao_media": f"{(sum(ok_recs) / n_ok):.6f}" if n_ok else "",
                "n_ok": n_ok,
                "n_erro_nao_reduziu": n_no_drop,
            }
        )

    rob_csv = art_dir / "typo" / "robustez_lex_sem.csv"
    with rob_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["system", "queda_media", "recuperacao_media", "n_ok", "n_erro_nao_reduziu"],
        )
        writer.writeheader()
        writer.writerows(rob_rows)

    record_stage_result("typo", typo_csv, config=config, artifacts_dir=art_dir)
    return {
        "best_spell": {"k": best_k, "jaccard_min": best_j, "max_edit": best_ed},
        "robustness": rob_rows,
    }


if __name__ == "__main__":
    summary = run_spell_dev_and_test_experiment()
    print(json.dumps(summary, indent=2))
