from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import pytest
import yaml

from src.preprocessing import tokenize
from src.spell import (
    ERROR_TYPES,
    build_kgram_index,
    correct_query,
    introduce_typo,
    jaccard,
    kgrams,
    osa_distance,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = REPO_ROOT / "artifacts"
RESULTS_DIR = REPO_ROOT / "results"
CONFIG_PATH = REPO_ROOT / "config" / "config.yaml"


def test_kgrams():
    assert kgrams("casa", 2) == {"$c", "ca", "as", "sa", "a$"}
    assert kgrams("casa", 3) == {"$ca", "cas", "asa", "sa$"}


def test_jaccard_fixed_sets():
    assert jaccard({"a", "b", "c"}, {"b", "c", "d"}) == pytest.approx(0.5)
    assert jaccard(set(), set()) == 0.0
    assert jaccard({"a"}, {"b"}) == 0.0


def test_osa_distance():
    assert osa_distance("ab", "ba") == 1
    assert osa_distance("abc", "abd") == 1
    assert osa_distance("abc", "abcd") == 1
    assert osa_distance("", "abc") == 3
    assert osa_distance("abc", "abc") == 0


def test_build_kgram_index():
    dictionary = Counter({"casa": 10, "caso": 5, "mesa": 3})
    idx = build_kgram_index(dictionary, k=2)
    for term in dictionary:
        for kg in kgrams(term, 2):
            assert term in idx[kg]


def test_correct_all_four_error_types_and_known_unchanged():
    dictionary = Counter({"omega": 50, "cancer": 80, "diet": 40, "nutrition": 30})
    idx = build_kgram_index(dictionary, k=2)

    cases = [
        ("insercao", "diet dieet", "diet diet"),
        ("remocao", "diet cance", "diet cancer"),
        ("substituicao", "diet omaga", "diet omega"),
        ("transposicao", "diet deit", "diet diet"),
    ]
    for _, query_err, expected in cases:
        corrected, log = correct_query(
            query_err,
            dictionary=dictionary,
            kgram_index=idx,
            k=2,
            jaccard_min=0.2,
            max_edit=2,
        )
        assert corrected == expected
        assert len(log) == 1
        assert log[0]["chosen"] is not None

    unchanged, log_known = correct_query(
        "omega diet",
        dictionary=dictionary,
        kgram_index=idx,
        k=2,
        jaccard_min=0.2,
        max_edit=2,
    )
    assert unchanged == "omega diet"
    assert log_known == []


def test_deterministic_error_generator():
    dictionary = Counter({"omega": 50, "cancer": 80, "diet": 40, "nutrition": 30})
    for err_type in ERROR_TYPES:
        res1 = introduce_typo("Q-123", "omega and nutrition for cancer", err_type, dictionary)
        res2 = introduce_typo("Q-123", "omega and nutrition for cancer", err_type, dictionary)
        assert res1 == res2
        assert res1 is not None
        query_err, orig_term, err_term = res1
        assert err_term not in dictionary
        assert err_term in tokenize(query_err)
        assert orig_term in tokenize("omega and nutrition for cancer")


@pytest.mark.data
def test_spell_grid_dev_and_frozen_config():
    grid_path = ARTIFACTS_DIR / "dev" / "spell_grid_dev.csv"
    assert grid_path.exists()

    with grid_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 16
    assert set(rows[0].keys()) == {
        "k",
        "jaccard_min",
        "max_edit",
        "correction_accuracy",
        "ndcg10_corr_mean",
        "n_queries",
    }

    best = min(
        rows,
        key=lambda r: (
            -float(r["correction_accuracy"]),
            -float(r["ndcg10_corr_mean"]),
            int(r["k"]),
            float(r["jaccard_min"]),
            int(r["max_edit"]),
        ),
    )
    cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    assert cfg["frozen"] is True
    assert int(cfg["spell"]["k"]) == int(best["k"])
    assert float(cfg["spell"]["jaccard_min"]) == pytest.approx(float(best["jaccard_min"]))
    assert int(cfg["spell"]["max_edit"]) == int(best["max_edit"])


@pytest.mark.data
def test_typo_experiment_and_robustness_artifacts():
    typo_path = RESULTS_DIR / "typo_experiment.csv"
    rob_path = ARTIFACTS_DIR / "typo" / "robustez_lex_sem.csv"
    log_path = ARTIFACTS_DIR / "typo" / "log.json"

    assert typo_path.exists()
    assert rob_path.exists()
    assert log_path.exists()

    with typo_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 10
    assert {r["tipo_erro"] for r in rows} == set(ERROR_TYPES)

    for r in rows:
        orig_lex = float(r["NDCG10_original"])
        err_lex = float(r["NDCG10_erro"])
        corr_lex = float(r["NDCG10_corrigida"])
        assert orig_lex > 0.0

        denom = orig_lex - err_lex
        if denom > 0.0:
            assert r["status"] == "ok"
            expected_rec = (corr_lex - err_lex) / denom
            assert float(r["recuperacao"]) == pytest.approx(expected_rec, abs=1e-12)
        else:
            assert r["status"] == "erro_nao_reduziu"
            assert r["recuperacao"] == ""

        orig_sem = float(r["NDCG10_sem_original"])
        err_sem = float(r["NDCG10_sem_erro"])
        corr_sem = float(r["NDCG10_sem_corrigida"])
        denom_sem = orig_sem - err_sem
        if denom_sem > 0.0:
            assert r["status_sem"] == "ok"
            expected_rec_sem = (corr_sem - err_sem) / denom_sem
            assert float(r["recuperacao_sem"]) == pytest.approx(expected_rec_sem, abs=1e-12)
        else:
            assert r["status_sem"] == "erro_nao_reduziu"
            assert r["recuperacao_sem"] == ""

    with rob_path.open("r", encoding="utf-8", newline="") as f:
        rob_rows = list(csv.DictReader(f))
    assert len(rob_rows) == 2
    assert {r["system"] for r in rob_rows} == {"lexical", "semantic"}
