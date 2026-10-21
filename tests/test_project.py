"""Run:  pytest -q      (needs the data and trained models: python run_all.py first)"""
import sys
from pathlib import Path
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


# ---------------------------------------------------------------- data hygiene: no leakage between train and test
def _overlap(a, b):
    norm = lambda s: " ".join(str(s).lower().split())
    return len({norm(x) for x in a} & {norm(x) for x in b})


def test_spam_split_has_no_duplicates_across_sets():
    from data import load_spam
    tr, te, dupes = load_spam()
    assert dupes > 300                                   # the raw file really contains ~400 exact duplicates
    assert _overlap(tr.text, te.text) == 0


def test_hinglish_sentiment_split_has_no_overlap_and_expected_size():
    from data2 import load_hinglish_sentiment
    tr, te = load_hinglish_sentiment()
    assert _overlap(tr.text, te.text) == 0
    assert set(tr.label) == {"negative", "neutral", "positive"}
    assert 3000 < len(tr) < 3200 and 700 < len(te) < 850


def test_english_sentiment_splits_have_no_overlap():
    from data2 import load_english_sentiment
    for name, (tr, te) in load_english_sentiment().items():
        assert _overlap(tr.text, te.text) == 0, name


def test_scam_challenge_is_deduplicated():
    from data2 import load_scam_challenge
    for name, d in load_scam_challenge().items():
        assert d.text.is_unique and len(d) < 120, name  # raw file has 5,000 rows but ~101 unique lines


def test_language_train_and_test_do_not_overlap():
    from data import load_language_v2
    tr, tests = load_language_v2()
    for name, t in tests.items():
        if name.startswith("hand-written"):
            continue                                     # separate, never-trained OOD lines
        assert _overlap(tr.text, t.text) <= 2, name


# ---------------------------------------------------------------- text normalisation + features
def test_normalizer_undoes_spacing_and_homoglyphs():
    from text_norm import attack_spacing, attack_homoglyph, normalize
    s = "free prize waiting claim now"
    assert normalize(attack_spacing(s)).split() == s.split()
    assert normalize(attack_homoglyph(s)) == s


def test_structural_features_fire_on_scam_cues():
    from spam_features import structural_matrix
    f = structural_matrix(["WIN Rs 5000 free prize now!! call 9876543210 http://x.com", "see you at lunch"], lexicon=True)
    assert f[0][:4].sum() >= 3 and f[1][:4].sum() == 0
    assert f[0][-1] > 0 and f[1][-1] == 0                # lexicon feature


def test_stats_helpers():
    from stats import bootstrap_ci, mcnemar
    y = np.array([0, 1] * 100); p = y.copy(); p[:10] = 1 - p[:10]
    lo, hi = bootstrap_ci(y, p, "accuracy")
    assert lo < 0.95 < hi
    assert mcnemar(y, p, p)[2] == 1.0


# ---------------------------------------------------------------- deployed pipeline behaviour
@pytest.fixture(scope="module")
def models_ready():
    need = ["language_final", "spam_final", "sentiment", "hinglish_sentiment", "theme_en", "theme_hinglish"]
    if not all((ROOT / "models" / f"{m}.joblib").exists() for m in need):
        pytest.skip("run python run_all.py first")


def test_english_genuine_message(models_ready):
    from pipeline import analyze
    r = analyze("My order has not arrived yet and nobody is replying to my emails")
    assert r["language"] == "en" and r["spam"] == "ham" and r["sentiment"] in ("positive", "negative") and "cluster" in r


def test_english_spam_and_evasion(models_ready):
    from pipeline import analyze
    from text_norm import attack_spacing
    t = "Congratulations! You have won a free prize, call 09061701461 now to claim"
    assert analyze(t)["spam"] == "spam"
    assert analyze(attack_spacing(t))["spam"] == "spam"


def test_hinglish_routes_to_hinglish_sentiment(models_ready):
    from pipeline import analyze
    r = analyze("mera order abhi tak nahi pahuncha, bahut gussa aa raha hai")
    assert r["language"] == "hinglish" and r["sentiment"] in ("negative", "neutral", "positive") and r["spam"] == "ham"


def test_other_language_is_not_sent_to_spam_model(models_ready):
    from pipeline import analyze
    r = analyze("Der Film war wirklich gut, ich habe ihn sehr genossen")
    assert r["language"] in ("de", "uncertain") and r.get("spam") is None


def test_uncertain_language_is_flagged_for_review(models_ready):
    from pipeline import analyze
    r = analyze("Merci beaucoup pour votre aide, a bientot")
    assert r["language"] == "uncertain" and r["needs_review"] is True


def test_oracle_language_forces_route(models_ready):
    from pipeline import analyze
    assert analyze("hello there", force_language="de")["language"] == "de"


# ---------------------------------------------------------------- recorded results satisfy basic sanity bounds
def test_recorded_results_are_sane():
    import json
    r = ROOT / "results"
    if not (r / "e2e_summary.json").exists():
        pytest.skip("no results yet")
    e = json.loads((r / "e2e_summary.json").read_text())
    assert e["language"]["accuracy_on_accepted"] > 0.95
    assert e["spam"]["english_predicted_routing"]["recall"] > 0.9
    s = json.loads((r / "sentiment_summary.json").read_text())
    assert s["hinglish"]["test_macro_F1"] > s["hinglish"]["majority_baseline_macroF1"] + 0.3
