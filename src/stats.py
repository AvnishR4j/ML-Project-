"""Small statistics helpers: bootstrap confidence intervals and McNemar's test."""
import numpy as np
from scipy.stats import chi2
from sklearn.metrics import f1_score, accuracy_score


def bootstrap_ci(y, pred, metric="macro_f1", n=1000, seed=0, **kw):
    """95% percentile bootstrap CI for accuracy / macro-F1 / positive-class F1 on one fixed test set."""
    y, pred = np.asarray(y), np.asarray(pred)
    rng = np.random.default_rng(seed)
    fn = {"macro_f1": lambda a, b: f1_score(a, b, average="macro"),
          "accuracy": accuracy_score,
          "f1": lambda a, b: f1_score(a, b, **kw)}[metric]
    vals = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        vals.append(fn(y[i], pred[i]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def mcnemar(y, pred_a, pred_b):
    """McNemar test (continuity-corrected) on paired correctness. Returns (b, c, p_value)."""
    y, a, b = np.asarray(y), np.asarray(pred_a), np.asarray(pred_b)
    ok_a, ok_b = a == y, b == y
    n01 = int((ok_a & ~ok_b).sum())      # A right, B wrong
    n10 = int((~ok_a & ok_b).sum())      # A wrong, B right
    if n01 + n10 == 0:
        return n01, n10, 1.0
    stat = (abs(n01 - n10) - 1) ** 2 / (n01 + n10)
    return n01, n10, float(chi2.sf(stat, 1))
