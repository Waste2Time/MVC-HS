"""阶段二：在训练中对 top-k 困难样本重加权，比较欧氏困难度与边界 margin 困难度的训练价值。"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def make_data(seed: int, n_per_class: int, mean_distance: float):
    rng = np.random.default_rng(seed)
    half = mean_distance / 2.0
    mu0 = np.array([-half, 0.0], dtype=np.float64)
    mu1 = np.array([half, 0.0], dtype=np.float64)
    cov = np.array([[1.0, 0.2], [0.2, 1.0]], dtype=np.float64)

    x0 = rng.multivariate_normal(mu0, cov, size=n_per_class)
    x1 = rng.multivariate_normal(mu1, cov, size=n_per_class)
    x = np.vstack([x0, x1])
    y = np.hstack([np.zeros(n_per_class, dtype=int), np.ones(n_per_class, dtype=int)])
    return x, y, mu0, mu1


def boundary_mask(x: np.ndarray, mu0: np.ndarray, mu1: np.ndarray, band: float):
    normal = mu1 - mu0
    midpoint = 0.5 * (mu0 + mu1)
    signed = (x - midpoint) @ normal / (np.linalg.norm(normal) + 1e-12)
    return np.abs(signed) <= band


def euclidean_difficulty(x: np.ndarray, y: np.ndarray, mu0: np.ndarray, mu1: np.ndarray):
    prototypes = np.vstack([mu0, mu1])
    d = cdist(x, prototypes)
    return d[np.arange(len(x)), y]


def margin_difficulty(x: np.ndarray, y: np.ndarray, seed: int):
    clf = LogisticRegression(random_state=seed, solver="lbfgs")
    clf.fit(x, y)
    prob = clf.predict_proba(x)[:, 1]
    margin = np.abs(prob - 0.5) * 2.0
    return 1.0 - margin


def train_weighted_lr(x, y, sample_weight, seed):
    clf = LogisticRegression(random_state=seed, solver="lbfgs", max_iter=500)
    clf.fit(x, y, sample_weight=sample_weight)
    return clf


def evaluate_model(clf, x_test, y_test, bmask_test):
    pred = clf.predict(x_test)
    overall = accuracy_score(y_test, pred)
    if np.any(bmask_test):
        boundary = accuracy_score(y_test[bmask_test], pred[bmask_test])
    else:
        boundary = float("nan")
    return float(overall), float(boundary)


def make_weights(score: np.ndarray, topk_ratio: float, up_weight: float):
    w = np.ones(len(score), dtype=np.float64)
    k = max(5, int(len(score) * topk_ratio))
    idx = np.argsort(score)[-k:]
    w[idx] = up_weight
    return w, idx


def main():
    parser = argparse.ArgumentParser(description="Stage 2: hard-sample reweighting in training")
    parser.add_argument("--output-dir", type=str, default="phenomenon_study_1/outputs/stage2")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--n-per-class", type=int, default=1200)
    parser.add_argument("--mean-distance", type=float, default=2.2)
    parser.add_argument("--boundary-band", type=float, default=0.25)
    parser.add_argument("--topk-ratio", type=float, default=0.15)
    parser.add_argument("--up-weight", type=float, default=3.0)
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    for seed in args.seeds:
        x, y, mu0, mu1 = make_data(seed, args.n_per_class, args.mean_distance)
        x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size=0.35, random_state=seed, stratify=y)

        bmask_te = boundary_mask(x_te, mu0, mu1, args.boundary_band)

        # baseline
        clf_base = train_weighted_lr(x_tr, y_tr, sample_weight=np.ones(len(y_tr)), seed=seed)
        base_overall, base_boundary = evaluate_model(clf_base, x_te, y_te, bmask_te)

        # euclidean-hard reweight
        score_e = euclidean_difficulty(x_tr, y_tr, mu0, mu1)
        w_e, idx_e = make_weights(score_e, args.topk_ratio, args.up_weight)
        clf_e = train_weighted_lr(x_tr, y_tr, sample_weight=w_e, seed=seed)
        e_overall, e_boundary = evaluate_model(clf_e, x_te, y_te, bmask_te)

        # boundary-margin reweight
        score_m = margin_difficulty(x_tr, y_tr, seed)
        w_m, idx_m = make_weights(score_m, args.topk_ratio, args.up_weight)
        clf_m = train_weighted_lr(x_tr, y_tr, sample_weight=w_m, seed=seed)
        m_overall, m_boundary = evaluate_model(clf_m, x_te, y_te, bmask_te)

        hard_tr = boundary_mask(x_tr, mu0, mu1, args.boundary_band)

        rows.append(
            {
                "seed": seed,
                "base_overall": base_overall,
                "base_boundary": base_boundary,
                "eu_overall": e_overall,
                "eu_boundary": e_boundary,
                "margin_overall": m_overall,
                "margin_boundary": m_boundary,
                "hard_precision_eu": float(np.mean(hard_tr[idx_e])),
                "hard_precision_margin": float(np.mean(hard_tr[idx_m])),
            }
        )

    summary = {
        "overall_base": float(np.mean([r["base_overall"] for r in rows])),
        "overall_eu": float(np.mean([r["eu_overall"] for r in rows])),
        "overall_margin": float(np.mean([r["margin_overall"] for r in rows])),
        "boundary_base": float(np.mean([r["base_boundary"] for r in rows])),
        "boundary_eu": float(np.mean([r["eu_boundary"] for r in rows])),
        "boundary_margin": float(np.mean([r["margin_boundary"] for r in rows])),
        "hard_precision_eu": float(np.mean([r["hard_precision_eu"] for r in rows])),
        "hard_precision_margin": float(np.mean([r["hard_precision_margin"] for r in rows])),
    }

    with open(out / "trials.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    with open(out / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
