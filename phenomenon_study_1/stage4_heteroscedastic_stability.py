"""阶段四：异方差高斯下稳定性测试，比较欧氏困难度与决策边界困难度。"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def make_heteroscedastic(seed: int, n_per_class: int, mean_distance: float, var_scale_1: float):
    rng = np.random.default_rng(seed)
    half = mean_distance / 2.0
    mu0 = np.array([-half, 0.0], dtype=np.float64)
    mu1 = np.array([half, 0.2], dtype=np.float64)

    cov0 = np.array([[1.0, 0.25], [0.25, 0.75]], dtype=np.float64)
    cov1_base = np.array([[0.8, -0.18], [-0.18, 1.25]], dtype=np.float64)
    cov1 = cov1_base * var_scale_1

    x0 = rng.multivariate_normal(mu0, cov0, size=n_per_class)
    x1 = rng.multivariate_normal(mu1, cov1, size=n_per_class)
    x = np.vstack([x0, x1])
    y = np.hstack([np.zeros(n_per_class, dtype=int), np.ones(n_per_class, dtype=int)])
    return x, y, mu0, mu1


def boundary_mask_approx(x: np.ndarray, mu0: np.ndarray, mu1: np.ndarray, band: float):
    # 异方差下真实边界是二次曲线，这里用最近原型差近似边界带用于统一评估
    d = cdist(x, np.vstack([mu0, mu1]))
    gap = np.abs(d[:, 0] - d[:, 1])
    return gap <= band


def euclidean_difficulty(x: np.ndarray, y: np.ndarray, mu0: np.ndarray, mu1: np.ndarray):
    d = cdist(x, np.vstack([mu0, mu1]))
    return d[np.arange(len(x)), y]


def margin_difficulty(x: np.ndarray, y: np.ndarray, seed: int):
    clf = LogisticRegression(random_state=seed, solver="lbfgs", max_iter=700)
    clf.fit(x, y)
    p = clf.predict_proba(x)[:, 1]
    margin = np.abs(p - 0.5) * 2.0
    return 1.0 - margin


def weighted_train_eval(x_tr, y_tr, x_te, y_te, bmask_te, score, topk_ratio, up_weight, seed):
    w = np.ones(len(y_tr), dtype=np.float64)
    k = max(5, int(len(y_tr) * topk_ratio))
    idx = np.argsort(score)[-k:]
    w[idx] = up_weight

    clf = LogisticRegression(random_state=seed, solver="lbfgs", max_iter=700)
    clf.fit(x_tr, y_tr, sample_weight=w)
    pred = clf.predict(x_te)
    overall = accuracy_score(y_te, pred)
    boundary = accuracy_score(y_te[bmask_te], pred[bmask_te]) if np.any(bmask_te) else float("nan")
    return float(overall), float(boundary), idx


def main():
    parser = argparse.ArgumentParser(description="Stage 4: heteroscedastic stability")
    parser.add_argument("--output-dir", type=str, default="phenomenon_study_1/outputs/stage4")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--n-per-class", type=int, default=1200)
    parser.add_argument("--mean-distance", type=float, default=2.0)
    parser.add_argument("--var-scales", type=float, nargs="+", default=[0.8, 1.0, 1.3, 1.6])
    parser.add_argument("--boundary-band", type=float, default=0.30)
    parser.add_argument("--topk-ratio", type=float, default=0.15)
    parser.add_argument("--up-weight", type=float, default=3.0)
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    for scale in args.var_scales:
        for seed in args.seeds:
            x, y, mu0, mu1 = make_heteroscedastic(seed, args.n_per_class, args.mean_distance, scale)
            x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size=0.35, random_state=seed, stratify=y)
            bmask_te = boundary_mask_approx(x_te, mu0, mu1, args.boundary_band)
            bmask_tr = boundary_mask_approx(x_tr, mu0, mu1, args.boundary_band)

            base = LogisticRegression(random_state=seed, solver="lbfgs", max_iter=700)
            base.fit(x_tr, y_tr)
            pred_b = base.predict(x_te)
            base_overall = accuracy_score(y_te, pred_b)
            base_boundary = accuracy_score(y_te[bmask_te], pred_b[bmask_te]) if np.any(bmask_te) else float("nan")

            se = euclidean_difficulty(x_tr, y_tr, mu0, mu1)
            oe, be, idx_e = weighted_train_eval(
                x_tr, y_tr, x_te, y_te, bmask_te, se, args.topk_ratio, args.up_weight, seed
            )

            sm = margin_difficulty(x_tr, y_tr, seed)
            om, bm, idx_m = weighted_train_eval(
                x_tr, y_tr, x_te, y_te, bmask_te, sm, args.topk_ratio, args.up_weight, seed
            )

            rows.append(
                {
                    "seed": seed,
                    "var_scale": scale,
                    "base_overall": float(base_overall),
                    "base_boundary": float(base_boundary),
                    "eu_overall": oe,
                    "eu_boundary": be,
                    "margin_overall": om,
                    "margin_boundary": bm,
                    "hard_precision_eu": float(np.mean(bmask_tr[idx_e])),
                    "hard_precision_margin": float(np.mean(bmask_tr[idx_m])),
                }
            )

    summary = {}
    for scale in args.var_scales:
        sub = [r for r in rows if r["var_scale"] == scale]
        summary[str(scale)] = {
            "hard_precision_eu": float(np.mean([r["hard_precision_eu"] for r in sub])),
            "hard_precision_margin": float(np.mean([r["hard_precision_margin"] for r in sub])),
            "delta_boundary_eu_vs_base": float(np.mean([r["eu_boundary"] - r["base_boundary"] for r in sub])),
            "delta_boundary_margin_vs_base": float(np.mean([r["margin_boundary"] - r["base_boundary"] for r in sub])),
            "delta_overall_eu_vs_base": float(np.mean([r["eu_overall"] - r["base_overall"] for r in sub])),
            "delta_overall_margin_vs_base": float(np.mean([r["margin_overall"] - r["base_overall"] for r in sub])),
        }

    # 可视化
    xvals = args.var_scales
    pe = [summary[str(v)]["hard_precision_eu"] for v in xvals]
    pm = [summary[str(v)]["hard_precision_margin"] for v in xvals]
    dbe = [summary[str(v)]["delta_boundary_eu_vs_base"] for v in xvals]
    dbm = [summary[str(v)]["delta_boundary_margin_vs_base"] for v in xvals]

    fig, ax = plt.subplots(1, 2, figsize=(10, 4), dpi=120)
    ax[0].plot(xvals, pe, "o-", label="euclidean")
    ax[0].plot(xvals, pm, "s-", label="margin")
    ax[0].set_title("Hard precision under heteroscedasticity")
    ax[0].set_xlabel("class-1 variance scale")
    ax[0].grid(alpha=0.25)
    ax[0].legend()

    ax[1].plot(xvals, dbe, "o-", label="euclidean")
    ax[1].plot(xvals, dbm, "s-", label="margin")
    ax[1].axhline(0, color="gray", linestyle="--", linewidth=1)
    ax[1].set_title("Boundary gain over baseline")
    ax[1].set_xlabel("class-1 variance scale")
    ax[1].grid(alpha=0.25)
    ax[1].legend()

    fig.tight_layout()
    fig.savefig(out / "heteroscedastic_stability.png")
    plt.close(fig)

    with open(out / "trials.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    with open(out / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
