"""阶段一：最简二分类高斯设定下，对比两类困难度排序对真实边界困难样本的识别能力。"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.linear_model import LogisticRegression


def make_homoscedastic_gaussian(seed: int, n_per_class: int, mean_distance: float):
    rng = np.random.default_rng(seed)
    half = mean_distance / 2.0
    mu0 = np.array([-half, 0.0], dtype=np.float64)
    mu1 = np.array([half, 0.0], dtype=np.float64)
    cov = np.array([[1.0, 0.15], [0.15, 1.0]], dtype=np.float64)

    x0 = rng.multivariate_normal(mu0, cov, size=n_per_class)
    x1 = rng.multivariate_normal(mu1, cov, size=n_per_class)
    x = np.vstack([x0, x1])
    y = np.hstack([np.zeros(n_per_class, dtype=int), np.ones(n_per_class, dtype=int)])
    return x, y, mu0, mu1


def true_boundary_hard_mask(x: np.ndarray, mu0: np.ndarray, mu1: np.ndarray, band: float):
    # 同协方差设定下，真实边界是两个均值中垂面；用点到边界有符号距离绝对值定义困难度
    normal = mu1 - mu0
    midpoint = 0.5 * (mu0 + mu1)
    signed_dist = (x - midpoint) @ normal / (np.linalg.norm(normal) + 1e-12)
    return np.abs(signed_dist) <= band


def euclidean_difficulty(x: np.ndarray, y: np.ndarray, mu0: np.ndarray, mu1: np.ndarray):
    # 按“到真实类别 prototype 的距离”定义困难度（越远越难）
    prototypes = np.vstack([mu0, mu1])
    d = cdist(x, prototypes)
    return d[np.arange(len(x)), y]


def margin_difficulty(x: np.ndarray, y: np.ndarray, seed: int):
    clf = LogisticRegression(random_state=seed, solver="lbfgs")
    clf.fit(x, y)
    prob = clf.predict_proba(x)[:, 1]
    margin = np.abs(prob - 0.5) * 2.0
    return 1.0 - margin


def precision_at_k(score: np.ndarray, hard_mask: np.ndarray, k: int):
    idx = np.argsort(score)[-k:]
    return float(np.mean(hard_mask[idx]))


def main():
    parser = argparse.ArgumentParser(description="Stage 1: boundary-dominant hard identification")
    parser.add_argument("--output-dir", type=str, default="phenomenon_study_1/outputs/stage1")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--n-per-class", type=int, default=1000)
    parser.add_argument("--mean-distance", type=float, default=2.4)
    parser.add_argument("--boundary-band", type=float, default=0.25)
    parser.add_argument("--topk-ratio", type=float, default=0.15)
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    for seed in args.seeds:
        x, y, mu0, mu1 = make_homoscedastic_gaussian(seed, args.n_per_class, args.mean_distance)
        hard_mask = true_boundary_hard_mask(x, mu0, mu1, args.boundary_band)
        k = max(5, int(len(x) * args.topk_ratio))

        s_euc = euclidean_difficulty(x, y, mu0, mu1)
        s_mar = margin_difficulty(x, y, seed)

        rows.append(
            {
                "seed": seed,
                "hard_ratio": float(np.mean(hard_mask)),
                "precision_euclidean": precision_at_k(s_euc, hard_mask, k),
                "precision_margin": precision_at_k(s_mar, hard_mask, k),
            }
        )

    summary = {
        "mean_hard_ratio": float(np.mean([r["hard_ratio"] for r in rows])),
        "mean_precision_euclidean": float(np.mean([r["precision_euclidean"] for r in rows])),
        "mean_precision_margin": float(np.mean([r["precision_margin"] for r in rows])),
    }

    with open(out / "trials.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    with open(out / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
