"""阶段二扩展：在线性 LR 外加入小 MLP classifier 对照，检验 hard sample 价值是否被模型能力压制。"""

import argparse
import json
from pathlib import Path
from statistics import mean, pstdev

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier


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
    d = cdist(x, np.vstack([mu0, mu1]))
    return d[np.arange(len(x)), y]


def make_weights(score: np.ndarray, topk_ratio: float, up_weight: float):
    w = np.ones(len(score), dtype=np.float64)
    k = max(5, int(len(score) * topk_ratio))
    idx = np.argsort(score)[-k:]
    w[idx] = up_weight
    return w, idx


def expand_by_weights(x: np.ndarray, y: np.ndarray, w: np.ndarray):
    """用重复采样模拟样本加权，避免不同模型 sample_weight 支持不一致。"""
    repeat = np.clip(np.rint(w).astype(int), 1, None)
    x_exp = np.repeat(x, repeat, axis=0)
    y_exp = np.repeat(y, repeat, axis=0)
    return x_exp, y_exp


def build_model(model_type: str, seed: int):
    if model_type == "lr":
        return LogisticRegression(random_state=seed, solver="lbfgs", max_iter=500)
    if model_type == "mlp":
        return MLPClassifier(
            hidden_layer_sizes=(32, 16),
            activation="relu",
            solver="adam",
            alpha=1e-4,
            learning_rate_init=1e-3,
            batch_size=128,
            max_iter=500,
            random_state=seed,
        )
    raise ValueError(f"Unsupported model_type: {model_type}")


def model_margin_difficulty(x: np.ndarray, y: np.ndarray, model_type: str, seed: int):
    """用各自模型的预测 margin 定义困难度（而不是统一用 LR）。"""
    model = build_model(model_type, seed)
    model.fit(x, y)
    prob = model.predict_proba(x)[:, 1]
    margin = np.abs(prob - 0.5) * 2.0
    return 1.0 - margin


def train_eval(
    model_type: str,
    x_tr: np.ndarray,
    y_tr: np.ndarray,
    x_te: np.ndarray,
    y_te: np.ndarray,
    bmask_te: np.ndarray,
    w: np.ndarray,
    seed: int,
):
    model = build_model(model_type, seed)
    x_fit, y_fit = expand_by_weights(x_tr, y_tr, w)
    model.fit(x_fit, y_fit)
    pred = model.predict(x_te)
    overall = accuracy_score(y_te, pred)
    boundary = accuracy_score(y_te[bmask_te], pred[bmask_te]) if np.any(bmask_te) else float("nan")
    return float(overall), float(boundary)


def mean_std(values):
    values = list(values)
    if len(values) == 1:
        return {"mean": float(values[0]), "std": 0.0}
    return {"mean": float(mean(values)), "std": float(pstdev(values))}


def paired_advantage(values_margin, values_eu):
    diffs = [m - e for m, e in zip(values_margin, values_eu)]
    return {
        "mean": float(mean(diffs)),
        "std": float(pstdev(diffs)) if len(diffs) > 1 else 0.0,
        "positive_ratio": float(sum(d > 0 for d in diffs) / len(diffs)),
    }


def main():
    parser = argparse.ArgumentParser(description="Stage2 capacity control: LR vs small MLP")
    parser.add_argument("--output-dir", type=str, default="phenomenon_study_1/outputs/stage2_capacity_control")
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
        bmask_tr = boundary_mask(x_tr, mu0, mu1, args.boundary_band)
        bmask_te = boundary_mask(x_te, mu0, mu1, args.boundary_band)

        score_e = euclidean_difficulty(x_tr, y_tr, mu0, mu1)
        w_base = np.ones(len(y_tr), dtype=np.float64)
        w_e, idx_e = make_weights(score_e, args.topk_ratio, args.up_weight)

        for model_type in ["lr", "mlp"]:
            # 使用各自模型的 margin 定义 hard sample
            score_m = model_margin_difficulty(x_tr, y_tr, model_type=model_type, seed=seed)
            w_m, idx_m = make_weights(score_m, args.topk_ratio, args.up_weight)

            base_overall, base_boundary = train_eval(model_type, x_tr, y_tr, x_te, y_te, bmask_te, w_base, seed)
            eu_overall, eu_boundary = train_eval(model_type, x_tr, y_tr, x_te, y_te, bmask_te, w_e, seed)
            margin_overall, margin_boundary = train_eval(model_type, x_tr, y_tr, x_te, y_te, bmask_te, w_m, seed)

            rows.append(
                {
                    "seed": seed,
                    "model": model_type,
                    "base_overall": base_overall,
                    "base_boundary": base_boundary,
                    "eu_overall": eu_overall,
                    "eu_boundary": eu_boundary,
                    "margin_overall": margin_overall,
                    "margin_boundary": margin_boundary,
                    "hard_precision_eu": float(np.mean(bmask_tr[idx_e])),
                    "hard_precision_margin": float(np.mean(bmask_tr[idx_m])),
                    "delta_boundary_eu_vs_base": eu_boundary - base_boundary,
                    "delta_boundary_margin_vs_base": margin_boundary - base_boundary,
                    "delta_overall_eu_vs_base": eu_overall - base_overall,
                    "delta_overall_margin_vs_base": margin_overall - base_overall,
                }
            )

    summary = {}
    for model_type in ["lr", "mlp"]:
        sub = [r for r in rows if r["model"] == model_type]
        summary[model_type] = {
            "base_overall": mean_std([r["base_overall"] for r in sub]),
            "base_boundary": mean_std([r["base_boundary"] for r in sub]),
            "hard_precision_eu": mean_std([r["hard_precision_eu"] for r in sub]),
            "hard_precision_margin": mean_std([r["hard_precision_margin"] for r in sub]),
            "delta_boundary_eu_vs_base": mean_std([r["delta_boundary_eu_vs_base"] for r in sub]),
            "delta_boundary_margin_vs_base": mean_std([r["delta_boundary_margin_vs_base"] for r in sub]),
            "delta_overall_eu_vs_base": mean_std([r["delta_overall_eu_vs_base"] for r in sub]),
            "delta_overall_margin_vs_base": mean_std([r["delta_overall_margin_vs_base"] for r in sub]),
            "paired_boundary_advantage_margin_minus_eu": paired_advantage(
                [r["delta_boundary_margin_vs_base"] for r in sub],
                [r["delta_boundary_eu_vs_base"] for r in sub],
            ),
            "paired_overall_advantage_margin_minus_eu": paired_advantage(
                [r["delta_overall_margin_vs_base"] for r in sub],
                [r["delta_overall_eu_vs_base"] for r in sub],
            ),
        }

    # 直观回答“是否被模型能力压制”：比较 LR 与 MLP 下 margin 相对 eu 的 paired 优势
    summary["capacity_gap"] = {
        "boundary_advantage_margin_minus_eu": {
            "lr": summary["lr"]["paired_boundary_advantage_margin_minus_eu"],
            "mlp": summary["mlp"]["paired_boundary_advantage_margin_minus_eu"],
        },
        "overall_advantage_margin_minus_eu": {
            "lr": summary["lr"]["paired_overall_advantage_margin_minus_eu"],
            "mlp": summary["mlp"]["paired_overall_advantage_margin_minus_eu"],
        },
    }

    with open(out / "trials.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    with open(out / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
