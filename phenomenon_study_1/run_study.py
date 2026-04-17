import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from scipy.stats import t as student_t
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import StandardScaler


@dataclass
class TrialMetrics:
    seed: int
    method: str
    acc: float
    nmi: float
    ari: float
    boundary_acc: float
    boundary_nmi: float
    boundary_ari: float
    boundary_margin: float
    boundary_disagreement: float
    hard_hit_bc_ratio: float
    hard_hit_d_ratio: float


def clustering_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    labels_true = np.unique(y_true)
    labels_pred = np.unique(y_pred)
    cost = np.zeros((len(labels_true), len(labels_pred)), dtype=np.int64)
    for i, lt in enumerate(labels_true):
        for j, lp in enumerate(labels_pred):
            cost[i, j] = np.sum((y_true == lt) & (y_pred == lp))
    row_ind, col_ind = linear_sum_assignment(cost.max() - cost)
    matched = cost[row_ind, col_ind].sum()
    return matched / len(y_true)


def js_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    p = p / p.sum(axis=1, keepdims=True)
    q = q / q.sum(axis=1, keepdims=True)
    m = 0.5 * (p + q)
    kl_pm = np.sum(p * np.log(p / m), axis=1)
    kl_qm = np.sum(q * np.log(q / m), axis=1)
    return 0.5 * (kl_pm + kl_qm)


def make_linear_map(theta_deg: float, sx: float, sy: float) -> np.ndarray:
    theta = np.deg2rad(theta_deg)
    r = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    s = np.diag([sx, sy])
    return r @ s


def sample_student_t_clusters(rng: np.random.Generator, n_per_cluster: int, df: int):
    centers = np.array([[-2.0, 0.0], [2.0, 0.0], [0.0, 2.8]], dtype=np.float64)
    transforms = [
        np.array([[0.9, 0.2], [0.0, 0.65]]),
        np.array([[0.75, -0.2], [0.15, 0.7]]),
        np.array([[0.85, 0.15], [-0.1, 0.75]]),
    ]

    all_s, all_y = [], []
    for k, (mu, lk) in enumerate(zip(centers, transforms)):
        t_samples = student_t.rvs(df=df, size=(n_per_cluster, 2), random_state=rng)
        cluster = mu + t_samples @ lk.T
        all_s.append(cluster)
        all_y.append(np.full(n_per_cluster, k))
    s = np.concatenate(all_s, axis=0)
    y = np.concatenate(all_y, axis=0)
    return s, y, centers


def compute_oracle_types(
    s: np.ndarray,
    y: np.ndarray,
    centers: np.ndarray,
    boundary_tau: float,
    conflict_mask: np.ndarray,
    pseudo_mask: np.ndarray,
):
    dist = cdist(s, centers)
    d_sorted = np.sort(dist, axis=1)
    near_gap = d_sorted[:, 1] - d_sorted[:, 0]
    boundary_mask = near_gap <= boundary_tau

    # A easy-consistent, B boundary-consistent, C boundary-conflict, D pseudo-hard
    oracle_type = np.full(len(s), "A", dtype=object)
    oracle_type[boundary_mask] = "B"
    oracle_type[boundary_mask & conflict_mask] = "C"
    oracle_type[pseudo_mask] = "D"
    return oracle_type, boundary_mask


def generate_multiview_data(
    seed: int,
    n_per_cluster: int,
    df: int,
    sigma1: float,
    sigma2: float,
    p_conflict: float,
    boundary_quantile: float,
    pseudo_outlier_frac: float,
    pseudo_corrupt_frac: float,
):
    rng = np.random.default_rng(seed)
    s, y, centers = sample_student_t_clusters(rng, n_per_cluster=n_per_cluster, df=df)
    n = len(s)

    a1 = make_linear_map(theta_deg=15, sx=1.1, sy=0.95)
    a2 = make_linear_map(theta_deg=-28, sx=0.92, sy=1.15)

    x1 = s @ a1.T + rng.normal(0, sigma1, size=s.shape)
    x2 = s @ a2.T + rng.normal(0, sigma2, size=s.shape)

    dist = cdist(s, centers)
    nearest = np.argmin(dist, axis=1)
    d_sorted = np.sort(dist, axis=1)
    boundary_gap = d_sorted[:, 1] - d_sorted[:, 0]
    boundary_tau = np.quantile(boundary_gap, boundary_quantile)
    boundary_band = boundary_gap <= boundary_tau

    conflict_mask = np.zeros(n, dtype=bool)
    boundary_idx = np.where(boundary_band)[0]
    choose_conflict = rng.random(len(boundary_idx)) < p_conflict
    selected_conflict_idx = boundary_idx[choose_conflict]
    conflict_mask[selected_conflict_idx] = True

    for idx in selected_conflict_idx:
        row = dist[idx]
        order = np.argsort(row)
        a, b = order[0], order[1]
        direction_s = centers[b] - centers[a]
        direction_s = direction_s / (np.linalg.norm(direction_s) + 1e-9)
        mapped_direction = a2 @ direction_s
        step = rng.uniform(0.6, 1.0)
        x2[idx] = x2[idx] + step * mapped_direction

    pseudo_mask = np.zeros(n, dtype=bool)
    tail_score = np.abs(s - centers[y]).sum(axis=1)
    n_outlier = max(1, int(n * pseudo_outlier_frac))
    n_corrupt = max(1, int(n * pseudo_corrupt_frac))

    outlier_candidates = np.where(~boundary_band & ~conflict_mask)[0]
    if len(outlier_candidates) > 0:
        sorted_idx = outlier_candidates[np.argsort(tail_score[outlier_candidates])[::-1]]
        outlier_idx = sorted_idx[: min(n_outlier, len(sorted_idx))]
        pseudo_mask[outlier_idx] = True

    clean_candidates = np.where(~boundary_band & ~conflict_mask & ~pseudo_mask)[0]
    if len(clean_candidates) > 0:
        corrupt_idx = rng.choice(clean_candidates, size=min(n_corrupt, len(clean_candidates)), replace=False)
        pseudo_mask[corrupt_idx] = True
        x2[corrupt_idx] = x2[corrupt_idx] + rng.normal(0, 1.4, size=(len(corrupt_idx), 2))

    oracle_type, boundary_mask = compute_oracle_types(
        s=s,
        y=y,
        centers=centers,
        boundary_tau=boundary_tau,
        conflict_mask=conflict_mask,
        pseudo_mask=pseudo_mask,
    )

    return {
        "s": s,
        "x1": x1,
        "x2": x2,
        "y": y,
        "oracle_type": oracle_type,
        "boundary_mask": boundary_mask,
    }


def soft_assign_by_distance(x: np.ndarray, centers: np.ndarray, temperature: float = 1.0):
    dist2 = cdist(x, centers, metric="sqeuclidean")
    logits = -dist2 / max(temperature, 1e-6)
    logits = logits - logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits)
    return exp_logits / exp_logits.sum(axis=1, keepdims=True)


def run_baseline(x1: np.ndarray, x2: np.ndarray, n_clusters: int, seed: int):
    scaler1 = StandardScaler()
    scaler2 = StandardScaler()
    z1 = scaler1.fit_transform(x1)
    z2 = scaler2.fit_transform(x2)
    x = np.concatenate([z1, z2], axis=1)
    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=20)
    pred = km.fit_predict(x)
    q = soft_assign_by_distance(x, km.cluster_centers_)

    km_v1 = KMeans(n_clusters=n_clusters, random_state=seed, n_init=20).fit(z1)
    km_v2 = KMeans(n_clusters=n_clusters, random_state=seed, n_init=20).fit(z2)
    q1 = soft_assign_by_distance(z1, km_v1.cluster_centers_)
    q2 = soft_assign_by_distance(z2, km_v2.cluster_centers_)

    return {"pred": pred, "q": q, "q1": q1, "q2": q2, "selected_idx": np.array([], dtype=int), "prototypes": km.cluster_centers_, "x_joint": x}


def run_hard_variant(x1: np.ndarray, x2: np.ndarray, n_clusters: int, seed: int, hard_frac: float = 0.18):
    base = run_baseline(x1, x2, n_clusters=n_clusters, seed=seed)
    x = base["x_joint"]
    q = base["q"]
    top2 = np.partition(q, -2, axis=1)[:, -2:]
    margin = np.abs(top2[:, 1] - top2[:, 0])
    n_hard = max(5, int(len(x) * hard_frac))
    selected_idx = np.argsort(margin)[:n_hard]

    pred = base["pred"]
    centers = np.zeros((n_clusters, x.shape[1]), dtype=np.float64)
    for k in range(n_clusters):
        members = x[pred == k]
        if len(members) == 0:
            centers[k] = x[np.random.randint(0, len(x))]
        else:
            centers[k] = members.mean(axis=0)

    synth = []
    for idx in selected_idx:
        i = np.argsort(q[idx])[::-1]
        c1, c2 = i[0], i[1]
        center_mix = 0.5 * (centers[c1] + centers[c2])
        new_x = 0.6 * x[idx] + 0.4 * center_mix
        synth.append(new_x)
    synth = np.asarray(synth)

    x_aug = np.concatenate([x, synth], axis=0)
    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=30)
    km.fit(x_aug)
    pred_new = km.predict(x)
    q_new = soft_assign_by_distance(x, km.cluster_centers_)

    z1 = StandardScaler().fit_transform(x1)
    z2 = StandardScaler().fit_transform(x2)
    km_v1 = KMeans(n_clusters=n_clusters, random_state=seed, n_init=20).fit(z1)
    km_v2 = KMeans(n_clusters=n_clusters, random_state=seed, n_init=20).fit(z2)
    q1 = soft_assign_by_distance(z1, km_v1.cluster_centers_)
    q2 = soft_assign_by_distance(z2, km_v2.cluster_centers_)

    return {"pred": pred_new, "q": q_new, "q1": q1, "q2": q2, "selected_idx": selected_idx, "prototypes": km.cluster_centers_, "x_joint": x}


def evaluate(
    y: np.ndarray,
    boundary_mask: np.ndarray,
    oracle_type: np.ndarray,
    pred: np.ndarray,
    q: np.ndarray,
    q1: np.ndarray,
    q2: np.ndarray,
    selected_idx: np.ndarray,
):
    top2 = np.partition(q, -2, axis=1)[:, -2:]
    margin = np.abs(top2[:, 1] - top2[:, 0])
    disagreement = js_divergence(q1, q2)

    global_acc = clustering_accuracy(y, pred)
    global_nmi = normalized_mutual_info_score(y, pred)
    global_ari = adjusted_rand_score(y, pred)

    by = y[boundary_mask]
    bp = pred[boundary_mask]
    b_acc = clustering_accuracy(by, bp)
    b_nmi = normalized_mutual_info_score(by, bp)
    b_ari = adjusted_rand_score(by, bp)

    b_margin = float(margin[boundary_mask].mean())
    b_dis = float(disagreement[boundary_mask].mean())

    if len(selected_idx) > 0:
        selected_types = oracle_type[selected_idx]
        hit_bc = np.mean(np.isin(selected_types, ["B", "C"]))
        hit_d = np.mean(selected_types == "D")
    else:
        hit_bc, hit_d = 0.0, 0.0

    return {
        "acc": float(global_acc),
        "nmi": float(global_nmi),
        "ari": float(global_ari),
        "boundary_acc": float(b_acc),
        "boundary_nmi": float(b_nmi),
        "boundary_ari": float(b_ari),
        "boundary_margin": b_margin,
        "boundary_disagreement": b_dis,
        "hard_hit_bc_ratio": float(hit_bc),
        "hard_hit_d_ratio": float(hit_d),
    }


def plot_scatter(output_dir: Path, x_joint: np.ndarray, oracle_type: np.ndarray, selected_idx: np.ndarray, prototypes: np.ndarray):
    pca = PCA(n_components=2, random_state=0)
    x2d = pca.fit_transform(x_joint)
    p2d = pca.transform(prototypes)

    fig, ax = plt.subplots(1, 1, figsize=(8, 6), dpi=120)

    ax.scatter(x2d[:, 0], x2d[:, 1], s=9, c="#bdbdbd", alpha=0.45, label="all samples")

    bc_mask = np.isin(oracle_type, ["B", "C"])
    ax.scatter(x2d[bc_mask, 0], x2d[bc_mask, 1], s=13, c="#e63946", alpha=0.8, label="oracle boundary (B+C)")

    if len(selected_idx) > 0:
        ax.scatter(
            x2d[selected_idx, 0],
            x2d[selected_idx, 1],
            s=40,
            facecolors="none",
            edgecolors="#1d3557",
            linewidths=1.0,
            marker="o",
            label="selected hard samples",
        )

    ax.scatter(p2d[:, 0], p2d[:, 1], s=180, c="#2a9d8f", marker="*", edgecolors="black", linewidths=0.8, label="prototypes")

    ax.set_title("Latent Scatter Plot with Boundary and Selected Hard Samples")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / "figure_1_latent_scatter.png")
    plt.close(fig)


def plot_bar(output_dir: Path, summary: dict):
    metrics = ["boundary_ari", "boundary_margin", "boundary_disagreement"]
    names = ["Boundary ARI ↑", "Boundary Margin ↑", "Boundary Disagreement ↓"]

    bvals = [summary["baseline"][m] for m in metrics]
    hvals = [summary["hard"][m] for m in metrics]

    x = np.arange(len(metrics))
    width = 0.36

    fig, ax = plt.subplots(1, 1, figsize=(8, 5), dpi=120)
    ax.bar(x - width / 2, bvals, width, label="Baseline", color="#8ecae6")
    ax.bar(x + width / 2, hvals, width, label="Baseline+Hard", color="#ffb703")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_title("Boundary-focused Metrics")
    ax.legend()
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / "figure_2_3_4_boundary_metrics.png")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Phenomenon study 1 runner")
    parser.add_argument("--output-dir", type=str, default="phenomenon_study_1/outputs")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--n-per-cluster", type=int, default=400)
    parser.add_argument("--df", type=int, default=4)
    parser.add_argument("--sigma1", type=float, default=0.1)
    parser.add_argument("--sigma2", type=float, default=0.1)
    parser.add_argument("--p-conflict", type=float, default=0.4)
    parser.add_argument("--boundary-quantile", type=float, default=0.28)
    parser.add_argument("--pseudo-outlier-frac", type=float, default=0.02)
    parser.add_argument("--pseudo-corrupt-frac", type=float, default=0.015)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_trials = []
    last_hard_artifacts = None

    for seed in args.seeds:
        data = generate_multiview_data(
            seed=seed,
            n_per_cluster=args.n_per_cluster,
            df=args.df,
            sigma1=args.sigma1,
            sigma2=args.sigma2,
            p_conflict=args.p_conflict,
            boundary_quantile=args.boundary_quantile,
            pseudo_outlier_frac=args.pseudo_outlier_frac,
            pseudo_corrupt_frac=args.pseudo_corrupt_frac,
        )

        baseline = run_baseline(data["x1"], data["x2"], n_clusters=3, seed=seed)
        b_metrics = evaluate(
            y=data["y"],
            boundary_mask=data["boundary_mask"],
            oracle_type=data["oracle_type"],
            pred=baseline["pred"],
            q=baseline["q"],
            q1=baseline["q1"],
            q2=baseline["q2"],
            selected_idx=baseline["selected_idx"],
        )
        all_trials.append(TrialMetrics(seed=seed, method="baseline", **b_metrics))

        hard = run_hard_variant(data["x1"], data["x2"], n_clusters=3, seed=seed)
        h_metrics = evaluate(
            y=data["y"],
            boundary_mask=data["boundary_mask"],
            oracle_type=data["oracle_type"],
            pred=hard["pred"],
            q=hard["q"],
            q1=hard["q1"],
            q2=hard["q2"],
            selected_idx=hard["selected_idx"],
        )
        all_trials.append(TrialMetrics(seed=seed, method="hard", **h_metrics))

        last_hard_artifacts = {
            "x_joint": hard["x_joint"],
            "oracle_type": data["oracle_type"],
            "selected_idx": hard["selected_idx"],
            "prototypes": hard["prototypes"],
        }

    def avg(method: str, key: str):
        vals = [getattr(t, key) for t in all_trials if t.method == method]
        return float(np.mean(vals))

    summary = {
        "baseline": {k: avg("baseline", k) for k in TrialMetrics.__dataclass_fields__.keys() if k not in ["seed", "method"]},
        "hard": {k: avg("hard", k) for k in TrialMetrics.__dataclass_fields__.keys() if k not in ["seed", "method"]},
    }

    with open(out_dir / "trials.json", "w", encoding="utf-8") as f:
        json.dump([asdict(t) for t in all_trials], f, ensure_ascii=False, indent=2)
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    if last_hard_artifacts is not None:
        plot_scatter(
            output_dir=out_dir,
            x_joint=last_hard_artifacts["x_joint"],
            oracle_type=last_hard_artifacts["oracle_type"],
            selected_idx=last_hard_artifacts["selected_idx"],
            prototypes=last_hard_artifacts["prototypes"],
        )
        plot_bar(out_dir=out_dir, summary=summary)

    print("Saved outputs to", out_dir)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
