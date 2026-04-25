"""Stage 5: All-anchor line-search necessity study (based on 方案2.pdf)."""

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
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import StandardScaler


@dataclass
class PolicyResult:
    seed: int
    policy: str
    acc: float
    nmi: float
    ari: float
    boundary_acc: float
    boundary_nmi: float
    boundary_ari: float
    boundary_margin: float
    coverage_class: float
    coverage_pair: float
    coverage_radial: float


def clustering_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    labels_true = np.unique(y_true)
    labels_pred = np.unique(y_pred)
    cost = np.zeros((len(labels_true), len(labels_pred)), dtype=np.int64)
    for i, lt in enumerate(labels_true):
        for j, lp in enumerate(labels_pred):
            cost[i, j] = np.sum((y_true == lt) & (y_pred == lp))
    row_ind, col_ind = linear_sum_assignment(cost.max() - cost)
    return cost[row_ind, col_ind].sum() / len(y_true)


def student_q(x: np.ndarray, centers: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    d2 = cdist(x, centers, metric="sqeuclidean")
    num = (1.0 + d2 / alpha) ** (-(alpha + 1.0) / 2.0)
    return num / np.clip(num.sum(axis=1, keepdims=True), 1e-12, None)


def make_t_multiview(seed: int, n_per_cluster: int, df: int, p_conflict: float):
    rng = np.random.default_rng(seed)
    centers = np.array([[-2.0, 0.0], [2.0, 0.0], [0.0, 2.8]], dtype=np.float64)
    transforms = [
        np.array([[0.9, 0.15], [0.05, 0.7]]),
        np.array([[0.75, -0.2], [0.1, 0.75]]),
        np.array([[0.85, 0.1], [-0.1, 0.8]]),
    ]

    s_all, y_all = [], []
    for k in range(3):
        t_sample = student_t.rvs(df=df, size=(n_per_cluster, 2), random_state=rng)
        s = centers[k] + t_sample @ transforms[k].T
        s_all.append(s)
        y_all.append(np.full(n_per_cluster, k))
    s = np.vstack(s_all)
    y = np.hstack(y_all)

    a1 = np.array([[1.05, -0.2], [0.1, 0.95]])
    a2 = np.array([[0.9, 0.35], [-0.2, 1.0]])

    x1 = s @ a1.T + rng.normal(0, 0.1, size=s.shape)
    x2 = s @ a2.T + rng.normal(0, 0.1, size=s.shape)

    # oracle boundary band
    dist = cdist(s, centers)
    srt = np.sort(dist, axis=1)
    gap = srt[:, 1] - srt[:, 0]
    boundary_tau = np.quantile(gap, 0.28)
    boundary_mask = gap <= boundary_tau

    # boundary conflict in view2
    conflict_mask = np.zeros(len(s), dtype=bool)
    boundary_idx = np.where(boundary_mask)[0]
    chosen = boundary_idx[rng.random(len(boundary_idx)) < p_conflict]
    conflict_mask[chosen] = True
    for idx in chosen:
        row = dist[idx]
        order = np.argsort(row)
        a, b = order[0], order[1]
        d = centers[b] - centers[a]
        d = d / (np.linalg.norm(d) + 1e-12)
        x2[idx] = x2[idx] + rng.uniform(0.6, 1.0) * (a2 @ d)

    return x1, x2, y, boundary_mask, conflict_mask


def radial_bins(z: np.ndarray, labels: np.ndarray, centers: np.ndarray):
    """Per-class radial bin: 0 core / 1 middle / 2 tail by class-wise distance quantiles."""
    bins = np.zeros(len(z), dtype=int)
    for k in range(centers.shape[0]):
        idx = np.where(labels == k)[0]
        if len(idx) == 0:
            continue
        d = np.linalg.norm(z[idx] - centers[k], axis=1)
        q1, q2 = np.quantile(d, [0.33, 0.66])
        b = np.zeros(len(idx), dtype=int)
        b[d > q1] = 1
        b[d > q2] = 2
        bins[idx] = b
    return bins


def line_search_generate(anchor: np.ndarray, c_pos: np.ndarray, c_neg: np.ndarray, centers: np.ndarray):
    direction = c_neg - c_pos
    norm = np.linalg.norm(direction) + 1e-12
    direction = direction / norm

    best = anchor.copy()
    best_margin = 1e9
    # simple 1D line search grid
    for t in np.linspace(0.0, 1.2, 25):
        cand = anchor + t * direction
        q = student_q(cand[None, :], centers)[0]
        top = np.sort(q)[-2:]
        margin = top[1] - top[0]
        if margin < best_margin:
            best_margin = margin
            best = cand
    return best


def select_policy_indices(policy: str, buckets: dict, n_select: int, rng: np.random.Generator):
    E = buckets["E"]
    M = buckets["M"]
    H = buckets["H"]

    if policy == "Only-H":
        pool = H
    elif policy == "M+H":
        pool = np.concatenate([M, H])
    elif policy == "All":
        pool = np.concatenate([E, M, H])
    elif policy == "Random":
        pool = np.concatenate([E, M, H])
    else:
        raise ValueError(policy)

    if len(pool) == 0:
        return np.array([], dtype=int)

    n = min(n_select, len(pool))
    if policy == "All":
        # balanced draw from E/M/H to enforce wider anchor coverage
        each = max(1, n // 3)
        parts = []
        for g in [E, M, H]:
            if len(g) == 0:
                continue
            k = min(each, len(g))
            parts.append(rng.choice(g, size=k, replace=False))
        idx = np.concatenate(parts) if parts else np.array([], dtype=int)
        remain = n - len(idx)
        if remain > 0:
            left = np.setdiff1d(pool, idx, assume_unique=False)
            if len(left) > 0:
                idx = np.concatenate([idx, rng.choice(left, size=min(remain, len(left)), replace=False)])
        return idx

    return rng.choice(pool, size=n, replace=False)


def evaluate_policy(z: np.ndarray, y_true: np.ndarray, boundary_mask: np.ndarray, policy: str, idx_anchor: np.ndarray, labels: np.ndarray, comp: np.ndarray):
    km = KMeans(n_clusters=3, random_state=0, n_init=30)
    pred = km.fit_predict(z)
    q = student_q(z, km.cluster_centers_)
    top2 = np.sort(q, axis=1)[:, -2:]
    margin = top2[:, 1] - top2[:, 0]

    # coverage metrics from selected anchors
    if len(idx_anchor) == 0:
        cov_class = cov_pair = cov_radial = 0.0
    else:
        selected_labels = labels[idx_anchor]
        cov_class = len(np.unique(selected_labels)) / 3.0

        pairs = np.stack([selected_labels, comp[idx_anchor]], axis=1)
        norm_pairs = {tuple(sorted((int(a), int(b)))) for a, b in pairs if int(a) != int(b)}
        cov_pair = len(norm_pairs) / 3.0  # for K=3, pair count is 3

        bins = radial_bins(z, labels, km.cluster_centers_)
        by_class = []
        for k in range(3):
            ids = idx_anchor[selected_labels == k]
            if len(ids) == 0:
                by_class.append(0.0)
                continue
            by_class.append(len(np.unique(bins[ids])) / 3.0)
        cov_radial = float(np.mean(by_class))

    by = y_true[boundary_mask]
    bp = pred[boundary_mask]

    return {
        "acc": float(clustering_accuracy(y_true, pred)),
        "nmi": float(normalized_mutual_info_score(y_true, pred)),
        "ari": float(adjusted_rand_score(y_true, pred)),
        "boundary_acc": float(clustering_accuracy(by, bp)),
        "boundary_nmi": float(normalized_mutual_info_score(by, bp)),
        "boundary_ari": float(adjusted_rand_score(by, bp)),
        "boundary_margin": float(margin[boundary_mask].mean()),
        "coverage_class": float(cov_class),
        "coverage_pair": float(cov_pair),
        "coverage_radial": float(cov_radial),
    }


def scatter_three_panels(output_dir: Path, z: np.ndarray, boundary_mask: np.ndarray, generated_by_policy: dict, centers: np.ndarray):
    policies = ["Only-H", "M+H", "All"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), dpi=120)
    for ax, p in zip(axes, policies):
        ax.scatter(z[:, 0], z[:, 1], s=10, c="#c9c9c9", alpha=0.4, label="all")
        ax.scatter(z[boundary_mask, 0], z[boundary_mask, 1], s=14, c="#e63946", alpha=0.8, label="boundary (B+C)")
        gen = generated_by_policy[p]
        if len(gen) > 0:
            ax.scatter(gen[:, 0], gen[:, 1], s=28, facecolors="none", edgecolors="#1d3557", label="generated hard")
        ax.scatter(centers[:, 0], centers[:, 1], marker="*", s=180, c="#2a9d8f", edgecolors="black", label="prototype")
        ax.set_title(p)
        ax.grid(alpha=0.2)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4)
    fig.suptitle("Latent Scatter with Generated Hard Samples under Different Anchor Policies", y=1.02)
    fig.tight_layout()
    fig.savefig(output_dir / "stage5_scatter_three_policies.png", bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Stage5: all-anchor line-search necessity")
    parser.add_argument("--output-dir", type=str, default="phenomenon_study/stage5_outputs")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--n-per-cluster", type=int, default=400)
    parser.add_argument("--df", type=int, default=4)
    parser.add_argument("--p-conflict", type=float, default=0.4)
    parser.add_argument("--anchor-ratio", type=float, default=0.20)
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    all_rows = []
    scatter_cache = None

    for seed in args.seeds:
        rng = np.random.default_rng(seed)
        x1, x2, y, boundary_mask, _ = make_t_multiview(seed, args.n_per_cluster, args.df, args.p_conflict)

        z1 = StandardScaler().fit_transform(x1)
        z2 = StandardScaler().fit_transform(x2)
        z = np.hstack([z1, z2])

        base_km = KMeans(n_clusters=3, random_state=seed, n_init=30)
        base_labels = base_km.fit_predict(z)
        centers = base_km.cluster_centers_

        q = student_q(z, centers)
        order = np.argsort(q, axis=1)
        top1 = q[np.arange(len(q)), order[:, -1]]
        top2 = q[np.arange(len(q)), order[:, -2]]
        margin = top1 - top2
        comp = order[:, -2]

        # E/M/H buckets by margin quantiles
        q1, q2 = np.quantile(margin, [0.33, 0.66])
        H = np.where(margin <= q1)[0]
        M = np.where((margin > q1) & (margin <= q2))[0]
        E = np.where(margin > q2)[0]
        buckets = {"E": E, "M": M, "H": H}

        n_select = max(6, int(len(z) * args.anchor_ratio))
        generated_by_policy = {}

        for policy in ["Only-H", "M+H", "All", "Random"]:
            idx_anchor = select_policy_indices(policy, buckets, n_select, rng)

            gen = []
            for i in idx_anchor:
                yi = base_labels[i]
                cj = comp[i]
                g = line_search_generate(z[i], centers[yi], centers[cj], centers)
                gen.append(g)
            gen = np.asarray(gen) if len(gen) > 0 else np.zeros((0, z.shape[1]))
            generated_by_policy[policy] = gen

            z_aug = np.vstack([z, gen]) if len(gen) > 0 else z.copy()
            metrics = evaluate_policy(
                z=z_aug[: len(z)],
                y_true=y,
                boundary_mask=boundary_mask,
                policy=policy,
                idx_anchor=idx_anchor,
                labels=base_labels,
                comp=comp,
            )
            all_rows.append(PolicyResult(seed=seed, policy=policy, **metrics))

        scatter_cache = (z, boundary_mask, generated_by_policy, centers)

    # aggregate
    policies = sorted(set(r.policy for r in all_rows))
    keys = [k for k in PolicyResult.__dataclass_fields__.keys() if k not in ["seed", "policy"]]
    summary = {
        p: {k: float(np.mean([getattr(r, k) for r in all_rows if r.policy == p])) for k in keys}
        for p in policies
    }

    with open(out / "stage5_trials.json", "w", encoding="utf-8") as f:
        json.dump([asdict(r) for r in all_rows], f, ensure_ascii=False, indent=2)
    with open(out / "stage5_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    if scatter_cache is not None:
        scatter_three_panels(out, *scatter_cache)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
