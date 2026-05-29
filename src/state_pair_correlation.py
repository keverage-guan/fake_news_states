"""
src/state_pair_correlation.py

Correlation analysis: does dissimilarity between HMM state pairs predict
how poorly a model trained in one state generalises to the other?

For each ordered state pair (i, j) where i ≠ j we compute:
  - Mean cross-window macro F1  (from the F1 matrix, averaging over all
    window pairs where train-window is in state i, test-window in state j)
  - Jensen-Shannon divergence between their 6-way class distributions
  - Euclidean distance between their PCA centroids (Z-scaled space)

We then correlate each dissimilarity measure with mean F1 using:
  - Spearman rank correlation  (non-parametric, robust to small N)
  - A permutation test on the Spearman r  (shuffle state labels, recompute)

Because F1 is a *generalisation* measure, lower F1 should correspond to
higher dissimilarity — we expect negative correlations.

Note: k=6 gives 6×5=30 ordered pairs, or 15 unique unordered pairs.
We use ordered pairs (i→j and j→i separately) since F1 is asymmetric
(train on i, test on j  ≠  train on j, test on i).

Outputs  (<output_dir>/)
-------
  statepair_f1.csv              mean F1 + both dissimilarity measures per pair
  correlation_results.csv       Spearman r, p (parametric + permutation) per measure
  plot_f1_vs_jsd.png            scatter: F1 vs JS divergence
  plot_f1_vs_pca_dist.png       scatter: F1 vs PCA centroid distance
  plot_correlation_null.png     permutation null distributions for both measures

Usage
-----
  python src/state_pair_correlation.py

  python src/state_pair_correlation.py \
      --decode_npz   data/hmm_hmm/6way/final_decode_k7.npz \
      --f1_npz       data/hmm_perf/6way/cross_window_f1.npz \
      --manifest     data/splits/hmm_windows/HMM_windows_manifest.csv \
      --output_dir   data/state_pair_correlation/k7 \
      --n_permutations 10000 \
      --seed 42
"""

import os
import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from tqdm import tqdm


# ── Label maps ────────────────────────────────────────────────────────────────

CLASS_NAMES = {
    0: "True",
    1: "Satire",
    2: "False Connection",
    3: "Imposter Content",
    4: "Manipulated Content",
    5: "Misleading Content",
}
CLS_COLS = [f"cls_{i}" for i in range(6)]

STATE_PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c",
    "#d62728", "#9467bd", "#8c564b",
    "#17becf", "#e377c2"
]


# ── Args ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--decode_npz",     default="data/hmm_hmm/6way/final_decode_k6.npz")
    p.add_argument("--f1_npz",         default="data/hmm_perf/6way/k6/cross_window_f1.npz")
    p.add_argument("--manifest",       default="data/splits/hmm_windows/HMM_windows_manifest.csv")
    p.add_argument("--output_dir",     default="data/state_pair_correlation")
    p.add_argument("--n_permutations", type=int, default=10_000)
    p.add_argument("--seed",           type=int, default=42)
    return p.parse_args()


# ── Dissimilarity measures ────────────────────────────────────────────────────

def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon divergence (base-2, range [0, 1])."""
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    def kl(a, b):
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / b[mask])))
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def pca_euclidean(means_i: np.ndarray, means_j: np.ndarray) -> float:
    return float(np.linalg.norm(means_i - means_j))


# ── Per-state class distribution from manifest ────────────────────────────────

def state_class_props(manifest: pd.DataFrame,
                      state_seq: np.ndarray,
                      window_ids: np.ndarray,
                      k: int) -> np.ndarray:
    """
    Returns array (k, 6) of class proportions per state.
    """
    props = np.zeros((k, 6), dtype=float)
    for s in range(k):
        mask     = state_seq == s
        sub      = manifest[mask]
        counts   = np.array([sub[f"cls_{c}"].sum() for c in range(6)], dtype=float)
        total    = counts.sum()
        props[s] = counts / total if total > 0 else counts
    return props


# ── Build state-pair table ────────────────────────────────────────────────────

def build_pair_table(state_seq, window_ids, f1_matrix, f1_cube,
                     class_props, hmm_means, k):
    """
    One row per ordered state pair (i, j), i ≠ j.
    """
    N = len(state_seq)
    rows = []

    for si in range(k):
        for sj in range(k):
            if si == sj:
                continue

            # Windows in each state
            idx_i = np.where(state_seq == si)[0]
            idx_j = np.where(state_seq == sj)[0]

            # Mean F1: train on windows in state si, test on windows in state sj
            vals = []
            for ti in idx_i:
                for tj in idx_j:
                    if f1_cube is not None:
                        seed_vals = f1_cube[ti, :, tj]
                        seed_vals = seed_vals[~np.isnan(seed_vals)]
                        vals.extend(seed_vals.tolist())
                    else:
                        v = f1_matrix[ti, tj]
                        if not np.isnan(v):
                            vals.append(v)

            mean_f1 = float(np.mean(vals)) if vals else np.nan
            n_pairs = len(vals)

            # Dissimilarities
            jsd  = js_divergence(class_props[si], class_props[sj])
            pdist = pca_euclidean(hmm_means[si], hmm_means[sj])

            rows.append({
                "state_i":    si,
                "state_j":    sj,
                "mean_f1":    round(mean_f1, 4),
                "n_pairs":    n_pairs,
                "jsd":        round(jsd,  6),
                "pca_dist":   round(pdist, 6),
            })

    return pd.DataFrame(rows)


# ── Spearman correlation + permutation test ───────────────────────────────────

def spearman_perm_test(x: np.ndarray, y: np.ndarray,
                       n_perm: int, rng: np.random.Generator,
                       desc: str = "") -> dict:
    """
    Spearman r between x and y, plus a permutation p-value (one-tailed,
    testing r ≤ observed, i.e. that the true correlation is more negative).
    """
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]

    r_obs, p_param = stats.spearmanr(x, y)

    null_r = np.empty(n_perm)
    for b in tqdm(range(n_perm), desc=desc, leave=False):
        null_r[b] = stats.spearmanr(rng.permutation(x), y)[0]

    # One-tailed: how often is null r ≤ observed r?
    # (we expect negative r, so significant = observed r is unusually negative)
    p_perm = float((null_r <= r_obs).mean())

    return {
        "r":          round(float(r_obs),  4),
        "p_param":    round(float(p_param), 4),
        "p_perm":     round(p_perm,         4),
        "n":          int(mask.sum()),
        "null_r":     null_r,
    }


# ── Plots ─────────────────────────────────────────────────────────────────────

def scatter_plot(x, y, labels_i, labels_j, k,
                 xlabel, ylabel, title, output_path,
                 r, p_perm):
    fig, ax = plt.subplots(figsize=(7, 6))

    # One point per ordered pair, coloured by train-state
    for idx, (xi, yi, si, sj) in enumerate(zip(x, y, labels_i, labels_j)):
        color = STATE_PALETTE[si % len(STATE_PALETTE)]
        ax.scatter(xi, yi, color=color, s=80, zorder=3,
                   edgecolors="white", linewidths=0.5)
        ax.annotate(f"{si}→{sj}", (xi, yi),
                    textcoords="offset points", xytext=(5, 3),
                    fontsize=7, color="dimgray")

    # Regression line
    valid = ~(np.isnan(x) | np.isnan(y))
    if valid.sum() >= 3:
        m, b = np.polyfit(x[valid], y[valid], 1)
        xr = np.linspace(x[valid].min(), x[valid].max(), 100)
        ax.plot(xr, m * xr + b, color="black", lw=1.5,
                linestyle="--", alpha=0.6, label="OLS trend")

    # Legend for train-states
    patches = [plt.Line2D([0], [0], marker="o", color="w",
                           markerfacecolor=STATE_PALETTE[s],
                           markersize=8, label=f"Train state {s}")
               for s in range(k)]
    ax.legend(handles=patches, fontsize=8, loc="upper right")

    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(f"{title}\nSpearman r = {r:.3f}  (perm p = {p_perm:.4f})",
                 fontsize=11)
    ax.grid(True, alpha=0.3, linestyle="--")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_null_distributions(null_jsd, obs_jsd, p_jsd,
                            null_pca, obs_pca, p_pca,
                            output_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for ax, null, obs, p, label in [
        (axes[0], null_jsd, obs_jsd, p_jsd, "JS divergence"),
        (axes[1], null_pca, obs_pca, p_pca, "PCA centroid distance"),
    ]:
        ax.hist(null, bins=60, color="steelblue", alpha=0.75,
                label="Null distribution")
        ax.axvline(obs, color="red", lw=2,
                   label=f"Observed r = {obs:.4f}\nperm p = {p:.4f}")
        ax.set_xlabel("Spearman r", fontsize=11)
        ax.set_ylabel("Count", fontsize=11)
        ax.set_title(f"Permutation null: F1 vs {label}", fontsize=11)
        ax.legend(fontsize=9)

    fig.suptitle("Permutation tests: state-pair dissimilarity predicts F1?",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    rng  = np.random.default_rng(args.seed)

    # ── Load ───────────────────────────────────────────────────────────────
    dec        = np.load(args.decode_npz, allow_pickle=True)
    state_seq  = dec["state_seq"].astype(int)
    window_ids = dec["window_ids"].astype(int)
    k          = int(dec.get("k", state_seq.max() + 1))
    hmm_means  = dec["means"].astype(float)          # (k, n_pca) — Z-scaled

    f1_data   = np.load(args.f1_npz, allow_pickle=True)
    f1_matrix = f1_data["f1_matrix"].astype(float)
    f1_cube   = f1_data["f1_per_seed_cube"].astype(float) \
                if "f1_per_seed_cube" in f1_data else None

    manifest  = pd.read_csv(args.manifest, parse_dates=["start_date", "end_date"])
    manifest  = manifest.set_index("window_local_idx").loc[window_ids].reset_index()
    for c in CLS_COLS:
        if c not in manifest.columns:
            manifest[c] = 0
        else:
            manifest[c] = manifest[c].fillna(0).astype(int)

    print(f"Loaded: k={k}, N={len(state_seq)} windows")

    # ── Per-state class proportions ────────────────────────────────────────
    class_props = state_class_props(manifest, state_seq, window_ids, k)

    print("\nPer-state class proportions:")
    header = f"  {'State':<8}" + "".join(f"{CLASS_NAMES[c]:>22}" for c in range(6))
    print(header)
    for s in range(k):
        row = f"  {s:<8}" + "".join(f"{class_props[s, c]:>21.1%}" for c in range(6))
        print(row)

    # ── Build pair table ───────────────────────────────────────────────────
    print("\nBuilding state-pair table …")
    df = build_pair_table(state_seq, window_ids, f1_matrix, f1_cube,
                          class_props, hmm_means, k)

    pair_path = os.path.join(args.output_dir, "statepair_f1.csv")
    df.to_csv(pair_path, index=False)
    print(f"  Saved: {pair_path}")
    print(df.to_string(index=False))

    # ── Correlations ───────────────────────────────────────────────────────
    f1_vals  = df["mean_f1"].values
    jsd_vals = df["jsd"].values
    pca_vals = df["pca_dist"].values

    print(f"\nRunning Spearman + permutation tests (n={args.n_permutations:,}) …")

    res_jsd = spearman_perm_test(jsd_vals, f1_vals, args.n_permutations, rng,
                                 desc="JSD permutation")
    res_pca = spearman_perm_test(pca_vals, f1_vals, args.n_permutations, rng,
                                 desc="PCA dist permutation")

    print(f"\n── Correlation results ──────────────────────────────────────────")
    print(f"  F1 vs JS divergence  :  r = {res_jsd['r']:+.4f}  "
          f"p_param = {res_jsd['p_param']:.4f}  "
          f"p_perm = {res_jsd['p_perm']:.4f}  (n={res_jsd['n']})")
    print(f"  F1 vs PCA distance   :  r = {res_pca['r']:+.4f}  "
          f"p_param = {res_pca['p_param']:.4f}  "
          f"p_perm = {res_pca['p_perm']:.4f}  (n={res_pca['n']})")

    corr_df = pd.DataFrame([
        {"measure": "JSD",      "spearman_r": res_jsd["r"],
         "p_param": res_jsd["p_param"], "p_perm": res_jsd["p_perm"],
         "n_pairs": res_jsd["n"]},
        {"measure": "PCA_dist", "spearman_r": res_pca["r"],
         "p_param": res_pca["p_param"], "p_perm": res_pca["p_perm"],
         "n_pairs": res_pca["n"]},
    ])
    corr_path = os.path.join(args.output_dir, "correlation_results.csv")
    corr_df.to_csv(corr_path, index=False)
    print(f"\n  Saved: {corr_path}")

    # ── Plots ──────────────────────────────────────────────────────────────
    scatter_plot(
        x=jsd_vals, y=f1_vals,
        labels_i=df["state_i"].values, labels_j=df["state_j"].values, k=k,
        xlabel="Jensen-Shannon divergence (class distributions)",
        ylabel="Mean cross-window macro F1",
        title="State-pair F1 vs. class distribution dissimilarity",
        output_path=os.path.join(args.output_dir, "plot_f1_vs_jsd.png"),
        r=res_jsd["r"], p_perm=res_jsd["p_perm"],
    )

    scatter_plot(
        x=pca_vals, y=f1_vals,
        labels_i=df["state_i"].values, labels_j=df["state_j"].values, k=k,
        xlabel="Euclidean distance between PCA centroids (Z-scaled)",
        ylabel="Mean cross-window macro F1",
        title="State-pair F1 vs. PCA centroid distance",
        output_path=os.path.join(args.output_dir, "plot_f1_vs_pca_dist.png"),
        r=res_pca["r"], p_perm=res_pca["p_perm"],
    )

    plot_null_distributions(
        null_jsd=res_jsd["null_r"], obs_jsd=res_jsd["r"], p_jsd=res_jsd["p_perm"],
        null_pca=res_pca["null_r"], obs_pca=res_pca["r"], p_pca=res_pca["p_perm"],
        output_path=os.path.join(args.output_dir, "plot_correlation_null.png"),
    )

    print("\nDone.")


if __name__ == "__main__":
    main()