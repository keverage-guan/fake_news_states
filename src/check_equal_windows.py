"""
src/check_equal_windows.py

Check: does the HMM segmentation predict cross-window generalisation
better than a naive equal-duration (calendar-time) segmentation?

Both segmentations divide the N windows into exactly k groups.

  HMM segmentation   — the Viterbi-decoded state labels already computed.

  Equal-duration seg — the N windows split into k contiguous equal-size
                       groups:  windows 0…⌊N/k⌋-1 → group 0,
                                windows ⌊N/k⌋…2⌊N/k⌋-1 → group 1, … etc.
                       (last group absorbs any remainder, matching how the
                       original Exp-2 temporal splits were made)

For each segmentation we compute EXACTLY the same statistics as
within_across_states.py:

  1.  Pooled within-group mean F1  and  across-group mean F1.
  2.  Gap = within − across.
  3.  Distance-conditioned label-shuffle test  (n_permutations shuffles,
      stratified within each temporal-lag stratum, so the temporal-proximity
      confound is removed for both methods equally).

Then we compare:
  - Observed gaps (point estimate)
  - p-values from shuffle tests
  - Per-lag within/across curves side-by-side

The key question: is the HMM gap significantly larger than the equal-duration
gap?  We test this with a direct permutation test on the difference of gaps.

Outputs  (<output_dir>/)
-------
  comparison_summary.csv          one row per method + one diff row
  distance_stratified_both.csv    per-lag within/across means for both methods
  plot_gap_comparison.png         bar chart: HMM vs equal-duration gap
  plot_f1_vs_distance_both.png    within/across curves overlaid for both
  plot_direct_perm_test.png       null distribution for (HMM_gap − equal_gap)

Usage
-----
  python src/check_equal_windows.py

  python src/check_equal_windows.py \
      --decode_npz    data/hmm_hmm/6way/final_decode_k8.npz \
      --f1_npz        data/hmm_perf/6way/cross_window_f1.npz \
      --output_dir    data/k8/check_equal_windows \
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
import matplotlib.patches as mpatches
from tqdm import tqdm


# ── Argument parsing ──────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--decode_npz",      default="data/hmm_hmm/6way/final_decode_k7.npz")
    p.add_argument("--f1_npz",          default="data/hmm_perf/6way/cross_window_f1.npz")
    p.add_argument("--output_dir",      default="data/check_equal_windows/k7")
    p.add_argument("--n_permutations",  type=int, default=10_000)
    p.add_argument("--seed",            type=int, default=42)
    return p.parse_args()


# ── Segmentation helpers ──────────────────────────────────────────────────────

def equal_duration_labels(N: int, k: int) -> np.ndarray:
    """
    Assign each of N consecutive windows to one of k contiguous groups,
    as equal in size as possible.  The remainder r = N % k is distributed
    round-robin: the first r groups get one extra window each.

    Example: N=35, k=6  →  sizes [6, 6, 6, 6, 6, 5]
    """
    base      = N // k
    remainder = N % k
    # sizes[g] = base + 1 for g < remainder, else base
    sizes  = [base + (1 if g < remainder else 0) for g in range(k)]
    labels = np.empty(N, dtype=int)
    start  = 0
    for g, sz in enumerate(sizes):
        labels[start:start + sz] = g
        start += sz
    assert start == N
    return labels


# ── Pair-table builder ────────────────────────────────────────────────────────

def build_pair_df(label_seq: np.ndarray,
                  f1_matrix: np.ndarray,
                  f1_cube: np.ndarray | None) -> pd.DataFrame:
    """
    One row per off-diagonal (i, j [, seed]) cell.
    Columns: train_win, test_win, seed, distance, same_group, label_i, label_j, f1
    """
    N    = len(label_seq)
    rows = []

    if f1_cube is not None:
        n_seeds = f1_cube.shape[1]
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                base = dict(
                    train_win  = i, test_win = j,
                    distance   = abs(i - j),
                    same_group = int(label_seq[i] == label_seq[j]),
                    label_i    = label_seq[i],
                    label_j    = label_seq[j],
                )
                for s in range(n_seeds):
                    v = f1_cube[i, s, j]
                    if np.isnan(v):
                        continue
                    rows.append({**base, "seed": s, "f1": v})
    else:
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                rows.append(dict(
                    train_win  = i, test_win = j, seed = -1,
                    distance   = abs(i - j),
                    same_group = int(label_seq[i] == label_seq[j]),
                    label_i    = label_seq[i],
                    label_j    = label_seq[j],
                    f1         = f1_matrix[i, j],
                ))
    return pd.DataFrame(rows)


# ── Statistics ────────────────────────────────────────────────────────────────

def pooled_gap(df: pd.DataFrame) -> float:
    within = df.loc[df["same_group"] == 1, "f1"].mean()
    across = df.loc[df["same_group"] == 0, "f1"].mean()
    return float(within - across), float(within), float(across)


def distance_stratified(df: pd.DataFrame):
    """
    Returns DataFrame with columns: distance, within_mean, across_mean, gap, n_within, n_across
    """
    records = []
    for d, grp in df.groupby("distance"):
        w = grp.loc[grp["same_group"] == 1, "f1"]
        a = grp.loc[grp["same_group"] == 0, "f1"]
        records.append({
            "distance":    d,
            "within_mean": w.mean() if len(w) else np.nan,
            "across_mean": a.mean() if len(a) else np.nan,
            "gap":         w.mean() - a.mean() if (len(w) and len(a)) else np.nan,
            "n_within":    len(w),
            "n_across":    len(a),
        })
    return pd.DataFrame(records).sort_values("distance")


def distance_conditioned_shuffle(df: pd.DataFrame,
                                 n_perm: int,
                                 rng: np.random.Generator,
                                 desc: str = "") -> tuple:
    """
    Within each temporal-lag stratum, shuffle the same_group labels n_perm
    times.  Statistic = harmonic-mean-weighted average of per-lag gaps.
    Returns (null_gaps, observed_gap, p_value).
    """
    distances = sorted(df["distance"].unique())
    strata    = [df[df["distance"] == d].reset_index(drop=True) for d in distances]
    sizes     = np.array([len(s) for s in strata])
    weights   = sizes / sizes.sum()   # harmonic-mean-friendly; use simple proportional

    def weighted_gap(shuffled_labels_per_stratum):
        gaps = []
        for s_df, sl in zip(strata, shuffled_labels_per_stratum):
            w = s_df.loc[sl == 1, "f1"]
            a = s_df.loc[sl == 0, "f1"]
            if len(w) == 0 or len(a) == 0:
                gaps.append(np.nan)
            else:
                gaps.append(w.mean() - a.mean())
        gaps   = np.array(gaps, dtype=float)
        valid  = ~np.isnan(gaps)
        w_norm = weights[valid] / weights[valid].sum()
        return float(np.dot(w_norm, gaps[valid]))

    # Observed
    obs_labels = [s["same_group"].values for s in strata]
    obs_stat   = weighted_gap(obs_labels)

    # Null
    null_gaps = np.empty(n_perm)
    for b in tqdm(range(n_perm), desc=desc or "Shuffle test", leave=False):
        shuffled = [rng.permutation(s["same_group"].values) for s in strata]
        null_gaps[b] = weighted_gap(shuffled)

    p_val = float((null_gaps >= obs_stat).mean())
    return null_gaps, obs_stat, p_val


# ── Plots ─────────────────────────────────────────────────────────────────────

COLORS = {
    "hmm":   "#1f77b4",
    "equal": "#ff7f0e",
}


def plot_gap_comparison(hmm_gap, equal_gap, hmm_p, equal_p, output_path):
    fig, ax = plt.subplots(figsize=(6, 5))
    labels = ["HMM\nsegmentation", "Equal-duration\nsegmentation"]
    gaps   = [hmm_gap, equal_gap]
    colors = [COLORS["hmm"], COLORS["equal"]]
    bars   = ax.bar(labels, gaps, color=colors, width=0.4, edgecolor="black", linewidth=0.8)

    for bar, g, p in zip(bars, gaps, [hmm_p, equal_p]):
        ax.text(bar.get_x() + bar.get_width() / 2,
                g + 0.002,
                f"gap = {g:+.4f}\np = {p:.4f}",
                ha="center", va="bottom", fontsize=10)

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_ylabel("Within-group F1 − Across-group F1", fontsize=11)
    ax.set_title("Generalisation gap: HMM vs. equal-duration segmentation", fontsize=12)
    ax.set_ylim(min(0, min(gaps)) - 0.03, max(gaps) + 0.06)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_f1_vs_distance(strat_hmm, strat_eq, output_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    for ax, strat, label, color in [
        (axes[0], strat_hmm, "HMM segmentation",           COLORS["hmm"]),
        (axes[1], strat_eq,  "Equal-duration segmentation", COLORS["equal"]),
    ]:
        # Only keep distances where BOTH within and across have observations
        has_within = set(strat.loc[strat["n_within"] > 0, "distance"])
        has_across = set(strat.loc[strat["n_across"] > 0, "distance"])
        common_d   = sorted(has_within & has_across)

        s = strat[strat["distance"].isin(common_d)].sort_values("distance")
        d = s["distance"].values

        ax.plot(d, s["within_mean"].values, color=color, lw=2,
                marker="o", ms=4, label="Within-group")
        ax.plot(d, s["across_mean"].values, color=color, lw=2,
                marker="s", ms=4, linestyle="--", label="Across-group", alpha=0.7)
        ax.fill_between(d,
                        s["within_mean"].values,
                        s["across_mean"].values,
                        alpha=0.10, color=color)
        ax.set_title(label, fontsize=12)
        ax.set_xlabel("Temporal distance |i − j| (windows)", fontsize=11)
        ax.set_xlim(min(common_d) - 0.5, max(common_d) + 0.5)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, linestyle="--")

    axes[0].set_ylabel("Macro F1", fontsize=11)
    fig.suptitle("Within vs. across-group F1 by temporal lag\n"
                 "(x-axis limited to distances present in both within and across pairs)",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_direct_perm(null_diffs, obs_diff, p_val, output_path):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(null_diffs, bins=60, color="steelblue", alpha=0.75, label="Null distribution")
    ax.axvline(obs_diff, color="red", lw=2, label=f"Observed diff = {obs_diff:+.4f}\np = {p_val:.4f}")
    ax.set_xlabel("HMM gap − equal-duration gap", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("Direct permutation test: does HMM segment better than equal-duration?",
                 fontsize=12)
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_state_comparison(hmm_labels, equal_labels, N, output_path):
    """
    Two-row bar chart showing which group each window belongs to under each
    segmentation, so it's visually clear how they differ.
    """
    import matplotlib.cm as cm
    k      = max(hmm_labels.max(), equal_labels.max()) + 1
    cmap   = cm.tab10
    colors = [cmap(i / 10) for i in range(k)]

    fig, axes = plt.subplots(2, 1, figsize=(14, 3), gridspec_kw={"hspace": 0.6})
    for ax, labels, title in [
        (axes[0], hmm_labels,   "HMM segmentation"),
        (axes[1], equal_labels, "Equal-duration segmentation"),
    ]:
        for i, lbl in enumerate(labels):
            ax.bar(i, 1, color=colors[lbl], edgecolor="white", linewidth=0.3)
        ax.set_xlim(-0.5, N - 0.5)
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        ax.set_xlabel("Window index →", fontsize=9)
        ax.set_title(title, fontsize=10)
        patches = [mpatches.Patch(color=colors[g], label=f"G{g}") for g in range(k)]
        ax.legend(handles=patches, loc="upper right", fontsize=7,
                  ncol=k, framealpha=0.7)

    fig.suptitle("Segmentation comparison (each bar = one 60-day window)", fontsize=11)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    rng  = np.random.default_rng(args.seed)

    # ── Load data ──────────────────────────────────────────────────────────
    dec        = np.load(args.decode_npz, allow_pickle=True)
    hmm_labels = dec["state_seq"].astype(int)
    window_ids = dec["window_ids"].astype(int)
    N          = len(hmm_labels)
    k          = int(dec.get("k", hmm_labels.max() + 1))

    f1_data    = np.load(args.f1_npz, allow_pickle=True)
    f1_matrix  = f1_data["f1_matrix"].astype(float)
    f1_cube    = f1_data["f1_per_seed_cube"].astype(float) \
                 if "f1_per_seed_cube" in f1_data else None

    print(f"N={N} windows, k={k}")
    print(f"F1 matrix shape: {f1_matrix.shape}")
    if f1_cube is not None:
        print(f"Per-seed cube shape: {f1_cube.shape}")

    # ── Build equal-duration labels ────────────────────────────────────────
    equal_labels = equal_duration_labels(N, k)
    print(f"\nHMM    labels : {hmm_labels}")
    print(f"Equal  labels : {equal_labels}")

    # ── Build pair DataFrames ──────────────────────────────────────────────
    print("\nBuilding pair tables …")
    df_hmm   = build_pair_df(hmm_labels,   f1_matrix, f1_cube)
    df_equal = build_pair_df(equal_labels, f1_matrix, f1_cube)
    obs_label = "per-seed" if f1_cube is not None else "window-mean"
    print(f"  HMM   pairs : {len(df_hmm):,}  ({obs_label})")
    print(f"  Equal pairs : {len(df_equal):,}")

    # ── Pooled gaps ────────────────────────────────────────────────────────
    hmm_gap, hmm_w, hmm_a     = pooled_gap(df_hmm)
    equal_gap, equal_w, equal_a = pooled_gap(df_equal)

    print(f"\n── Pooled results ───────────────────────────────")
    print(f"  HMM   : within={hmm_w:.4f}  across={hmm_a:.4f}  gap={hmm_gap:+.4f}")
    print(f"  Equal : within={equal_w:.4f}  across={equal_a:.4f}  gap={equal_gap:+.4f}")
    print(f"  HMM gap − Equal gap = {hmm_gap - equal_gap:+.4f}")

    # ── Distance-stratified curves ─────────────────────────────────────────
    strat_hmm   = distance_stratified(df_hmm)
    strat_equal = distance_stratified(df_equal)

    # ── Distance-conditioned shuffle tests ─────────────────────────────────
    print("\nRunning distance-conditioned shuffle test for HMM segmentation …")
    null_hmm, obs_hmm_stat, p_hmm = distance_conditioned_shuffle(
        df_hmm, args.n_permutations, rng, desc="HMM shuffle")

    print("Running distance-conditioned shuffle test for equal-duration …")
    null_eq, obs_eq_stat, p_eq = distance_conditioned_shuffle(
        df_equal, args.n_permutations, rng, desc="Equal shuffle")

    print(f"\n── Shuffle test results ─────────────────────────────────────────")
    print(f"  HMM   : observed stat = {obs_hmm_stat:+.4f}  p = {p_hmm:.4f}"
          f"  {'✓ significant' if p_hmm < 0.05 else '✗ not significant'} at α=0.05")
    print(f"  Equal : observed stat = {obs_eq_stat:+.4f}  p = {p_eq:.4f}"
          f"  {'✓ significant' if p_eq < 0.05 else '✗ not significant'} at α=0.05")

    # ── Direct permutation test: is HMM gap > equal gap? ──────────────────
    # Under the null, the two segmentations are exchangeable (re-assigning
    # group labels randomly).  We permute the HMM labels N_perm times and
    # recompute both gaps, then measure P(HMM_gap − equal_gap ≥ observed).
    # Since the equal-duration labels are fixed (deterministic), we only need
    # to permute the *assignment* of windows to groups for the HMM condition.
    print(f"\nRunning direct permutation test (HMM gap > equal gap) "
          f"(n={args.n_permutations:,}) …")
    obs_diff   = hmm_gap - equal_gap
    null_diffs = np.empty(args.n_permutations)

    for b in tqdm(range(args.n_permutations), desc="Direct perm"):
        perm_labels = rng.permutation(hmm_labels)
        df_perm     = build_pair_df(perm_labels, f1_matrix, f1_cube)
        g_perm, _, _ = pooled_gap(df_perm)
        null_diffs[b] = g_perm - equal_gap

    p_direct = float((null_diffs >= obs_diff).mean())
    print(f"  Observed HMM − equal gap = {obs_diff:+.4f}")
    print(f"  p-value (one-tailed)     = {p_direct:.4f}"
          f"  {'✓ significant' if p_direct < 0.05 else '✗ not significant'} at α=0.05")

    # ── Save outputs ───────────────────────────────────────────────────────

    # Comparison summary CSV
    summary_rows = [
        {
            "method":          "HMM",
            "within_mean_f1":  round(hmm_w,         4),
            "across_mean_f1":  round(hmm_a,         4),
            "gap":             round(hmm_gap,        4),
            "dc_shuffle_stat": round(obs_hmm_stat,   4),
            "dc_shuffle_p":    round(p_hmm,          4),
        },
        {
            "method":          "equal_duration",
            "within_mean_f1":  round(equal_w,        4),
            "across_mean_f1":  round(equal_a,        4),
            "gap":             round(equal_gap,      4),
            "dc_shuffle_stat": round(obs_eq_stat,    4),
            "dc_shuffle_p":    round(p_eq,           4),
        },
        {
            "method":          "HMM_minus_equal (direct perm)",
            "within_mean_f1":  "",
            "across_mean_f1":  "",
            "gap":             round(obs_diff,       4),
            "dc_shuffle_stat": "",
            "dc_shuffle_p":    round(p_direct,       4),
        },
    ]
    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(args.output_dir, "comparison_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"\n── Saved summary → {summary_path}")

    # Distance-stratified CSV (merged)
    strat_hmm["method"]   = "HMM"
    strat_equal["method"] = "equal_duration"
    strat_both = pd.concat([strat_hmm, strat_equal], ignore_index=True)
    strat_path = os.path.join(args.output_dir, "distance_stratified_both.csv")
    strat_both.to_csv(strat_path, index=False)
    print(f"── Saved distance-stratified → {strat_path}")

    # Plots
    plot_gap_comparison(
        hmm_gap, equal_gap, p_hmm, p_eq,
        os.path.join(args.output_dir, "plot_gap_comparison.png"))

    plot_f1_vs_distance(
        strat_hmm, strat_equal,
        os.path.join(args.output_dir, "plot_f1_vs_distance_both.png"))

    plot_direct_perm(
        null_diffs, obs_diff, p_direct,
        os.path.join(args.output_dir, "plot_direct_perm_test.png"))

    plot_state_comparison(
        hmm_labels, equal_labels, N,
        os.path.join(args.output_dir, "plot_segmentation_comparison.png"))

    print("\nDone.")


if __name__ == "__main__":
    main()