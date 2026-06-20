"""
src/jsd_stability_analysis.py

For each state, compute JSD between every pair of windows within that state.
Do the same for every pair of windows from different states.
Report mean/variance of each distribution and plot them.

Usage
-----
  python src/jsd_stability_analysis.py
  python src/jsd_stability_analysis.py \
      --decode_npz data/hmm_hmm/6way/final_decode_k7.npz \
      --manifest   data/splits/hmm_windows/HMM_windows_manifest.csv \
      --output_dir data/jsd_stability/k7
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CLS_COLS = [f"cls_{i}" for i in range(6)]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--decode_npz", default="data/hmm_hmm/6way/final_decode_k7.npz")
    p.add_argument("--manifest",   default="data/splits/hmm_windows/HMM_windows_manifest.csv")
    p.add_argument("--output_dir", default="data/jsd_stability/k7")
    p.add_argument("--seed",       type=int, default=42)
    p.add_argument("--n_perm",     type=int, default=10_000)
    p.add_argument("--dpi",        type=int, default=150)
    return p.parse_args()


def js_divergence(p, q):
    p = np.asarray(p, float); p /= p.sum()
    q = np.asarray(q, float); q /= q.sum()
    m = 0.5 * (p + q)
    def kl(a, b):
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / b[mask])))
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def load_class_props(manifest_path, window_ids):
    df = pd.read_csv(manifest_path)
    id_col = next(c for c in df.columns if "window_local" in c.lower())
    df = df.set_index(id_col).loc[window_ids]
    for c in CLS_COLS:
        if c not in df.columns:
            df[c] = 0
    counts = df[CLS_COLS].fillna(0).values.astype(float)
    totals = counts.sum(axis=1, keepdims=True)
    totals[totals == 0] = 1.0
    return counts / totals


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    dec       = np.load(args.decode_npz, allow_pickle=True)
    state_seq = dec["state_seq"].astype(int)
    win_ids   = dec["window_ids"].astype(int)
    k         = int(dec.get("k", state_seq.max() + 1))
    N         = len(state_seq)

    props = load_class_props(args.manifest, win_ids)   # (N, 6)

    # ── Pairwise JSD matrix ───────────────────────────────────────────────────
    jsd_mat = np.zeros((N, N))
    for i in range(N):
        for j in range(i + 1, N):
            v = js_divergence(props[i], props[j])
            jsd_mat[i, j] = jsd_mat[j, i] = v

    # ── Split into within-state and across-state pairs ────────────────────────
    within, across = [], []
    for i in range(N):
        for j in range(i + 1, N):
            (within if state_seq[i] == state_seq[j] else across).append(jsd_mat[i, j])
    within = np.array(within)
    across = np.array(across)

    print(f"Within-state pairs : n={len(within):4d}  "
          f"mean={within.mean():.4f}  var={within.var():.5f}")
    print(f"Across-state pairs : n={len(across):4d}  "
          f"mean={across.mean():.4f}  var={across.var():.5f}")

    # ── Permutation test (across mean > within mean) ──────────────────────────
    obs_diff = across.mean() - within.mean()
    all_vals = np.concatenate([within, across])
    n_w      = len(within)
    null     = np.array([
        rng.permutation(all_vals)[n_w:].mean() - rng.permutation(all_vals)[:n_w].mean()
        for _ in range(args.n_perm)
    ])
    p_val = float((null >= obs_diff).mean())
    print(f"Δ mean (across−within) = {obs_diff:+.4f},  "
          f"permutation p = {p_val:.4f}  (N={args.n_perm})")

    null_path = os.path.join(args.output_dir, "jsd_stability_null.npy")
    np.save(null_path, null)
    print(f"Saved: {null_path}")

    # ── Per-state breakdown ───────────────────────────────────────────────────
    per_state = {}
    for s in range(k):
        idx = np.where(state_seq == s)[0]
        vals = [jsd_mat[idx[a], idx[b]]
                for a in range(len(idx)) for b in range(a + 1, len(idx))]
        per_state[s] = np.array(vals) if vals else np.array([np.nan])

    # ── CSV summary ───────────────────────────────────────────────────────────
    rows = [
        {"group": "within-state", "n": len(within),
         "mean_jsd": within.mean(), "var_jsd": within.var()},
        {"group": "across-state", "n": len(across),
         "mean_jsd": across.mean(), "var_jsd": across.var()},
    ]
    for s in range(k):
        v = per_state[s][~np.isnan(per_state[s])]
        rows.append({
            "group":    f"state_{s}_within",
            "n":        len(v),
            "mean_jsd": v.mean() if len(v) else np.nan,
            "var_jsd":  v.var()  if len(v) else np.nan,
        })
    csv_path = os.path.join(args.output_dir, "jsd_stability_summary.csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False, float_format="%.5f")
    print(f"Saved: {csv_path}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    # ── Save null array for generate_all_figures.py ───────────────────────────
    null_path = os.path.join(args.output_dir, "jsd_stability_null.npy")
    np.save(null_path, null)
    print(f"Saved: {null_path}")

    # ── Also save raw within/across arrays for violin reconstruction ──────────
    np.save(os.path.join(args.output_dir, "jsd_within.npy"), within)
    np.save(os.path.join(args.output_dir, "jsd_across.npy"), across)

    # ── Side-by-side: violin + per-state bars ─────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # Left: violin comparison
    ax = axes[0]
    parts = ax.violinplot([within, across], positions=[0, 1],
                          showmedians=True, showextrema=True)
    colors = ["#2ca02c", "#d62728"]
    for pc, col in zip(parts["bodies"], colors):
        pc.set_facecolor(col); pc.set_alpha(0.45)
    for key in ("cmedians", "cbars", "cmaxes", "cmins"):
        parts[key].set_color("black")
    jitter_rng = np.random.default_rng(1)
    for pos, grp, col in zip([0, 1], [within, across], colors):
        jit = jitter_rng.uniform(-0.07, 0.07, len(grp))
        ax.scatter(pos + jit, grp, color=col, alpha=0.4, s=10, zorder=3)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(
        [f"Within-State\n($n={len(within)}$, $\\mu={within.mean():.3f}$)",
         f"Across-State\n($n={len(across)}$, $\\mu={across.mean():.3f}$)"],
        fontsize=11,
    )
    ax.set_ylabel("Pairwise Jensen\u2013Shannon Divergence", fontsize=12)
    ax.set_title("Within- vs. Across-State Pairwise JSD", fontsize=12)
    ax.grid(True, axis="y", alpha=0.25, ls="--")
    ax.spines[["top", "right"]].set_visible(False)

    # Right: per-state mean JSD bars
    ax2 = axes[1]
    state_means = []
    state_labels = []
    for s in range(k):
        v = per_state[s][~np.isnan(per_state[s])]
        state_means.append(v.mean() if len(v) else np.nan)
        state_labels.append(f"$S_{s}$")
    bar_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
                  "#9467bd", "#8c564b", "#17becf", "#e377c2"]
    ax2.bar(range(k), state_means,
            color=[bar_colors[s % len(bar_colors)] for s in range(k)],
            alpha=0.75, edgecolor="white")
    ax2.axhline(within.mean(), color="#2ca02c", ls="--", lw=1.5,
                label=f"Pooled within mean ({within.mean():.3f})")
    ax2.axhline(across.mean(), color="#d62728", ls="--", lw=1.5,
                label=f"Pooled across mean ({across.mean():.3f})")
    ax2.set_xticks(range(k))
    ax2.set_xticklabels(state_labels, fontsize=10)
    ax2.set_ylabel("Mean Pairwise JSD (Within State)", fontsize=12)
    ax2.set_title("Per-State Mean Within-State JSD", fontsize=12)
    ax2.legend(fontsize=9, framealpha=0.85)
    ax2.grid(True, axis="y", alpha=0.25, ls="--")
    ax2.spines[["top", "right"]].set_visible(False)

    fig.suptitle(
        f"JSD Stability: Within- vs. Across-State Window Pairs ($K={k}$)",
        fontsize=14,
    )
    fig.tight_layout()
    plot_path = os.path.join(args.output_dir, "plot_jsd_stability.png")
    fig.savefig(plot_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {plot_path}")

    # ── Null distribution + observed Δμ ───────────────────────────────────────
    fig2, ax = plt.subplots(figsize=(7, 4.5))
    pct95 = np.percentile(null, 95)
    ax.hist(null, bins=60, color="#94A3B8", edgecolor="white",
            linewidth=0.3, alpha=0.85, label="Permutation Null")
    ax.axvline(obs_diff, color="#DC2626", lw=2.5,
               label=f"Observed mean difference = {obs_diff:+.4f}  (p = {p_val:.4f})")
    ax.axvline(pct95, color="#1f2937", lw=1.2, linestyle="--", alpha=0.7,
               label=f"Null 95th Pct = {pct95:.4f}")
    ax.set_xlabel("Mean JSD (Across-State) \u2212 Mean JSD (Within-State)", fontsize=13)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title(
        f"Permutation Test: Across- vs. Within-State JSD ($K={k}$)",
        fontsize=13,
    )
    ax.legend(fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.spines[["top", "right"]].set_visible(False)
    fig2.tight_layout()
    null_plot_path = os.path.join(args.output_dir, "plot_jsd_stability_null.png")
    fig2.savefig(null_plot_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig2)
    print(f"Saved: {null_plot_path}")


if __name__ == "__main__":
    main()