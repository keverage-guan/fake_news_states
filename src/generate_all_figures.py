#!/usr/bin/env python3
"""
src/generate_all_figures.py

Single script that regenerates every figure whose underlying data is saved to
disk.  Outputs go to figures/ (k-independent) and figures/k{k}/ (per-k).

Covers:
  Global (once)
    - window_distributions.png          — sample counts at 5 temporal resolutions
    - window_stats.csv                  — matching summary table
    - cross_window_f1_heatmap_6way.png  — train-vs-test macro F1 heatmap
    - hmm_model_selection_6way.png      — BIC / AIC / LOO-CV vs k
    - hmm_ari_stability_6way.png        — Viterbi ARI across inits vs k
    - hmm_loo_per_fold_6way.png         — per-seed LOO curves
    - pca_sanity_before.png             — PC1/PC2 scatter of UNALIGNED weights
    - pca_sanity_after.png              — PC1/PC2 scatter after alignment
    - pca_component_selection.png       — scree + retained-component threshold

  Per-k  (k=7 and k=8 by default)
    - k{k}/transition_matrix.png        — HMM transition probability heatmap
    - k{k}/state_strip.png              — Viterbi state sequence (scatter+step)
    - k{k}/state_timeline_seeds.png     — Viterbi decode on centroid
    - k{k}/timeline_with_classes.png    — state bar + stacked class proportions
    - k{k}/f1_vs_distance.png           — within/across F1 vs temporal lag
    - k{k}/distance_conditioned_null.png — permutation-test null histogram
    - k{k}/statepair_heatmap.png        — mean F1 per (train-state, test-state)
    - k{k}/f1_heatmap_with_states.png   — full F1 matrix sorted by HMM state

Usage
-----
    # defaults: k=7,8  output=figures/
    python src/generate_all_figures.py

    python src/generate_all_figures.py \\
        --k 6 7 8 \\
        --output_dir figures \\
        --hmm_dir    data/hmm_hmm/6way \\
        --perf_dir   data/hmm_perf/6way \\
        --wa_dir     data/hmm_within_across/6way \\
        --pca_npz    data/hmm_weights/weights_pca.npz \\
        --manifest   data/splits/hmm_windows/HMM_windows_manifest.csv \\
        --dpi 300
"""

import os
import sys
import glob
import argparse
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib.dates as mdates
import matplotlib.ticker as mticker

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ── Shared constants ──────────────────────────────────────────────────────────

N_WAY = 6

CLASS_NAMES = [
    "True", "Satire", "False Connection",
    "Imposter Content", "Manipulated Content", "Misleading Content",
]
CLASS_COLORS = [
    "#009E73", "#56B4E9", "#E69F00",
    "#CC79A7", "#D55E00", "#0072B2",
]

TAB10         = plt.cm.tab10.colors
STATE_PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#17becf", "#e377c2",
]
COLORS_WA = {"within": "#2563EB", "across": "#DC2626"}


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Regenerate all saved-data figures for 6-way HMM analysis."
    )
    p.add_argument("--k", nargs="+", type=int, default=[7, 8],
                   help="HMM state counts to plot  (default: 7 8)")
    p.add_argument("--output_dir", default="figures")
    p.add_argument("--hmm_dir",  default="data/hmm_hmm/6way",
                   help="Contains hmm_scores.npz and final_decode_k*.npz")
    p.add_argument("--perf_dir", default="data/hmm_perf/6way",
                   help="Contains cross_window_f1.npz")
    p.add_argument("--wa_dir",   default="data/hmm_within_across/6way",
                   help="Parent of k{k}/ within-across output dirs")
    p.add_argument("--pca_npz",  default="data/hmm_weights/6way/weights_pca.npz")
    p.add_argument("--manifest",
                   default="data/splits/hmm_windows/HMM_windows_manifest.csv",
                   help="Window manifest CSV with start/end dates and cls_* columns")
    p.add_argument("--dpi", type=int, default=150)
    return p.parse_args()


# ── Utility ───────────────────────────────────────────────────────────────────

def savefig(fig, path, dpi=150):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {path}")


def load_npz(path, label=""):
    if not os.path.exists(path):
        print(f"  [skip] {label or path} — file not found")
        return None
    return np.load(path, allow_pickle=True)


def banner(title):
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")


def contiguous_runs(seq):
    """Return list of (state, start_idx, end_idx) for each run in seq."""
    runs, i = [], 0
    while i < len(seq):
        j = i
        while j < len(seq) and seq[j] == seq[i]:
            j += 1
        runs.append((seq[i], i, j - 1))
        i = j
    return runs


def window_calendar_dates(window_ids, manifest_path):
    """
    Map window indices → (start_dates, end_dates) as numpy datetime64[D].
    Tries the manifest first; falls back to 60-day windows from 2013-01-01.
    """
    origin = np.datetime64("2013-01-01", "D")

    candidates = [
        manifest_path,
        manifest_path.replace("HMM_windows_manifest.csv", "manifest.csv"),
        manifest_path.replace("manifest.csv",             "HMM_windows_manifest.csv"),
    ]
    for mpath in candidates:
        if not os.path.exists(mpath):
            continue
        mdf = pd.read_csv(mpath)

        id_col = next(
            (c for c in mdf.columns
             if "window" in c.lower() and ("id" in c.lower() or "idx" in c.lower())),
            mdf.columns[0],
        )
        start_col = next((c for c in mdf.columns if "start" in c.lower()), None)
        end_col   = next((c for c in mdf.columns if "end"   in c.lower()), None)

        if start_col and end_col:
            mdf = mdf.set_index(id_col)
            starts, ends = [], []
            for wid in window_ids:
                if wid in mdf.index:
                    starts.append(np.datetime64(str(mdf.loc[wid, start_col])[:10], "D"))
                    ends.append(  np.datetime64(str(mdf.loc[wid, end_col  ])[:10], "D"))
                else:
                    starts.append(origin + np.timedelta64(int(wid) * 60,      "D"))
                    ends.append(  origin + np.timedelta64(int(wid) * 60 + 60, "D"))
            return np.array(starts), np.array(ends)

    # Fallback: synthetic 60-day windows
    starts = np.array([origin + np.timedelta64(int(w) * 60,      "D") for w in window_ids])
    ends   = np.array([origin + np.timedelta64(int(w) * 60 + 60, "D") for w in window_ids])
    return starts, ends


def date_to_mpl(dt64):
    return mdates.date2num(pd.Timestamp(dt64).to_pydatetime())


# ═════════════════════════════════════════════════════════════════════════════
# GLOBAL FIGURES
# ═════════════════════════════════════════════════════════════════════════════

# ── 1. Window distributions ───────────────────────────────────────────────────

def figure_window_distributions(out_dir, dpi):
    banner("Window distributions")

    def _load():
        split_dir = "data/splits"
        cols = {"id", "created_utc", "hasImage", "2_way_label", "6_way_label"}
        if os.path.isdir(split_dir):
            tsv_files = glob.glob(os.path.join(split_dir, "*.tsv"))
            if tsv_files:
                dfs = []
                for f in tsv_files:
                    try:
                        dfs.append(pd.read_csv(f, sep="\t", low_memory=False,
                                               usecols=lambda c: c in cols))
                    except Exception:
                        pass
                if dfs:
                    data = pd.concat(dfs, ignore_index=True)
                    if "id" in data.columns:
                        data = data.drop_duplicates(subset="id")
                    data["created_utc"] = pd.to_numeric(data["created_utc"], errors="coerce")
                    data = data.dropna(subset=["created_utc"])
                    data["created_dt"] = pd.to_datetime(data["created_utc"], unit="s", utc=True)
                    return data.sort_values("created_dt").reset_index(drop=True)

        raw_dir   = "assets/raw/multimodal_only_samples"
        raw_paths = [os.path.join(raw_dir, f) for f in
                     ["multimodal_train.tsv", "multimodal_validate.tsv",
                      "multimodal_test_public.tsv"]]
        if all(os.path.exists(p) for p in raw_paths):
            dfs = [pd.read_csv(p, sep="\t", low_memory=False,
                               usecols=lambda c: c in cols) for p in raw_paths]
            data = pd.concat(dfs, ignore_index=True)
            if "hasImage" in data.columns:
                data = data[data["hasImage"] == True]
            data["created_utc"] = pd.to_numeric(data["created_utc"], errors="coerce")
            data = data.dropna(subset=["created_utc"])
            data["created_dt"] = pd.to_datetime(data["created_utc"], unit="s", utc=True)
            return data.sort_values("created_dt").reset_index(drop=True)
        return None

    data = _load()
    if data is None:
        print("  [skip] no TSV data found")
        return

    days  = 60
    color = "#4C72B0"

    ts   = data["created_dt"].dt.tz_localize(None)
    t0dt = ts.min().normalize()
    t1dt = ts.max().normalize() + pd.Timedelta(days=1)

    total_days = (t1dt - t0dt).days
    n_wins     = total_days // days + (1 if total_days % days else 0)
    rows = []
    for i in range(n_wins):
        w_start = t0dt + pd.Timedelta(days=i * days)
        w_end   = min(t0dt + pd.Timedelta(days=(i + 1) * days), t1dt)
        count   = int(((ts >= w_start) & (ts < w_end)).sum())
        rows.append({"window_start": w_start, "count": count,
                     "actual_days": (w_end - w_start).days})
    wdf = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.bar(wdf["window_start"], wdf["count"],
           width=pd.Timedelta(days=days * 0.9), align="edge",
           color=color, alpha=0.85, linewidth=0.3, edgecolor="white")
    mean_val = wdf["count"].mean()
    ax.axhline(mean_val, color="black", lw=1.0, ls="--", alpha=0.6, zorder=5)
    ax.text(wdf["window_start"].iloc[-1], mean_val * 1.05,
            f"mean={mean_val:,.0f}", fontsize=8,
            va="bottom", ha="right", alpha=0.75)
    last_days = int(wdf["actual_days"].iloc[-1])
    note = f"  [last window = {last_days} days]" if last_days != days else ""
    ax.set_title(
        f"r/Fakeddit — Sample counts per 60-day window  "
        f"(n_windows={len(wdf)}, total={wdf['count'].sum():,}){note}",
        fontsize=12, fontweight="bold")
    ax.set_ylabel("Posts per window", fontsize=10)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.set_xlim(ts.min(), ts.max())
    ax.tick_params(axis="x", labelsize=8, rotation=30)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25, lw=0.5)
    fig.tight_layout()
    savefig(fig, os.path.join(out_dir, "window_distributions.png"), dpi=dpi)

    # CSV
    c = wdf["count"]
    pd.DataFrame([{"Resolution": "60 days", "N windows": len(wdf),
                   "Total posts": int(c.sum()), "Mean/window": f"{c.mean():.0f}",
                   "Std": f"{c.std():.0f}", "Min": int(c.min()),
                   "Max": int(c.max())}]).to_csv(
        os.path.join(out_dir, "window_stats.csv"), index=False)
    print(f"    Saved: {os.path.join(out_dir, 'window_stats.csv')}")


# ── 2. Cross-window F1 heatmap ────────────────────────────────────────────────

def figure_cross_window_heatmap(perf_dir, out_dir, dpi):
    banner("Cross-window F1 heatmap")
    d = load_npz(os.path.join(perf_dir, "cross_window_f1.npz"), "cross_window_f1.npz")
    if d is None:
        return

    f1_matrix = d["f1_matrix"]
    valid_ids = d["valid_ids"].tolist()
    n = len(valid_ids)

    plot_mat = f1_matrix.copy().astype(float)
    np.fill_diagonal(plot_mat, np.nan)

    fig, ax = plt.subplots(figsize=(max(6, n * 0.4), max(5, n * 0.4)))
    im = ax.imshow(plot_mat, aspect="auto", cmap="RdYlGn",
                   vmin=np.nanmin(plot_mat), vmax=np.nanmax(plot_mat))
    plt.colorbar(im, ax=ax, label="Macro F1")
    tick_labels = [f"W{i:03d}" for i in valid_ids]
    ax.set_xticks(range(n)); ax.set_xticklabels(tick_labels, rotation=90, fontsize=7)
    ax.set_yticks(range(n)); ax.set_yticklabels(tick_labels, fontsize=7)
    ax.set_xlabel("Test window", fontsize=11)
    ax.set_ylabel("Train window", fontsize=11)
    ax.set_title("Cross-window macro F1 (6-way)", fontsize=13)
    fig.tight_layout()
    savefig(fig, os.path.join(out_dir, "cross_window_f1_heatmap_6way.png"), dpi=dpi)


# ── 3. HMM model selection ────────────────────────────────────────────────────

def figure_model_selection(hmm_dir, out_dir, dpi):
    banner("HMM model selection")

    scores_path = None
    for candidate in [os.path.join(hmm_dir, "hmm_scores.npz"),
                      os.path.join(os.path.dirname(hmm_dir), "hmm_scores.npz")]:
        if os.path.exists(candidate):
            scores_path = candidate
            break
    if scores_path is None:
        print("  [skip] hmm_scores.npz not found")
        return

    d = np.load(scores_path)
    k_arr    = d["k_range"]
    loo_mean = d["loo_mean"]
    loo_std  = d["loo_std"]
    ari_mean = d["ari_mean"]
    ari_all  = d["ari_all"]
    loo_all  = d["loo_all"]

    valid = np.isfinite(loo_mean)

    # ── LOO-CV panel ─────────────────────────────────────────────────────────
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))

    if valid.any():
        ax.plot(k_arr[valid], loo_mean[valid], "o-", color="#16A34A",
                lw=2, ms=6, label="Mean LOO-CV")
        ax.fill_between(k_arr[valid],
                        loo_mean[valid] - loo_std[valid],
                        loo_mean[valid] + loo_std[valid],
                        alpha=0.15, color="#16A34A")
    ax.axvline(7, color="#16A34A", ls="--", alpha=0.6,
               label=r"$k = 7$ (selected)")
    ax.set_xlabel("Number of States ($k$)", fontsize=12)
    ax.set_ylabel("Mean Held-Out Log-Likelihood per Observation", fontsize=12)
    ax.set_title("Leave-One-Out Cross-Validation", fontsize=12)
    ax.set_xticks(k_arr)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    savefig(fig, os.path.join(out_dir, "hmm_model_selection_6way.png"), dpi=dpi)

    # ── ARI stability ────────────────────────────────────────────────────────
    ari_mean_r = np.nanmean(ari_all, axis=1)
    ari_std_r  = np.nanstd(ari_all,  axis=1)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(k_arr, ari_mean_r, "o-", color="#EA580C", lw=2, ms=6)
    ax.fill_between(k_arr, ari_mean_r - ari_std_r, ari_mean_r + ari_std_r,
                    alpha=0.15, color="#EA580C")
    ax.set_xlabel("Number of States ($k$)", fontsize=12)
    ax.set_ylabel("Mean Pairwise ARI", fontsize=12)
    ax.set_title("Viterbi Decode Stability Across Random Initialisations", fontsize=12)
    ax.set_xticks(k_arr)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    savefig(fig, os.path.join(out_dir, "hmm_ari_stability_6way.png"), dpi=dpi)

    # ── LOO per-fold ─────────────────────────────────────────────────────────
    n_folds = loo_all.shape[1] if loo_all.ndim > 1 else 0
    if n_folds > 1:
        fig, ax = plt.subplots(figsize=(8, 6))
        for fold in range(n_folds):
            ax.plot(k_arr, loo_all[:, fold], "o--", lw=1, ms=4, alpha=0.6,
                    label=f"Fold {fold}")
        ax.plot(k_arr[valid], loo_mean[valid], "o-", color="black",
                lw=2.5, ms=6, label="Mean", zorder=5)
        ax.set_xlabel("Number of States ($k$)", fontsize=12)
        ax.set_ylabel("Held-Out Log-Likelihood per Observation", fontsize=12)
        ax.set_title("LOO-CV Log-Likelihood per Fold", fontsize=12)
        ax.set_xticks(k_arr)
        ax.legend(fontsize=8, ncol=min(n_folds, 5))
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        savefig(fig, os.path.join(out_dir, "hmm_loo_per_fold_6way.png"), dpi=dpi)


# ── 4. PCA sanity check ───────────────────────────────────────────────────────

def figure_pca_sanity(pca_npz, out_dir, dpi):
    banner("PCA sanity check")
    d = load_npz(pca_npz, "weights_pca.npz")
    if d is None:
        return

    Z_scaled   = d["Z_scaled"].astype(np.float32)
    window_ids = d["window_ids"].astype(int)
    seed_ids   = d["seed_ids"].astype(int)
    evr        = d["explained_variance_ratio"]
    evr_full   = d["evr_full"]
    threshold  = float(np.asarray(d["component_threshold"]).ravel()[0])

    unique_wins  = np.unique(window_ids)
    unique_seeds = np.unique(seed_ids)
    n_wins  = len(unique_wins)
    MARKERS = ["o", "s", "^", "D", "v", "P", "*", "X", "h", "+"]
    win_to_idx  = {w: i for i, w in enumerate(unique_wins)}
    seed_to_idx = {s: i for i, s in enumerate(unique_seeds)}
    cmap_win = plt.cm.viridis

    def _scatter(Z, title, out_path, xlabel="PC1", ylabel="PC2"):
        fig, ax = plt.subplots(figsize=(10, 7))
        for i in range(len(window_ids)):
            c = cmap_win(win_to_idx[window_ids[i]] / max(n_wins - 1, 1))
            m = MARKERS[seed_to_idx[seed_ids[i]] % len(MARKERS)]
            ax.scatter(Z[i, 0], Z[i, 1], color=c, marker=m, s=25, alpha=0.6)
        sm = plt.cm.ScalarMappable(cmap=cmap_win,
                                   norm=plt.Normalize(0, max(n_wins - 1, 1)))
        sm.set_array([])
        plt.colorbar(sm, ax=ax, label="Window index")
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=12)
        legend_els = [
            mlines.Line2D([], [], color="grey",
                          marker=MARKERS[seed_to_idx[s] % len(MARKERS)],
                          linestyle="None", markersize=7, alpha=0.8, label=f"seed {s}")
            for s in unique_seeds[:len(MARKERS)]
        ]
        ax.legend(handles=legend_els, title="seed", fontsize=8, loc="upper right")
        fig.tight_layout()
        savefig(fig, out_path, dpi=dpi)

    # ── 4a: PC1 vs PC2 scatter BEFORE alignment ───────────────────────────────
    if "Z_before_scaled" in d:
        Z_before = d["Z_before_scaled"].astype(np.float32)
        _scatter(
            Z_before,
            title="PCA of UNALIGNED MLP weights\n"
                  "(coloured by window, marker by seed — seed clustering expected)",
            out_path=os.path.join(out_dir, "pca_sanity_before.png"),
            xlabel="PC1 (unaligned)",
            ylabel="PC2 (unaligned)",
        )
    else:
        print("    [skip] pca_sanity_before — Z_before_scaled not in weights_pca.npz "
              "(re-run extract_weights_pca.py to populate it)")

    # ── 4b: PC1 vs PC2 scatter AFTER alignment ────────────────────────────────
    _scatter(
        Z_scaled,
        title="PCA of aligned MLP weights\n(coloured by window, marker by seed)",
        out_path=os.path.join(out_dir, "pca_sanity_after.png"),
        xlabel=f"PC1 ({evr[0]*100:.1f}% var)",
        ylabel=f"PC2 ({evr[1]*100:.1f}% var)",
    )

    # ── 4c: Component selection scree ────────────────────────────────────────
    n_full = len(evr_full)
    n_show = min(100, n_full)
    xs     = np.arange(1, n_show + 1)

    broken_stick = np.array(
        [(1 / n_full) * sum(1 / j for j in range(i, n_full + 1))
         for i in range(1, n_show + 1)]
    )

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(xs, evr_full[:n_show], "o-", color="#DC2626",
            lw=1.5, ms=3, label="Per-component EVR")
    ax.plot(xs, broken_stick, "--", color="#2563EB",
            lw=1.5, label="Broken-stick expectation")
    ax.axvline(len(evr), color="black", ls="--", lw=1.5,
               label=f"Retained: {len(evr)} components (threshold={threshold:.3f})")
    ax.set_xlabel("Principal component", fontsize=12)
    ax.set_ylabel("Explained variance ratio", fontsize=12)
    ax.set_title(f"PCA component selection (first {n_show} of {n_full} PCs shown)",
                 fontsize=12)
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    savefig(fig, os.path.join(out_dir, "pca_component_selection.png"), dpi=dpi)


# ═════════════════════════════════════════════════════════════════════════════
# PER-K FIGURES
# ═════════════════════════════════════════════════════════════════════════════

def _transition_matrix(trans_mat, k, out_path, dpi):
    fig, ax = plt.subplots(figsize=(max(5, k * 0.9), max(4, k * 0.8)))
    im = ax.imshow(trans_mat, cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, label="Transition probability")
    labels = [f"State {s}" for s in range(k)]
    ax.set_xticks(range(k)); ax.set_xticklabels(labels, fontsize=10)
    ax.set_yticks(range(k)); ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("To state", fontsize=11)
    ax.set_ylabel("From state", fontsize=11)
    ax.set_title(f"HMM Transition Matrix (k={k}, 6-way)", fontsize=12)
    for i in range(k):
        for j in range(k):
            ax.text(j, i, f"{trans_mat[i,j]:.3f}", ha="center", va="center",
                    fontsize=8, color="white" if trans_mat[i,j] > 0.6 else "black")
    fig.tight_layout()
    savefig(fig, out_path, dpi=dpi)


def _state_strip(state_seq, start_dates, end_dates, k, out_path, dpi):
    # ── Relabel states in order of first chronological appearance ────────────
    relabel = {}
    next_id = 0
    for s in state_seq:
        if s not in relabel:
            relabel[s] = next_id
            next_id += 1
    state_seq = np.array([relabel[s] for s in state_seq])
    k_plot    = next_id   # may be < k if some states never appear in the decode

    n_win     = len(state_seq)
    x_vals    = np.arange(n_win)
    colors    = [TAB10[s % 10] for s in state_seq]
    mid_dates = [s + (e - s) / 2 for s, e in zip(start_dates, end_dates)]

    fig, ax = plt.subplots(figsize=(8, 3))

    ax.scatter(x_vals, state_seq, c=colors, s=80, zorder=3,
               edgecolors="white", linewidths=0.5)
    ax.step(x_vals, state_seq, where="mid",
            color="gray", linewidth=0.8, alpha=0.6, zorder=2)

    for idx in np.where(np.diff(state_seq) != 0)[0]:
        ax.axvline(idx + 0.5, color="black", linewidth=1.2,
                   linestyle="--", alpha=0.6)

    tick_step = max(1, n_win // 8)
    tick_idxs = list(range(0, n_win, tick_step))
    ax.set_xticks(tick_idxs)
    ax.set_xticklabels(
        [pd.Timestamp(mid_dates[i]).strftime("%b %Y") for i in tick_idxs],
        rotation=30, ha="right", fontsize=8,
    )
    ax.set_yticks(range(k_plot))
    ax.set_yticklabels([f"State {i}" for i in range(k_plot)], fontsize=9)
    ax.set_ylabel("HMM State", fontsize=11)
    ax.set_xlabel("Window (chronological)", fontsize=11)
    ax.set_title(f"Viterbi State Sequence on Seed Centroid  (k={k}, 6-way)", fontsize=12)
    ax.grid(axis="x", alpha=0.3, linestyle=":")
    ax.set_xlim(-0.5, n_win - 0.5)
    ax.set_ylim(-0.5, k_plot - 0.5)

    legend_patches = [mpatches.Patch(color=TAB10[s % 10], label=f"State {s}")
                      for s in range(k_plot)]
    ax.legend(handles=legend_patches, loc="upper left", fontsize=8, ncol=k_plot,
              framealpha=0.85)
    fig.tight_layout()
    savefig(fig, out_path, dpi=dpi)


def _state_timeline_seeds(dec, start_dates, end_dates, k, out_path, dpi):
    state_seq    = dec["state_seq"].astype(int)
    state_matrix = dec["state_matrix"] if "state_matrix" in dec else None
    seeds_dec    = dec["seed_ids_decoded"].tolist() if "seed_ids_decoded" in dec else []
    n_win        = len(state_seq)

    per_seed = {}
    if state_matrix is not None:
        for idx, s in enumerate(seeds_dec):
            per_seed[s] = state_matrix[idx]

    if not per_seed:
        print(f"    [info] state_timeline_seeds: no per-seed decode data in npz "
              f"(state_matrix/seed_ids_decoded absent) — showing centroid decode only")

    valid_seeds = list(per_seed.keys())
    n_rows      = len(valid_seeds) + 1

    fig, axes = plt.subplots(n_rows, 1,
                              figsize=(14, max(4, n_rows * 1.1)), sharex=True)
    if n_rows == 1:
        axes = [axes]

    boundary_nums = []
    for idx in np.where(np.diff(state_seq) != 0)[0]:
        boundary_nums.append(date_to_mpl(end_dates[idx]))

    def draw_row(ax, seq, title):
        for i in range(n_win):
            left  = date_to_mpl(start_dates[i])
            right = date_to_mpl(end_dates[i])
            ax.barh(0, right - left, left=left, height=0.8,
                    color=TAB10[seq[i] % 10], alpha=0.85, linewidth=0)
        for bn in boundary_nums:
            ax.axvline(bn, color="black", lw=1.2, ls="--", alpha=0.7, zorder=5)
        ax.set_yticks([0]); ax.set_yticklabels([title], fontsize=9)
        ax.set_ylim(-0.6, 0.6); ax.xaxis_date()
        ax.grid(axis="x", alpha=0.25, ls=":")

    for ax, s in zip(axes[:-1], valid_seeds):
        draw_row(ax, per_seed[s], f"seed {s}")
    draw_row(axes[-1], state_seq, "Viterbi (centroid)")
    axes[-1].set_facecolor("#F0F4FF")
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    axes[-1].xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    fig.autofmt_xdate(rotation=30, ha="right")

    legend_patches = [mpatches.Patch(color=TAB10[s % 10], label=f"State {s}")
                      for s in range(k)]
    axes[0].legend(handles=legend_patches, loc="upper right", fontsize=8, ncol=k)
    fig.suptitle(f"HMM State Timeline — Viterbi on per-window centroids (k={k}, 6-way)",
                 fontsize=12)
    fig.tight_layout()
    savefig(fig, out_path, dpi=dpi)


def _timeline_with_classes(dec, manifest_path, k, out_path, dpi):
    """State colour bar (top) + stacked 6-way class proportions (bottom)."""
    state_seq  = dec["state_seq"].astype(int)
    window_ids = dec["window_ids"].astype(int)
    CLS_COLS   = [f"cls_{i}" for i in range(6)]

    # Load manifest; need cls_* columns
    mdf = None
    for mpath in [manifest_path,
                  manifest_path.replace("HMM_windows_manifest.csv", "manifest.csv"),
                  manifest_path.replace("manifest.csv", "HMM_windows_manifest.csv")]:
        if not os.path.exists(mpath):
            continue
        _m = pd.read_csv(mpath)
        if any(c in _m.columns for c in CLS_COLS):
            mdf = _m
            break

    if mdf is None:
        print("    [skip] timeline_with_classes — manifest without cls_* columns not found")
        return

    id_col    = next((c for c in mdf.columns if "window" in c.lower()), mdf.columns[0])
    start_col = next((c for c in mdf.columns if "start" in c.lower()), None)
    mdf       = mdf.set_index(id_col)
    origin    = np.datetime64("2013-01-01", "D")

    starts, props = [], []
    for wid in window_ids:
        if start_col and wid in mdf.index:
            row = mdf.loc[wid]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            starts.append(pd.Timestamp(str(row[start_col])[:10]))
        else:
            starts.append(pd.Timestamp(origin + np.timedelta64(int(wid) * 60, "D")))

        if wid in mdf.index:
            row = mdf.loc[wid]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            cls_vals = []
            for c in CLS_COLS:
                try:
                    v = row[c]
                    v = 0.0 if pd.isna(v) else float(v)
                except (TypeError, ValueError):
                    v = 0.0
                cls_vals.append(max(v, 0.0))
            total = sum(cls_vals)
            if total > 0:
                props.append([v / total for v in cls_vals])
            else:
                props.append([0.0] * N_WAY)
        else:
            props.append([0.0] * N_WAY)

    props_arr = np.array(props)
    starts_dt = [s.to_pydatetime() for s in starts]
    ends_dt   = [pd.Timestamp(s + pd.Timedelta(days=60)).to_pydatetime() for s in starts]
    date_nums = [date_to_mpl(s) for s in starts_dt]

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(14, 7), sharex=True,
        gridspec_kw={"height_ratios": [1, 3]})

    # Top: state colour bar with run labels
    runs = contiguous_runs(state_seq)
    for state, i0, i1 in runs:
        left  = date_to_mpl(starts_dt[i0])
        right = date_to_mpl(ends_dt[i1])
        ax_top.barh(0, right - left, left=left, height=0.8,
                    color=TAB10[state % 10], alpha=0.9, linewidth=0)
        ax_top.text((left + right) / 2, 0, str(state),
                    ha="center", va="center", fontsize=9,
                    fontweight="bold", color="white")

    # State boundary lines on both panels
    boundary_nums = []
    for idx in np.where(np.diff(state_seq) != 0)[0]:
        boundary_nums.append(date_to_mpl(ends_dt[idx]))
    for bn in boundary_nums:
        ax_top.axvline(bn, color="black", lw=1.0, ls="--", alpha=0.7)
        ax_bot.axvline(bn, color="black", lw=1.0, ls="--", alpha=0.7)

    ax_top.set_ylim(-0.5, 0.5); ax_top.set_yticks([])
    ax_top.set_title(f"HMM State (k={k})", fontsize=10, loc="left")

    # Bottom: stacked area
    ax_bot.stackplot(date_nums, props_arr.T,
                     labels=CLASS_NAMES, colors=CLASS_COLORS, alpha=0.85)
    ax_bot.set_ylabel("Class proportion", fontsize=11)
    ax_bot.set_ylim(0, 1)
    ax_bot.xaxis_date()
    ax_bot.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax_bot.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    ax_bot.legend(loc="upper left", fontsize=8, ncol=3, framealpha=0.7)
    ax_bot.grid(axis="y", alpha=0.3, ls="--")

    fig.autofmt_xdate(rotation=30, ha="right")
    fig.suptitle(
        f"State Timeline with 6-way Class Proportions (k={k})",
        fontsize=13, fontweight="bold")
    fig.tight_layout()
    savefig(fig, out_path, dpi=dpi)


def _f1_vs_distance(summary_df, out_path, dpi):
    df = summary_df[summary_df["distance"] != "pooled"].copy()
    df["distance"] = df["distance"].astype(int)
    df = df.sort_values("distance")

    fig, ax = plt.subplots(figsize=(8, 6))
    for kind, color, label in [
        ("within", COLORS_WA["within"], "Within-state"),
        ("across", COLORS_WA["across"], "Across-state"),
    ]:
        m  = df[f"{kind}_mean"].values
        lo = df[f"{kind}_lo"].values
        hi = df[f"{kind}_hi"].values
        d  = df["distance"].values
        ax.plot(d, m, color=color, lw=2, marker="o", ms=4, label=label)
        ax.fill_between(d, lo, hi, color=color, alpha=0.15)

    ax.set_xlabel("Temporal Distance |i − j| (windows)", fontsize=12)
    ax.set_ylabel("Macro F1", fontsize=12)
    ax.set_title("Cross-Window F1 vs. Temporal Distance\nSplit by HMM State Membership",
                 fontsize=13)
    ax.legend(fontsize=11); ax.grid(True, alpha=0.3, ls="--")
    fig.tight_layout()
    savefig(fig, out_path, dpi=dpi)


def _distance_conditioned_null(null_npz_path, out_path, dpi):
    d = load_npz(null_npz_path, "distance_conditioned_null.npz")
    if d is None:
        return
    null_gaps    = d["null_gaps"]
    observed_gap = float(d["observed_gap"])
    p_value      = float(np.mean(null_gaps >= observed_gap))

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(null_gaps, bins=60, color="#94A3B8", edgecolor="white",
            lw=0.3, alpha=0.85, label="Label-shuffle null")
    ax.axvline(observed_gap, color="#DC2626", lw=2.5,
               label=f"Observed gap = {observed_gap:.4f}  (p = {p_value:.4f})")
    ax.set_xlabel("Within − Across F1 Gap", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Distance-Conditioned Permutation Test", fontsize=13)
    ax.legend(fontsize=11); ax.grid(True, alpha=0.3, ls="--")
    fig.tight_layout()
    savefig(fig, out_path, dpi=dpi)


def _statepair_heatmap(f1_matrix, valid_ids_f1, state_seq, window_ids_dec,
                       k, out_path, dpi):
    wid_to_state = dict(zip(window_ids_dec, state_seq))
    valid_arr    = np.array(valid_ids_f1)

    sp_mat = np.full((k, k), np.nan)
    for si in range(k):
        for sj in range(k):
            vals = [f1_matrix[ri, ci]
                    for ri, wi in enumerate(valid_arr)
                    if wid_to_state.get(wi) == si
                    for ci, wj in enumerate(valid_arr)
                    if wid_to_state.get(wj) == sj and not np.isnan(f1_matrix[ri, ci])]
            if vals:
                sp_mat[si, sj] = np.mean(vals)

    fig, ax = plt.subplots(figsize=(max(5, k), max(4, k - 1)))
    im = ax.imshow(sp_mat, aspect="auto", cmap="RdYlGn",
                   vmin=np.nanmin(sp_mat), vmax=np.nanmax(sp_mat))
    plt.colorbar(im, ax=ax, label="Mean macro F1")
    labels = [f"State {s}" for s in range(k)]
    ax.set_xticks(range(k)); ax.set_xticklabels(labels, fontsize=10)
    ax.set_yticks(range(k)); ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("Test-window state", fontsize=11)
    ax.set_ylabel("Train-window state", fontsize=11)
    ax.set_title(f"Mean cross-window F1 by state pair (k={k}, 6-way)", fontsize=12)
    for si in range(k):
        for sj in range(k):
            v = sp_mat[si, sj]
            if not np.isnan(v):
                ax.text(sj, si, f"{v:.3f}", ha="center", va="center",
                        fontsize=9, color="black" if 0.3 < v < 0.85 else "white")
    for s in range(k):
        rect = mpatches.FancyBboxPatch((s - 0.5, s - 0.5), 1, 1,
                                       boxstyle="square,pad=0", lw=2.5,
                                       edgecolor="navy", facecolor="none")
        ax.add_patch(rect)
    fig.tight_layout()
    savefig(fig, out_path, dpi=dpi)


def _f1_heatmap_with_states(f1_matrix, valid_ids_f1, state_seq, window_ids_dec,
                             k, out_path, dpi):
    """Full F1 heatmap in natural window order, with state boundaries marked."""
    wid_to_state = {w: s for w, s in zip(window_ids_dec, state_seq)}
    valid_arr    = np.array(valid_ids_f1)
    N            = len(valid_arr)
    states_valid = np.array([wid_to_state.get(w, -1) for w in valid_arr])

    plot_mat = f1_matrix.copy().astype(float)
    np.fill_diagonal(plot_mat, np.nan)

    fig, ax = plt.subplots(figsize=(8, 6))

    im = ax.imshow(plot_mat, aspect="auto", cmap="RdYlGn",
                   vmin=np.nanmin(plot_mat), vmax=np.nanmax(plot_mat),
                   extent=[-0.5, N - 0.5, N - 0.5, -0.5])
    plt.colorbar(im, ax=ax, label="Macro F1", fraction=0.035, pad=0.02)

    # ── State boundary lines ──────────────────────────────────────────────────
    boundaries = [i - 0.5 for i in range(1, N)
                  if states_valid[i] != states_valid[i - 1]
                  and states_valid[i] >= 0 and states_valid[i - 1] >= 0]
    for b in boundaries:
        for spine in (
            dict(color="white",   lw=2.5, ls="-", alpha=1.0, zorder=3),
            dict(color="#1a1a1a", lw=1.2, ls="-", alpha=1.0, zorder=4),
        ):
            ax.axvline(b, **spine)
            ax.axhline(b, **spine)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("Test Window", fontsize=11)
    ax.set_ylabel("Train Window", fontsize=11)
    ax.set_title("Cross-Window Macro F1", fontsize=12)

    fig.tight_layout()
    savefig(fig, out_path, dpi=dpi)

def run_per_k(k, hmm_dir, perf_dir, wa_dir, manifest_path, out_dir, dpi):
    print(f"\n{'═'*60}")
    print(f"  Generating figures for k = {k}")
    print(f"{'═'*60}")

    k_out = os.path.join(out_dir, f"k{k}")
    os.makedirs(k_out, exist_ok=True)

    # Load HMM decode file
    decode_path = os.path.join(hmm_dir, f"final_decode_k{k}.npz")
    dec = load_npz(decode_path, f"final_decode_k{k}.npz")
    if dec is None:
        print(f"  [skip] all k={k} figures — decode file missing")
        return

    state_seq  = dec["state_seq"].astype(int)
    window_ids = dec["window_ids"].astype(int)
    trans_mat  = dec["transition_matrix"]

    start_dates, end_dates = window_calendar_dates(window_ids, manifest_path)

    _transition_matrix(trans_mat, k,
                       os.path.join(k_out, "transition_matrix.png"), dpi)

    _state_strip(state_seq, start_dates, end_dates, k,
                 os.path.join(k_out, "state_strip.png"), dpi)

    _state_timeline_seeds(dec, start_dates, end_dates, k,
                          os.path.join(k_out, "state_timeline_seeds.png"), dpi)

    _timeline_with_classes(dec, manifest_path, k,
                           os.path.join(k_out, "timeline_with_classes.png"), dpi)

    # Within / across figures
    wa_k_dir    = os.path.join(wa_dir, f"k{k}")
    summary_csv = os.path.join(wa_k_dir, "within_across_summary.csv")
    null_npz    = os.path.join(wa_k_dir, "distance_conditioned_null.npz")

    if os.path.exists(summary_csv):
        _f1_vs_distance(pd.read_csv(summary_csv),
                        os.path.join(k_out, "f1_vs_distance.png"), dpi)
    else:
        print(f"    [skip] f1_vs_distance — {summary_csv} not found")

    _distance_conditioned_null(null_npz,
                               os.path.join(k_out, "distance_conditioned_null.png"), dpi)

    # State-pair and sorted-heatmap figures (need cross-window F1 matrix)
    f1d = load_npz(os.path.join(perf_dir, "cross_window_f1.npz"))
    if f1d is not None:
        f1_matrix    = f1d["f1_matrix"]
        valid_ids_f1 = f1d["valid_ids"].tolist()
        wids_list    = window_ids.tolist()

        _statepair_heatmap(f1_matrix, valid_ids_f1, state_seq, wids_list, k,
                           os.path.join(k_out, "statepair_heatmap.png"), dpi)

        _f1_heatmap_with_states(f1_matrix, valid_ids_f1, state_seq, wids_list, k,
                                os.path.join(k_out, "f1_heatmap_with_states.png"), dpi)
    else:
        print("    [skip] statepair_heatmap and f1_heatmap_with_states — "
              "cross_window_f1.npz not found")


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"\n{'═'*60}")
    print(f"  generate_all_figures.py")
    print(f"  output_dir : {args.output_dir}")
    print(f"  k values   : {args.k}")
    print(f"  n_way      : {N_WAY}")
    print(f"  dpi        : {args.dpi}")
    print(f"{'═'*60}")

    # Global (k-independent) figures
    figure_window_distributions(args.output_dir, args.dpi)
    figure_cross_window_heatmap(args.perf_dir, args.output_dir, args.dpi)
    figure_model_selection(args.hmm_dir, args.output_dir, args.dpi)
    figure_pca_sanity(args.pca_npz, args.output_dir, args.dpi)

    # Per-k figures
    for k in args.k:
        run_per_k(k, args.hmm_dir, args.perf_dir, args.wa_dir,
                  args.manifest, args.output_dir, args.dpi)

    print(f"\n{'═'*60}")
    print(f"  Done.  All figures in: {args.output_dir}/")
    print(f"{'═'*60}\n")


if __name__ == "__main__":
    main()