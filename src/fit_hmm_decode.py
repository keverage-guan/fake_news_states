"""
src/fit_hmm_decode.py

Final HMM decode at k=6 (or --k <chosen>).

Procedure
---------
1.  Load per-window, per-seed PCA vectors Z_scaled from weights_pca.npz.
2.  Build per-seed sequences, then compute a single centroid sequence by
    averaging PCA vectors across seeds at each window position.
3.  Fit a GaussianHMM(n_components=k, covariance_type='diag') with n_inits
    random initialisations on the centroid sequence (n_windows × D) — the
    best-LL model is kept, identical selection criterion to the BIC+ARI pass
    in fit_hmm_select_states.py.
4.  Decode the Viterbi state sequence over the centroid sequence directly
    (no per-seed decoding, no majority vote).
5.  Save final_decode_k{k}.npz with state_seq, window_ids, log_likelihood,
    transition_matrix, means, covars.
6.  Produce three plots:
      state_timeline_k{k}.png  — centroid state bar vs calendar date
      state_strip_k{k}.png     — compact dot/step strip
      transition_matrix_k{k}.png — transition matrix heatmap

Usage
-----
    python src/fit_hmm_decode.py \\
        --input_npz   data/hmm_weights/weights_pca.npz \\
        --output_dir  data/hmm_hmm \\
        --manifest    data/splits/hmm_windows/manifest.csv \\
        --k           6 \\
        --n_inits     50 \\
        --n_iter      200

    # Minimal (defaults match agreed design):
    python src/fit_hmm_decode.py

Requirements: hmmlearn, numpy, matplotlib, pandas
"""

import os
import sys
import time
import argparse
import warnings
import pickle

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.dates as mdates
from hmmlearn import hmm

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*KMeans.*")
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", message=".*did not converge.*")

# ── Palette (tab10, consistent with selection plots) ──────────────────────────

TAB10 = plt.cm.tab10.colors


# ── Timing helper ─────────────────────────────────────────────────────────────

def elapsed(t0):
    s = time.time() - t0
    return f"{int(s // 60):02d}:{s % 60:05.2f}"


# ── Data loading (mirrors select_states.py exactly) ───────────────────────────

def load_data(npz_path):
    data       = np.load(npz_path)
    C          = data["C"].astype(np.float64)
    Z_scaled   = data["Z_scaled"].astype(np.float64)
    window_ids = data["window_ids"].astype(int)
    seed_ids   = data["seed_ids"].astype(int)
    return C, Z_scaled, window_ids, seed_ids


def build_seed_sequences(Z_scaled, window_ids, seed_ids):
    """
    Group per-(window, seed) PCA vectors into per-seed sequences.
    Returns dict: seed_idx -> (n_windows, n_components) sorted by window.
    Also returns the sorted unique window IDs (shared across all seeds).
    """
    seeds     = sorted(set(seed_ids))
    seed_seqs = {}
    for s in seeds:
        mask  = seed_ids == s
        order = np.argsort(window_ids[mask])
        seed_seqs[s] = Z_scaled[mask][order]

    # All seeds must share the same window ordering
    ref_wins = None
    for s in seeds:
        mask  = seed_ids == s
        wins  = np.sort(window_ids[mask])
        if ref_wins is None:
            ref_wins = wins
        else:
            assert np.array_equal(wins, ref_wins), \
                f"Seed {s} has different window IDs than seed {seeds[0]}"

    return seed_seqs, ref_wins   # ref_wins: (n_windows,) sorted


def compute_centroid_sequence(seed_seqs, seeds):
    """
    Average PCA vectors across seeds at each window position.

    Parameters
    ----------
    seed_seqs : dict  seed -> (n_windows, D)
    seeds     : list of seed keys (order does not matter)

    Returns
    -------
    centroid : (n_windows, D) float64 — mean across seeds per window
    """
    stacked = np.stack([seed_seqs[s] for s in seeds], axis=0)  # (n_seeds, n_windows, D)
    return stacked.mean(axis=0)                                  # (n_windows, D)


# ── HMM helpers (mirrors select_states.py) ────────────────────────────────────

def make_hmm(k, n_iter, random_state, covariance_type="diag"):
    return hmm.GaussianHMM(
        n_components=k,
        covariance_type=covariance_type,
        n_iter=n_iter,
        tol=1e-5,
        random_state=random_state,
        verbose=False,
    )


def fit_best(centroid_seq, k, n_inits, n_iter, covariance_type="diag"):
    """
    Fit n_inits HMMs on the centroid sequence (n_windows × D).
    Returns the best model (highest training LL) and that LL.
    Raises RuntimeError if every init fails.
    """
    X       = centroid_seq                        # (n_windows, D)
    lengths = [X.shape[0]]
    best_model = None
    best_ll    = -np.inf
    first_exc  = None

    for seed in range(n_inits):
        model = make_hmm(k, n_iter, random_state=seed * 100 + k,
                         covariance_type=covariance_type)
        try:
            model.fit(X, lengths)
            ll = model.score(X, lengths)
        except Exception as e:
            if first_exc is None:
                first_exc = e
            continue
        if ll > best_ll:
            best_ll    = ll
            best_model = model

    if best_model is None:
        raise RuntimeError(
            f"All {n_inits} HMM inits failed. "
            f"First exception: {type(first_exc).__name__}: {first_exc}"
        )
    return best_model, best_ll


# ── Viterbi decode on centroid sequence ───────────────────────────────────────

def decode_centroid(model, centroid_seq):
    """
    Run Viterbi on the centroid sequence.

    Returns
    -------
    state_seq : (n_windows,) int32
    """
    _, state_seq = model.decode(
        centroid_seq,
        lengths=[centroid_seq.shape[0]],
        algorithm="viterbi",
    )
    return state_seq.astype(np.int32)


# ── Window → calendar date mapping (unchanged from original) ──────────────────

def load_window_dates(manifest_path, window_ids):
    if manifest_path and os.path.isfile(manifest_path):
        try:
            df = pd.read_csv(manifest_path)
            df.columns = [c.lower().strip() for c in df.columns]
            id_col    = next(c for c in df.columns if "window" in c and "id" in c)
            date_cols = [c for c in df.columns if "date" in c or "start" in c]
            end_cols  = [c for c in df.columns if "end" in c]
            df = df.set_index(id_col)
            starts, ends, mids = [], [], []
            for wid in window_ids:
                if wid in df.index:
                    s = pd.Timestamp(df.loc[wid, date_cols[0]])
                    e = (pd.Timestamp(df.loc[wid, end_cols[0]])
                         if end_cols
                         else s + pd.Timedelta(days=60))
                else:
                    s = pd.Timestamp("2013-01-01") + pd.Timedelta(days=int(wid) * 60)
                    e = s + pd.Timedelta(days=60)
                starts.append(s)
                ends.append(e)
                mids.append(s + (e - s) / 2)
            return (
                np.array(starts, dtype="datetime64[D]"),
                np.array(mids,   dtype="datetime64[D]"),
                np.array(ends,   dtype="datetime64[D]"),
            )
        except Exception as exc:
            print(f"  [warn] Could not parse manifest ({exc}); using synthetic dates.")

    origin = np.datetime64("2013-01-01", "D")
    starts = np.array([origin + np.timedelta64(int(w) * 60, "D") for w in window_ids])
    ends   = starts + np.timedelta64(60, "D")
    mids   = starts + np.timedelta64(30, "D")
    return starts, mids, ends


# ── Helper: contiguous run-length encoding ────────────────────────────────────

def _contiguous_runs(seq):
    runs = []
    i = 0
    while i < len(seq):
        j = i
        while j < len(seq) and seq[j] == seq[i]:
            j += 1
        runs.append((seq[i], i, j - 1))
        i = j
    return runs


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_state_timeline(state_seq, window_ids, start_dates, end_dates, k, out_path):
    """
    Single-row timeline of the centroid-derived Viterbi state sequence.
    Colored by state; state-boundary dashed lines drawn at run transitions.
    """
    n_win = len(window_ids)

    fig, ax = plt.subplots(figsize=(14, 2.2))

    # Boundary positions (date-axis numeric)
    boundary_dates = []
    for idx in np.where(np.diff(state_seq) != 0)[0]:
        boundary_dates.append(
            mdates.date2num(pd.Timestamp(end_dates[idx]).to_pydatetime())
        )

    for i in range(n_win):
        s_dt  = pd.Timestamp(start_dates[i]).to_pydatetime()
        e_dt  = pd.Timestamp(end_dates[i]).to_pydatetime()
        width = (e_dt - s_dt).days
        ax.barh(y=0, width=width,
                left=mdates.date2num(s_dt),
                height=0.8,
                color=TAB10[state_seq[i] % 10],
                alpha=0.85, linewidth=0)

    for bd in boundary_dates:
        ax.axvline(bd, color="black", linewidth=1.2,
                   linestyle="--", alpha=0.7, zorder=5)

    ax.set_yticks([0])
    ax.set_yticklabels(["centroid"], fontsize=9)
    ax.set_ylim(-0.6, 0.6)
    ax.xaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    ax.grid(axis="x", alpha=0.25, linestyle=":")
    fig.autofmt_xdate(rotation=30, ha="right")
    ax.set_xlabel("Date", fontsize=11)

    patches = [mpatches.Patch(color=TAB10[s % 10], label=f"State {s}")
               for s in range(k)]
    ax.legend(handles=patches, loc="upper right", fontsize=8,
              framealpha=0.9, title="HMM State", title_fontsize=8,
              ncol=min(k, 5))

    fig.suptitle(
        f"HMM State Timeline  (k={k}, Viterbi on seed-centroid sequence)",
        fontsize=12, y=1.02,
    )
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_state_summary_strip(state_seq, window_ids, mid_dates, k, out_path):
    fig, ax = plt.subplots(figsize=(14, 3))
    x_vals  = np.arange(len(window_ids))
    colors  = [TAB10[s % 10] for s in state_seq]

    ax.scatter(x_vals, state_seq, c=colors, s=80, zorder=3,
               edgecolors="white", linewidths=0.5)
    ax.step(x_vals, state_seq, where="mid",
            color="gray", linewidth=0.8, alpha=0.6, zorder=2)

    for idx in np.where(np.diff(state_seq) != 0)[0]:
        ax.axvline(idx + 0.5, color="black", linewidth=1.2,
                   linestyle="--", alpha=0.6)

    tick_step = max(1, len(window_ids) // 8)
    tick_idxs = list(range(0, len(window_ids), tick_step))
    ax.set_xticks(tick_idxs)
    ax.set_xticklabels(
        [pd.Timestamp(mid_dates[i]).strftime("%b %Y") for i in tick_idxs],
        rotation=30, ha="right", fontsize=8,
    )
    ax.set_yticks(range(k))
    ax.set_yticklabels([f"State {i}" for i in range(k)], fontsize=9)
    ax.set_ylabel("HMM State", fontsize=11)
    ax.set_xlabel("Window (chronological)", fontsize=11)
    ax.set_title(f"Viterbi State Sequence — seed centroid  (k={k})", fontsize=12)
    ax.grid(axis="x", alpha=0.3, linestyle=":")
    ax.set_xlim(-0.5, len(window_ids) - 0.5)
    ax.set_ylim(-0.5, k - 0.5)

    patches = [mpatches.Patch(color=TAB10[s % 10], label=f"State {s}")
               for s in range(k)]
    ax.legend(handles=patches, loc="upper right", fontsize=8, ncol=k,
              framealpha=0.9, title="HMM State", title_fontsize=8)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_transition_matrix(trans_matrix, k, out_path):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(trans_matrix, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, label="Transition probability")
    ax.set_xticks(range(k)); ax.set_yticks(range(k))
    ax.set_xticklabels([f"S{i}" for i in range(k)])
    ax.set_yticklabels([f"S{i}" for i in range(k)])
    ax.set_xlabel("To state", fontsize=11)
    ax.set_ylabel("From state", fontsize=11)
    ax.set_title(f"HMM Transition Matrix  (k={k})", fontsize=12)
    for i in range(k):
        for j in range(k):
            val = trans_matrix[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=9, color="white" if val > 0.6 else "black")
        rect = mpatches.FancyBboxPatch(
            (i - 0.5, i - 0.5), 1, 1,
            boxstyle="square,pad=0",
            linewidth=2, edgecolor="red", facecolor="none",
        )
        ax.add_patch(rect)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args):
    t_global = time.time()
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print(f"  src/fit_hmm_decode.py  —  k={args.k}, n_inits={args.n_inits}")
    print("=" * 60)

    # ── 1. Load Z_scaled and build centroid sequence ──────────────────────────
    print(f"\n[1] Loading Z_scaled from: {args.input_npz}")
    C, Z_scaled, window_ids, seed_ids = load_data(args.input_npz)
    seed_seqs, sorted_window_ids = build_seed_sequences(Z_scaled, window_ids, seed_ids)
    seeds = sorted(seed_seqs.keys())

    n_windows = seed_seqs[seeds[0]].shape[0]
    D         = seed_seqs[seeds[0]].shape[1]
    n_seeds   = len(seeds)

    centroid_seq = compute_centroid_sequence(seed_seqs, seeds)  # (n_windows, D)

    print(f"    Z_scaled shape   : {Z_scaled.shape}  (obs × PCA dims)")
    print(f"    Seeds            : {seeds}  ({n_seeds} total)")
    print(f"    Windows          : {n_windows}")
    print(f"    PCA dims (D)     : {D}")
    print(f"    Centroid shape   : {centroid_seq.shape}  (mean across {n_seeds} seeds)")

    # ── 2. Fit HMM on centroid sequence ───────────────────────────────────────
    print(f"\n[2] Fitting GaussianHMM(k={args.k}, cov={args.covariance_type}) "
          f"with {args.n_inits} random inits on centroid sequence …")
    t_fit = time.time()
    best_model, best_ll = fit_best(
        centroid_seq, args.k, args.n_inits, args.n_iter, args.covariance_type,
    )
    print(f"    Best log-likelihood : {best_ll:.4f}  (wall: {elapsed(t_fit)})")

    # ── 3. Viterbi decode on centroid sequence ────────────────────────────────
    print(f"\n[3] Viterbi decoding on centroid sequence …")
    state_seq = decode_centroid(best_model, centroid_seq)
    print(f"    state_seq : {state_seq.tolist()}")

    runs = _contiguous_runs(state_seq)
    print(f"\n    Run-length encoded ({len(runs)} runs):")
    for state, i0, i1 in runs:
        print(f"      State {state}  windows "
              f"{sorted_window_ids[i0]:03d}–{sorted_window_ids[i1]:03d}  "
              f"({i1 - i0 + 1} windows)")

    # ── 4. Extract model parameters ───────────────────────────────────────────
    trans_matrix = best_model.transmat_
    means        = best_model.means_
    startprob    = best_model.startprob_
    covars       = best_model.covars_

    print(f"\n    Transition matrix (rounded):")
    for i in range(args.k):
        row = "  ".join(f"{p:.3f}" for p in trans_matrix[i])
        print(f"      S{i}: [{row}]")

    # ── 5. Save .npz and pickle ───────────────────────────────────────────────
    out_npz = os.path.join(args.output_dir, f"final_decode_k{args.k}.npz")
    np.savez(
        out_npz,
        state_seq         = state_seq,            # (n_windows,)
        centroid_seq      = centroid_seq,          # (n_windows, D)
        window_ids        = sorted_window_ids,
        log_likelihood    = np.float64(best_ll),
        transition_matrix = trans_matrix,
        means             = means,
        covars            = covars,
        startprob         = startprob,
        k                 = np.int32(args.k),
        covariance_type   = np.str_(args.covariance_type),
    )
    print(f"\n[4] Saved: {out_npz}")

    out_pkl = os.path.join(args.output_dir, f"final_hmm_k{args.k}.pkl")
    with open(out_pkl, "wb") as f:
        pickle.dump(best_model, f)
    print(f"    Saved: {out_pkl}")

    # ── 6. Calendar dates ─────────────────────────────────────────────────────
    start_dates, mid_dates, end_dates = load_window_dates(
        args.manifest, sorted_window_ids
    )

    # ── 7. Plots ──────────────────────────────────────────────────────────────
    print(f"\n[5] Generating plots …")

    plot_state_timeline(
        state_seq, sorted_window_ids, start_dates, end_dates,
        k        = args.k,
        out_path = os.path.join(args.output_dir, f"state_timeline_k{args.k}.png"),
    )
    plot_state_summary_strip(
        state_seq, sorted_window_ids, mid_dates,
        k        = args.k,
        out_path = os.path.join(args.output_dir, f"state_strip_k{args.k}.png"),
    )
    plot_transition_matrix(
        trans_matrix,
        k        = args.k,
        out_path = os.path.join(args.output_dir, f"transition_matrix_k{args.k}.png"),
    )

    # ── 8. Text summary ───────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"  DECODE SUMMARY  (k={args.k})")
    print(f"{'=' * 60}")
    print(f"  Log-likelihood  : {best_ll:.4f}")
    print(f"  n_windows       : {n_windows}")
    print(f"  n_seeds         : {n_seeds}  (averaged into centroid before fitting)")
    print(f"  PCA dims (D)    : {D}")
    print(f"  Covariance type : {args.covariance_type}")
    print(f"\n  State assignments:")
    state_counts = {s: 0 for s in range(args.k)}
    for s in state_seq:
        state_counts[s] += 1
    for s in range(args.k):
        n = state_counts[s]
        pct = 100 * n / n_windows
        bar = "█" * n + "░" * (n_windows - n)
        print(f"    State {s}: {n:3d} windows ({pct:5.1f}%)  {bar}")

    print(f"\n  Runs: {len(runs)} contiguous state segments")
    for state, i0, i1 in runs:
        n_run = i1 - i0 + 1
        try:
            s_label = pd.Timestamp(start_dates[i0]).strftime("%b %Y")
            e_label = pd.Timestamp(end_dates[i1]).strftime("%b %Y")
            date_str = f"  [{s_label} – {e_label}]"
        except Exception:
            date_str = ""
        print(f"    W{sorted_window_ids[i0]:03d}–W{sorted_window_ids[i1]:03d}  "
              f"State {state}  ({n_run} windows){date_str}")

    print(f"\n  Total wall time : {elapsed(t_global)}")
    print(f"{'=' * 60}")
    print()
    print("  Next: src/cross_window_eval.py  —  build the 35×35 F1 matrix")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Final HMM decode at chosen k using seed-centroid sequence. "
                    "Averages PCA vectors across seeds per window before fitting."
    )
    p.add_argument("--input_npz",
                   default="data/hmm_weights/weights_pca.npz",
                   help="Path to weights_pca.npz (from src/extract_weights_pca.py)")
    p.add_argument("--output_dir",
                   default="data/hmm_hmm",
                   help="Directory for outputs (created if absent)")
    p.add_argument("--manifest",
                   default="data/splits/hmm_windows/manifest.csv",
                   help="Window manifest CSV (for calendar dates). "
                        "Falls back to synthetic dates if absent.")
    p.add_argument("--k",       type=int, default=6,
                   help="Number of HMM states (chosen from selection step)")
    p.add_argument("--n_inits", type=int, default=50,
                   help="Number of random HMM initialisations (take best LL)")
    p.add_argument("--n_iter",  type=int, default=200,
                   help="Baum-Welch EM iterations per init")
    p.add_argument("--covariance_type", default="diag",
                   choices=["diag", "full", "tied", "spherical"],
                   help="HMM covariance type — must match selection step")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)