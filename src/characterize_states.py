"""
src/characterise_states.py

Produces a rich characterisation of each HMM latent state.

For each state (and overall):
  - Date range of member windows (start of first → end of last)
  - Number of windows  + total samples
  - 6-way class distribution (counts + %)
  - Top-10 subreddits by share (requires raw window TSVs)
  - Per-state emission mean in PCA space  (from HMM, 3 dims)
  - Per-state emission mean in *unscaled* PCA space
  - PCA loadings matrix  C  (printed once, saved to CSV)
  - Cross-window generalisation:  mean within-state F1  (diagonal blocks
    of the F1 matrix) vs. mean across-state F1, per state

Outputs
-------
  <output_dir>/state_report.txt       human-readable terminal-width report
  <output_dir>/state_summary.csv      one row per state
  <output_dir>/state_class_dist.csv   class % per state (wide)
  <output_dir>/state_subreddits.csv   top-N subreddits per state
  <output_dir>/pca_loadings.csv       C matrix  (n_components × n_weights)

Usage
-----
  python src/characterise_states.py

  # Override paths:
  python src/characterise_states.py \\
      --decode_npz   data/hmm_hmm/6way/final_decode_k6.npz \\
      --weights_npz  data/hmm_weights/6way/weights_pca.npz \\
      --manifest     data/splits/hmm_windows/manifest.csv \\
      --splits_dir   data/splits/hmm_windows \\
      --f1_npz       data/hmm_perf/6way/k6/cross_window_f1.npz \\
      --output_dir   data/state_characterisation \\
      --top_subreddits 10
"""

import os
import argparse
import textwrap

import numpy as np
import pandas as pd


# ── Label maps ────────────────────────────────────────────────────────────────

CLASS_NAMES = {
    0: "True",
    1: "Satire",
    2: "False Connection",
    3: "Imposter Content",
    4: "Manipulated Content",
    5: "Misleading Content",
}
# manifest columns: cls_0 … cls_5
CLS_COLS = [f"cls_{i}" for i in range(6)]


# ── Argument parsing ──────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--decode_npz",    default="data/hmm_hmm/6way/final_decode_k8.npz")
    p.add_argument("--weights_npz",   default="data/hmm_weights/6way/weights_pca.npz")
    p.add_argument("--manifest",      default="data/splits/hmm_windows/HMM_windows_manifest.csv")
    p.add_argument("--splits_dir",    default="data/splits/hmm_windows")
    p.add_argument("--f1_npz",        default="data/hmm_perf/6way/k8/cross_window_f1.npz",
                   help="cross_window_f1.npz from merge_cross_window.py")
    p.add_argument("--output_dir",    default="data/state_characterisation/k8")
    p.add_argument("--top_subreddits", type=int, default=10)
    p.add_argument("--no_tsv",        action="store_true",
                   help="Skip loading raw window TSVs (subreddit breakdown skipped)")
    return p.parse_args()


# ── Helpers ───────────────────────────────────────────────────────────────────

def hline(char="─", width=80):
    return char * width


def section(title, width=80):
    pad = width - len(title) - 4
    return f"╔══ {title} {'═' * pad}╗"


def fmt_pct(v):
    return f"{v:6.1f}%"


def load_subreddits(splits_dir, manifest, state_seq):
    """
    Load raw window TSVs and return a dict:  window_idx -> pd.Series of
    subreddit counts (sorted descending).
    Returns None if TSVs not found.
    """
    results = {}
    for _, row in manifest.iterrows():
        idx  = int(row["window_local_idx"])
        path = os.path.join(splits_dir, row["filename"])
        if not os.path.exists(path):
            print(f"  [warn] TSV not found: {path}  — subreddit breakdown skipped")
            return None
        df = pd.read_csv(path, sep="\t", usecols=["subreddit"], low_memory=False)
        results[idx] = df["subreddit"].value_counts()
    return results


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    WIDTH = 88

    # ── 1. Load decode ─────────────────────────────────────────────────────
    dec        = np.load(args.decode_npz, allow_pickle=True)
    state_seq  = dec["state_seq"].astype(int)       # (N,)
    window_ids = dec["window_ids"].astype(int)       # (N,)
    k          = int(dec.get("k", state_seq.max() + 1))
    hmm_means  = dec["means"].astype(float)          # (k, n_pca)
    # transition matrix
    trans_mat  = dec.get("transition_matrix")
    if trans_mat is not None:
        trans_mat = np.array(trans_mat, dtype=float)

    n_pca = hmm_means.shape[1]
    N     = len(state_seq)
    print(f"Loaded decode: k={k}, N={N} windows, n_pca={n_pca}")

    # ── 2. Load PCA weights ────────────────────────────────────────────────
    wpca       = np.load(args.weights_npz, allow_pickle=True)
    C          = wpca["C"].astype(float)             # (n_pca, n_weight_dims)
    Z_scaled   = wpca["Z_scaled"].astype(float)      # (N*n_seeds, n_pca)  standardised
    # Recover scaler stats to un-standardise emission means
    # Z_scaled = (C @ w - mu) / sigma   per PCA component
    # We store the per-component mean/std used during z-scoring.
    # If not saved, approximate from Z_scaled itself.
    if "scaler_mean" in wpca and "scaler_std" in wpca:
        scaler_mean = wpca["scaler_mean"].astype(float)   # (n_pca,)
        scaler_std  = wpca["scaler_std"].astype(float)
    else:
        scaler_mean = Z_scaled.mean(axis=0)
        scaler_std  = Z_scaled.std(axis=0)
        print("  [info] scaler_mean/std not in npz — approximated from Z_scaled")

    # Un-standardised emission means  (in raw PCA score space)
    hmm_means_raw = hmm_means * scaler_std + scaler_mean   # (k, n_pca)

    print(f"PCA loadings C shape: {C.shape}")

    # ── 3. Load manifest ───────────────────────────────────────────────────
    manifest = pd.read_csv(args.manifest, parse_dates=["start_date", "end_date"])
    # Align to window_ids from decode (may be a subset)
    manifest = manifest.set_index("window_local_idx").loc[window_ids].reset_index()

    # Fill any missing cls_ columns with 0
    for c in CLS_COLS:
        if c not in manifest.columns:
            manifest[c] = 0
        else:
            manifest[c] = manifest[c].fillna(0).astype(int)

    # ── 4. Optionally load subreddit data ──────────────────────────────────
    if args.no_tsv:
        sub_data = None
    else:
        print("Loading window TSVs for subreddit breakdown …")
        sub_data = load_subreddits(args.splits_dir, manifest, state_seq)

    # ── 5. Load F1 matrix ──────────────────────────────────────────────────
    f1_available = False
    f1_matrix    = None
    if os.path.exists(args.f1_npz):
        f1_data   = np.load(args.f1_npz, allow_pickle=True)
        f1_matrix = f1_data["f1_matrix"].astype(float)   # (N, N)
        f1_available = True
        print(f"Loaded F1 matrix: {f1_matrix.shape}")
    else:
        print(f"  [warn] F1 matrix not found at {args.f1_npz} — F1 stats skipped")

    # ── 6. Build per-state summary ─────────────────────────────────────────
    lines  = []           # text report lines
    rows   = []           # CSV rows
    cls_rows  = []
    sub_rows  = []

    def pr(*a, **kw):
        lines.append(" ".join(str(x) for x in a))

    pr(hline("═", WIDTH))
    pr(f"  HMM STATE CHARACTERISATION  —  k={k}")
    pr(hline("═", WIDTH))

    # ── Overall summary ────────────────────────────────────────────────────
    pr()
    pr(f"  Windows : {N}   (window IDs {window_ids[0]}–{window_ids[-1]})")
    pr(f"  Date span: {manifest['start_date'].min().date()} → "
       f"{manifest['end_date'].max().date()}")
    pr(f"  PCA dims : {n_pca}   |   HMM emission means shape: {hmm_means.shape}")
    pr()

    # State sequence overview
    pr("  State sequence (window index → state):")
    seq_str = "  " + "  ".join(f"W{i:02d}→S{s}" for i, s in zip(window_ids, state_seq))
    for chunk in textwrap.wrap(seq_str, width=WIDTH - 2, subsequent_indent="    "):
        pr(chunk)
    pr()

    # Transition matrix
    if trans_mat is not None:
        pr("  Transition matrix  (row = from-state, col = to-state):")
        header = "        " + "  ".join(f"S{j:1d}" for j in range(k))
        pr(header)
        for i in range(k):
            row_str = f"  S{i:1d}  [ " + "  ".join(f"{trans_mat[i,j]:.3f}" for j in range(k)) + " ]"
            pr(row_str)
        pr()

    # ── Per-state blocks ───────────────────────────────────────────────────
    for s in range(k):
        mask     = state_seq == s
        win_idxs = window_ids[mask]              # global window IDs in this state
        local_idxs = np.where(mask)[0]           # positions in the N-window array
        state_manifest = manifest[mask]

        start_date = state_manifest["start_date"].min().date()
        end_date   = state_manifest["end_date"].max().date()
        n_windows  = mask.sum()
        total_samp = int(state_manifest["sampled_n"].sum())

        # class counts
        cls_counts = {c: int(state_manifest[f"cls_{c}"].sum()) for c in range(6)}
        cls_total  = sum(cls_counts.values())
        cls_pcts   = {c: (cls_counts[c] / cls_total * 100 if cls_total > 0 else 0.0)
                      for c in range(6)}

        # F1 within/across
        if f1_available:
            within_f1  = []
            across_f1  = []
            for i in local_idxs:
                for j in range(N):
                    if i == j:
                        continue
                    v = f1_matrix[i, j]
                    if np.isnan(v):
                        continue
                    if state_seq[j] == s:
                        within_f1.append(v)
                    else:
                        across_f1.append(v)
            wf = np.mean(within_f1) if within_f1 else float("nan")
            af = np.mean(across_f1) if across_f1 else float("nan")
            gap = wf - af
        else:
            wf = af = gap = float("nan")

        # ── Print block ────────────────────────────────────────────────
        pr(hline("─", WIDTH))
        pr(f"  STATE {s}  │  Windows: {n_windows}  │  "
           f"{start_date}  →  {end_date}  │  Samples: {total_samp:,}")
        pr(hline("─", WIDTH))

        pr()
        pr("  Window IDs in this state:", ", ".join(str(w) for w in sorted(win_idxs)))
        pr()

        # Class distribution
        pr("  6-way class distribution:")
        pr(f"    {'Class':<25} {'Count':>8}  {'%':>7}")
        pr(f"    {'─'*25} {'─'*8}  {'─'*7}")
        for c in range(6):
            bar_len = int(cls_pcts[c] / 2)
            bar     = "█" * bar_len
            pr(f"    {CLASS_NAMES[c]:<25} {cls_counts[c]:>8,}  {fmt_pct(cls_pcts[c])}  {bar}")
        pr()

        # PCA emission means (standardised)
        pr("  HMM emission mean  (Z-scaled PCA space):")
        pca_dim_labels = [f"PC{i+1}" for i in range(n_pca)]
        for i, (lbl, v) in enumerate(zip(pca_dim_labels, hmm_means[s])):
            pr(f"    {lbl}: {v:+.4f}")
        pr()

        # PCA emission means (raw / un-standardised)
        pr("  HMM emission mean  (raw PCA score space):")
        for i, (lbl, v) in enumerate(zip(pca_dim_labels, hmm_means_raw[s])):
            pr(f"    {lbl}: {v:+.6f}")
        pr()

        # F1 generalisation
        if f1_available:
            pr(f"  Cross-window F1 generalisation (macro F1):")
            pr(f"    Within-state mean F1  : {wf:.4f}  (n={len(within_f1):,} pairs)")
            pr(f"    Across-state mean F1  : {af:.4f}  (n={len(across_f1):,} pairs)")
            pr(f"    Gap (within − across) : {gap:+.4f}")
        pr()

        # Subreddits
        if sub_data is not None:
            # Aggregate subreddit counts across windows in this state
            agg = pd.Series(dtype=int)
            for local_i in local_idxs:
                win_idx = int(window_ids[local_i])
                # sub_data is keyed by window_local_idx (position 0…N-1)
                agg = agg.add(sub_data.get(local_i, pd.Series(dtype=int)),
                              fill_value=0)
            agg = agg.sort_values(ascending=False)
            total_posts = agg.sum()
            top_n = agg.head(args.top_subreddits)
            pr(f"  Top-{args.top_subreddits} subreddits (of {len(agg):,} unique):")
            pr(f"    {'Subreddit':<35} {'Posts':>7}  {'%':>7}")
            pr(f"    {'─'*35} {'─'*7}  {'─'*7}")
            for sr, cnt in top_n.items():
                pct = cnt / total_posts * 100
                pr(f"    {sr:<35} {int(cnt):>7,}  {fmt_pct(pct)}")
            pr()

            # save to sub_rows
            for rank, (sr, cnt) in enumerate(agg.head(args.top_subreddits).items(), 1):
                sub_rows.append({
                    "state": s, "rank": rank, "subreddit": sr,
                    "count": int(cnt), "pct": cnt / total_posts * 100
                })

        # CSV accumulation
        row = {
            "state":        s,
            "n_windows":    n_windows,
            "window_ids":   ";".join(str(w) for w in sorted(win_idxs)),
            "start_date":   str(start_date),
            "end_date":     str(end_date),
            "total_samples": total_samp,
            "within_f1_mean": round(wf, 4),
            "across_f1_mean": round(af, 4),
            "f1_gap":        round(gap, 4),
        }
        for i in range(n_pca):
            row[f"pca_mean_scaled_PC{i+1}"]   = round(hmm_means[s][i],     6)
            row[f"pca_mean_raw_PC{i+1}"]       = round(hmm_means_raw[s][i], 6)
        rows.append(row)

        cls_rows.append({
            "state": s,
            **{CLASS_NAMES[c]: round(cls_pcts[c], 2) for c in range(6)},
        })

    # ── PCA loadings ───────────────────────────────────────────────────────
    pr(hline("═", WIDTH))
    pr("  PCA LOADINGS  (C matrix)")
    pr(f"  Shape: {C.shape}  —  rows = PCs, cols = weight dimensions")
    pr()
    pr(f"  {'PC':<6}  {'Norm':>10}  {'Max |loading|':>15}  {'Argmax':>8}")
    pr(f"  {'─'*6}  {'─'*10}  {'─'*15}  {'─'*8}")
    for i in range(C.shape[0]):
        row_c = C[i]
        pr(f"  PC{i+1:<4}  {np.linalg.norm(row_c):>10.4f}  "
           f"{np.abs(row_c).max():>15.6f}  {np.abs(row_c).argmax():>8d}")
    pr()

    # Pairwise distances between state centroids (scaled space)
    if k > 1:
        pr("  Pairwise Euclidean distances between state centroids (Z-scaled):")
        header = "        " + "  ".join(f"  S{j}" for j in range(k))
        pr(header)
        for i in range(k):
            vals = []
            for j in range(k):
                d = np.linalg.norm(hmm_means[i] - hmm_means[j])
                vals.append(f"{d:5.3f}")
            pr(f"  S{i}  [ " + "  ".join(vals) + " ]")
        pr()

    pr(hline("═", WIDTH))

    # ── Write outputs ──────────────────────────────────────────────────────
    report_path = os.path.join(args.output_dir, "state_report.txt")
    with open(report_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\n── Saved report → {report_path}")

    summary_df = pd.DataFrame(rows)
    summary_path = os.path.join(args.output_dir, "state_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"── Saved summary → {summary_path}")

    cls_df = pd.DataFrame(cls_rows)
    cls_path = os.path.join(args.output_dir, "state_class_dist.csv")
    cls_df.to_csv(cls_path, index=False)
    print(f"── Saved class dist → {cls_path}")

    if sub_rows:
        sub_df = pd.DataFrame(sub_rows)
        sub_path = os.path.join(args.output_dir, "state_subreddits.csv")
        sub_df.to_csv(sub_path, index=False)
        print(f"── Saved subreddits → {sub_path}")

    # PCA loadings CSV  (can be very wide; save as PC × dim)
    pca_df = pd.DataFrame(C, index=[f"PC{i+1}" for i in range(C.shape[0])])
    pca_path = os.path.join(args.output_dir, "pca_loadings.csv")
    pca_df.to_csv(pca_path)
    print(f"── Saved PCA loadings → {pca_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()