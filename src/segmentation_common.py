"""
src/segmentation_common.py

Shared helpers for the reviewer-requested baseline segmentations that are
compared against the weight-space HMM:

    hmm_on_class_dist.py   — HMM on the per-window class distribution
    kmeans_on_weights.py   — k-means on the per-window weight centroids
    pelt_on_weights.py     — PELT change-point detection on the weight trajectory

Every baseline writes a decode file in EXACTLY the same format that
src/fit_hmm_decode.py produces, so the existing evaluation scripts run on the
baselines unchanged:

    python src/within_across_states.py  --decode_npz <baseline>.npz --f1_npz ...
    python src/check_equal_windows.py   --decode_npz <baseline>.npz --f1_npz ...
    python src/state_pair_correlation.py --decode_npz <baseline>.npz --f1_npz ...

The whole point of these baselines is to hold the *features* and the *evaluation*
fixed and vary only the *segmentation mechanism*, isolating what the HMM
machinery contributes over (a) ordinary clustering of the same weight centroids,
(b) a segmentation that comes from the class distribution directly, and
(c) a simpler contiguity-respecting change-point method.

To keep features identical to the HMM, k-means and PELT operate on the SAME
per-window centroid sequence the HMM was fit on: the across-seed mean of the
z-scored PCA vectors at each window (built here exactly as fit_hmm_decode.py
builds it).

Requirements: numpy, pandas, matplotlib
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CLS_COLS = [f"cls_{i}" for i in range(6)]
CLASS_NAMES = {
    0: "True", 1: "Satire", 2: "False Connection",
    3: "Imposter Content", 4: "Manipulated Content", 5: "Misleading Content",
}
TAB10 = plt.cm.tab10.colors


# ---------------------------------------------------------------------------
# Loading the weight-PCA representation (mirrors fit_hmm_decode.load_data)
# ---------------------------------------------------------------------------

def load_weights_pca(npz_path):
    """Load the weight-PCA bundle written by extract_weights_pca.py."""
    data       = np.load(npz_path)
    C          = data["C"].astype(np.float64)            # (N_windows, D)  unused here
    Z_scaled   = data["Z_scaled"].astype(np.float64)     # (N_windows*N_seeds, D)
    window_ids = data["window_ids"].astype(int)
    seed_ids   = data["seed_ids"].astype(int)
    return C, Z_scaled, window_ids, seed_ids


def build_centroid_sequence(Z_scaled, window_ids, seed_ids):
    """
    Replicates fit_hmm_decode.build_seed_sequences + compute_centroid_sequence:
    group per-(window, seed) PCA vectors by seed, then average across seeds at
    each window position.

    Returns
    -------
    centroid_seq     : (N_windows, D) float64   — identical to the HMM's input
    sorted_window_ids: (N_windows,)  int        — chronological window order
    """
    seeds = sorted(set(seed_ids.tolist()))
    seed_seqs = {}
    ref_wins = None
    for s in seeds:
        mask  = seed_ids == s
        order = np.argsort(window_ids[mask])
        seed_seqs[s] = Z_scaled[mask][order]
        wins = np.sort(window_ids[mask])
        if ref_wins is None:
            ref_wins = wins
        else:
            assert np.array_equal(wins, ref_wins), \
                f"Seed {s} has different window IDs than seed {seeds[0]}"

    stacked = np.stack([seed_seqs[s] for s in seeds], axis=0)  # (n_seeds, N, D)
    centroid_seq = stacked.mean(axis=0)                         # (N, D)
    return centroid_seq, ref_wins


# ---------------------------------------------------------------------------
# Loading the per-window class distribution (mirrors state_pair_correlation)
# ---------------------------------------------------------------------------

def load_class_distribution(manifest_path, window_ids):
    """
    Build the (N_windows, 6) matrix of class *proportions* per window, taken
    from the cls_0..cls_5 count columns of the window manifest. Rows are
    aligned to `window_ids` (the chronological order used everywhere else).

    Returns
    -------
    props     : (N_windows, 6) float64  — rows sum to 1
    counts    : (N_windows, 6) float64  — raw counts (for reporting)
    """
    manifest = pd.read_csv(manifest_path, parse_dates=["start_date", "end_date"])
    manifest = manifest.set_index("window_local_idx").loc[window_ids].reset_index()
    for c in CLS_COLS:
        if c not in manifest.columns:
            manifest[c] = 0
        manifest[c] = manifest[c].fillna(0).astype(float)

    counts = manifest[CLS_COLS].to_numpy(dtype=np.float64)      # (N, 6)
    totals = counts.sum(axis=1, keepdims=True)
    totals[totals == 0] = 1.0
    props = counts / totals
    return props, counts


def load_window_dates(manifest_path, window_ids):
    """Return mid-point dates for plotting, aligned to window_ids."""
    manifest = pd.read_csv(manifest_path, parse_dates=["start_date", "end_date"])
    manifest = manifest.set_index("window_local_idx").loc[window_ids].reset_index()
    mid = manifest["start_date"] + (manifest["end_date"] - manifest["start_date"]) / 2
    return mid.to_numpy()


# ---------------------------------------------------------------------------
# Label canonicalisation + decode-file writer
# ---------------------------------------------------------------------------

def relabel_by_first_appearance(labels):
    """
    Remap integer labels so that the first window has label 0, the next new
    label is 1, etc. This is a bijection, so it leaves every same-group /
    different-group relationship (the only thing the within/across test uses)
    untouched; it just makes timelines read left-to-right.
    """
    labels = np.asarray(labels)
    mapping = {}
    out = np.empty_like(labels)
    nxt = 0
    for i, lab in enumerate(labels):
        if lab not in mapping:
            mapping[lab] = nxt
            nxt += 1
        out[i] = mapping[lab]
    return out


def contiguous_runs(labels):
    """Run-length encode a label sequence -> list of (label, start, end_incl)."""
    labels = np.asarray(labels)
    runs = []
    i = 0
    N = len(labels)
    while i < N:
        j = i
        while j + 1 < N and labels[j + 1] == labels[i]:
            j += 1
        runs.append((int(labels[i]), i, j))
        i = j + 1
    return runs


def group_pca_centroids(state_seq, centroid_seq, k):
    """
    Per-group centroid in the z-scored PCA weight space. Stored as `means` so
    state_pair_correlation.py's weight-space (PCA-distance) correlation works
    uniformly across all baselines, regardless of which features drove the
    segmentation.
    """
    D = centroid_seq.shape[1]
    means = np.zeros((k, D), dtype=np.float64)
    for s in range(k):
        mask = state_seq == s
        if mask.any():
            means[s] = centroid_seq[mask].mean(axis=0)
    return means


def save_decode(out_path, *, state_seq, window_ids, k, centroid_seq,
                method, extra=None):
    """
    Write a decode .npz compatible with within_across_states.py,
    check_equal_windows.py and state_pair_correlation.py.

    Always writes: state_seq, window_ids, k, means, centroid_seq, method.
    `extra` is a dict of additional method-specific arrays to store.
    """
    means = group_pca_centroids(state_seq, centroid_seq, k)
    payload = dict(
        state_seq    = np.asarray(state_seq, dtype=int),
        window_ids   = np.asarray(window_ids, dtype=int),
        k            = np.int32(k),
        means        = means,                 # (k, D) PCA-space group centroids
        centroid_seq = np.asarray(centroid_seq, dtype=np.float64),
        method       = np.str_(method),
    )
    if extra:
        payload.update(extra)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    np.savez(out_path, **payload)
    return out_path


def print_segmentation(state_seq, window_ids, k, title):
    print(f"\n  {title}")
    print(f"    state_seq : {np.asarray(state_seq).tolist()}")
    runs = contiguous_runs(state_seq)
    print(f"    {len(runs)} contiguous run(s), {k} group(s):")
    for lab, i0, i1 in runs:
        print(f"      group {lab}  windows "
              f"{int(window_ids[i0]):03d}-{int(window_ids[i1]):03d}  "
              f"({i1 - i0 + 1} windows)")


def plot_timeline(state_seq, window_ids, k, title, out_path, dates=None):
    """Compact one-row strip: each window coloured by its assigned group."""
    state_seq = np.asarray(state_seq)
    N = len(state_seq)
    colors = [TAB10[i % 10] for i in range(k)]

    fig, ax = plt.subplots(figsize=(12, 1.8))
    for i, lab in enumerate(state_seq):
        ax.bar(i, 1, width=1.0, color=colors[int(lab)],
               edgecolor="white", linewidth=0.4)
    ax.set_xlim(-0.5, N - 0.5)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    if dates is not None:
        tick_idx = np.linspace(0, N - 1, min(N, 8)).astype(int)
        ax.set_xticks(tick_idx)
        ax.set_xticklabels([pd.Timestamp(dates[i]).strftime("%b %Y")
                            for i in tick_idx], rotation=30, ha="right", fontsize=8)
        ax.set_xlabel("Window (chronological)")
    else:
        ax.set_xlabel("Window index")
    ax.set_title(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"    Saved: {out_path}")