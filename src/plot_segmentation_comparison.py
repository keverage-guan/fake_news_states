"""
src/plot_segmentation_comparison.py

Side-by-side comparison of all five window segmentations as stacked colour
strips, one row per method:

    1. HMM on weights        (the paper's method)
    2. k-means on weights
    3. PELT on weights
    4. HMM on class distribution
    5. Equal partition        (the original naive baseline)

Each row shows the 35 chronological windows coloured by the group each window
was assigned to under that method, with vertical lines marking the segment
boundaries so it is easy to see where the methods agree and where they diverge.
Colours are assigned per row (group 0 of one method is unrelated to group 0 of
another), ordered by first appearance so each row reads left-to-right.

The equal partition is computed here (contiguous equal-size blocks, the
remainder distributed to the first groups) exactly as in check_equal_windows.py,
so no decode file is needed for it. The other four rows are read from the decode
.npz files produced by fit_hmm_decode.py and the three baseline scripts.

Output: <output_dir>/segmentation_comparison.png  (default data/baselines/)

Usage
-----
  python src/plot_segmentation_comparison.py \
      --hmm_weights_decode data/hmm_hmm/6way/final_decode_k7.npz \
      --kmeans_decode      data/baselines/kmeans/final_decode_kmeans_k7.npz \
      --pelt_decode        data/baselines/pelt/final_decode_pelt_k7.npz \
      --classdist_decode   data/baselines/classdist/final_decode_classdist_k7.npz \
      --manifest           data/splits/hmm_windows/HMM_windows_manifest.csv \
      --output_dir         data/baselines

  # minimal (defaults match the layout above)
  python src/plot_segmentation_comparison.py

Requirements: numpy, pandas, matplotlib
"""

import os
import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from segmentation_common import (
    load_window_dates, relabel_by_first_appearance, contiguous_runs, TAB10,
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--hmm_weights_decode", default="data/hmm_hmm/6way/final_decode_k7.npz")
    p.add_argument("--kmeans_decode",      default="data/baselines/kmeans/final_decode_kmeans_k7.npz")
    p.add_argument("--pelt_decode",        default="data/baselines/pelt/final_decode_pelt_k7.npz")
    p.add_argument("--classdist_decode",   default="data/baselines/classdist/final_decode_classdist_k7.npz")
    p.add_argument("--manifest",           default="data/splits/hmm_windows/HMM_windows_manifest.csv")
    p.add_argument("--output_dir",         default="data/baselines")
    p.add_argument("--k", type=int, default=None,
                   help="Groups for the equal partition (default: k of HMM-weights decode)")
    p.add_argument("--dpi", type=int, default=200)
    return p.parse_args()


def equal_duration_labels(N, k):
    """N consecutive windows -> k contiguous equal-size groups; the remainder
    r = N % k is distributed to the first r groups (one extra each). Matches
    check_equal_windows.equal_duration_labels."""
    base, rem = divmod(N, k)
    sizes = [base + (1 if i < rem else 0) for i in range(k)]
    labels = np.empty(N, dtype=int)
    idx = 0
    for g, sz in enumerate(sizes):
        labels[idx:idx + sz] = g
        idx += sz
    return labels


def load_decode(path):
    """Return (state_seq relabelled by first appearance, window_ids) or None."""
    if not path or not os.path.exists(path):
        print(f"  [skip] {path} not found")
        return None
    d = np.load(path, allow_pickle=True)
    state_seq  = relabel_by_first_appearance(d["state_seq"].astype(int))
    window_ids = d["window_ids"].astype(int)
    return state_seq, window_ids


def draw_row(ax, labels, label_text):
    """Render one segmentation as a colour strip with boundary lines."""
    labels = relabel_by_first_appearance(np.asarray(labels))
    N = len(labels)
    k = int(labels.max()) + 1
    for i, lab in enumerate(labels):
        ax.bar(i, 1.0, width=1.0, color=TAB10[int(lab) % 10],
               edgecolor="white", linewidth=0.4, align="edge")
    # boundary lines between contiguous runs
    for _, i0, _ in contiguous_runs(labels)[1:]:
        ax.axvline(i0, color="#1a1a1a", lw=1.4, zorder=5)
    ax.set_xlim(0, N)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_xticks([])
    n_runs = len(contiguous_runs(labels))
    tag = "contiguous" if n_runs == k else f"{n_runs} runs"
    ax.set_ylabel(f"{label_text}\n({k} groups, {tag})",
                  rotation=0, ha="right", va="center", fontsize=9.5)
    for s in ax.spines.values():
        s.set_visible(False)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Reference ordering/dates come from the HMM-weights decode.
    hmm = load_decode(args.hmm_weights_decode)
    if hmm is None:
        raise SystemExit(f"HMM-weights decode is required: {args.hmm_weights_decode}")
    hmm_states, window_ids = hmm
    N = len(window_ids)
    k_equal = args.k if args.k is not None else int(hmm_states.max()) + 1
    dates = load_window_dates(args.manifest, window_ids)

    km = load_decode(args.kmeans_decode)
    pe = load_decode(args.pelt_decode)
    cd = load_decode(args.classdist_decode)

    # Rows in the order the user asked for; skip any missing decode gracefully.
    rows = [("HMM on weights", hmm_states)]
    if km is not None: rows.append(("k-means on weights", km[0]))
    if pe is not None: rows.append(("PELT on weights", pe[0]))
    if cd is not None: rows.append(("HMM on class dist.", cd[0]))
    rows.append(("Equal partition", equal_duration_labels(N, k_equal)))

    # sanity: all decode rows share the reference window ordering
    for tag, dec in [("k-means", km), ("PELT", pe), ("class-dist", cd)]:
        if dec is not None and not np.array_equal(dec[1], window_ids):
            print(f"  [warn] {tag} window_ids differ from HMM-weights ordering")

    n = len(rows)
    fig, axes = plt.subplots(n, 1, figsize=(13, 0.95 * n + 1.1),
                             gridspec_kw={"hspace": 0.55})
    if n == 1:
        axes = [axes]

    for ax, (label_text, labels) in zip(axes, rows):
        draw_row(ax, labels, label_text)

    # shared date axis on the bottom row
    bottom = axes[-1]
    tick_idx = np.linspace(0, N - 1, min(N, 9)).astype(int)
    bottom.set_xticks(tick_idx + 0.5)
    bottom.set_xticklabels([pd.Timestamp(dates[i]).strftime("%b %Y") for i in tick_idx],
                           rotation=30, ha="right", fontsize=8.5)
    bottom.set_xlabel("Window (chronological)", fontsize=10)

    fig.suptitle("Window segmentations across methods "
                 f"(N={N} windows, k={k_equal})", fontsize=12, y=0.99)

    out_path = os.path.join(args.output_dir, "segmentation_comparison.png")
    fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()