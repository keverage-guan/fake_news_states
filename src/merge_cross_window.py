"""
src/merge_cross_window.py

Merge the per-row .npz files produced by the SLURM array job into the
final cross_window_f1.npz, cross_window_f1.csv, and heatmap.

This script is HMM-agnostic: it assembles the raw F1 matrix only.
HMM state labels are joined downstream in src/within_across_states.py.

Run after all array tasks complete:
    python src/merge_cross_window.py

Then, when you are ready to analyse with HMM states (which can be
re-run at any time with a different k or covariance type):
    python src/within_across_states.py \\
        --f1_npz     data/hmm_perf/6way/cross_window_f1.npz \\
        --decode_npz data/hmm_hmm/6way/final_decode_k6.npz \\
        --output_dir data/hmm_within_across/6way/k6

Usage
-----
    python src/merge_cross_window.py \
        --rows_dir   data/hmm_perf/6way/rows \
        --output_dir data/hmm_perf/6way \
        --n_way      6
"""

import os
import sys
import argparse
import glob
import time

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")

# Reuse plot + save helpers from the eval script
sys.path.insert(0, os.path.dirname(__file__))
from cross_window_eval import plot_heatmap, _save_and_plot


def elapsed(t0: float) -> str:
    s = time.time() - t0
    return f"{s:.1f}s" if s < 60 else f"{s / 60:.1f}min"


def main(args: argparse.Namespace) -> None:
    t0 = time.time()
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("  src/merge_cross_window.py")
    print("=" * 60)

    # ── Collect row files ──────────────────────────────────────────────────
    row_files = sorted(glob.glob(os.path.join(args.rows_dir, "row_*.npz")))
    print(f"  Found {len(row_files)} row files in {args.rows_dir}")

    if not row_files:
        print("ERROR: no row files found. Did the array job finish?")
        sys.exit(1)

    # Peek at one file to learn the shape
    probe     = np.load(row_files[0])
    valid_ids = probe["valid_ids"].tolist()
    n_valid   = len(valid_ids)
    n_way     = args.n_way

    # Infer max seed count from whichever file has the cube
    n_seeds_max = 10
    for fpath in row_files:
        d = np.load(fpath)
        if "row_f1_per_seed" in d:
            n_seeds_max = d["row_f1_per_seed"].shape[0]
            break
    print(f"  n_valid={n_valid}, n_way={n_way}, n_seeds_max={n_seeds_max}")

    # ── Assemble matrices ──────────────────────────────────────────────────
    f1_matrix        = np.full((n_valid, n_valid), np.nan)
    f1_per_class     = np.full((n_valid, n_valid, n_way), np.nan)
    f1_per_seed_cube = np.full((n_valid, n_seeds_max, n_valid), np.nan)
    rows_filled: set[int] = set()

    for fpath in row_files:
        d   = np.load(fpath)
        idx = int(d["row_idx"])

        if idx >= n_valid:
            print(f"  [warn] row_idx={idx} >= n_valid={n_valid} — skipping")
            continue
        if list(d["valid_ids"]) != valid_ids:
            print(f"  [warn] {fpath}: valid_ids mismatch — skipping")
            continue

        f1_matrix[idx]    = d["row_f1"]
        f1_per_class[idx] = d["row_per_class"]

        if "row_f1_per_seed" in d:
            fps = d["row_f1_per_seed"]          # (n_seeds, n_valid)
            s   = min(fps.shape[0], n_seeds_max)
            f1_per_seed_cube[idx, :s, :] = fps[:s]

        rows_filled.add(idx)

    missing = sorted(set(range(n_valid)) - rows_filled)
    if missing:
        print(f"\n  [warn] {len(missing)} rows missing: {missing}")
        print(f"  These will appear as NaN in the matrix.")
        print(f"  To rerun missing tasks only:")
        print(f"    sbatch --array={','.join(str(m) for m in missing)} "
              f"cross_window_eval.slurm")
    else:
        print(f"\n  All {n_valid} rows present — matrix is complete.")

    # ── Save + plot ────────────────────────────────────────────────────────
    # Build a minimal namespace for _save_and_plot
    class _Args:
        pass
    save_args          = _Args()
    save_args.output_dir = args.output_dir
    save_args.n_way      = n_way

    _save_and_plot(f1_matrix, f1_per_class, f1_per_seed_cube,
                   valid_ids, save_args, t0)

    print(
        "\n  Next steps:\n"
        "    1. (if not done) Run the HMM pipeline:\n"
        "         python src/extract_weights_pca.py\n"
        "         python src/fit_hmm_decode.py --k 6\n"
        "    2. Join F1 matrix with HMM states:\n"
        "         python src/within_across_states.py \\\n"
        f"             --f1_npz     {os.path.join(args.output_dir, 'cross_window_f1.npz')} \\\n"
        "             --decode_npz data/hmm_hmm/6way/final_decode_k6.npz \\\n"
        "             --output_dir data/hmm_within_across/6way/k6"
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Merge per-row .npz files into the full cross-window F1 matrix. "
            "HMM-agnostic: no decode_npz required. "
            "State labels are joined later in within_across_states.py."
        )
    )
    p.add_argument("--rows_dir",
                   default="data/hmm_perf/6way/rows",
                   help="Directory containing row_NNN.npz files from array job")
    p.add_argument("--output_dir",
                   default="data/hmm_perf/6way",
                   help="Where to write cross_window_f1.npz/.csv and heatmap")
    p.add_argument("--n_way", type=int, default=6, choices=[2, 6])
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())