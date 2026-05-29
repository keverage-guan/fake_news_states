"""
src_prepare_window_splits.py

Generates per-window data splits for the temporal HMM experiment.

Procedure
---------
1. Load and merge the three Fakeddit TSVs (same as src_prepare_splits.py).
2. Filter to multimodal-only (hasImage == True), parse & sort by timestamp.
3. Divide the full date range into fixed 60-day windows (last window absorbs
   any remainder).
4. Find the largest contiguous run of windows that each have >= MIN_SAMPLES
   raw samples (before any subsampling).
5. Compute N = min raw count across all windows in that contiguous run.
6. Subsample each qualifying window to exactly N samples, stratified by
   6-way label (class proportions preserved, not equalized).
7. Save each window as a TSV: HMM_window_000.tsv, HMM_window_001.tsv, ...
8. Save a manifest CSV summarising every window.

Design notes
------------
- No train/val split is done here; that's handled downstream.
- Subsampling uses 6-way labels as the stratification key (the harder
  constraint). The 2-way label is preserved in the output rows.
- Subsampling is stratified: each class contributes floor(N * class_prop)
  samples, with any remainder distributed to the largest classes so the
  total is exactly N.
- Windows are zero-indexed and named by their position in the *qualifying*
  contiguous run, not in the global window sequence.
- The random seed is fixed for reproducibility.

Usage
-----
    python src_prepare_window_splits.py

    # Override defaults
    python src_prepare_window_splits.py \\
        --data_dir  assets/raw/multimodal_only_samples \\
        --output_dir data/splits/hmm_windows \\
        --window_days 60 \\
        --min_samples 9000 \\
        --seed 42
"""

import os
import argparse
import numpy as np
import pandas as pd
from datetime import timedelta


# ── Defaults ──────────────────────────────────────────────────────────────────

DATA_DIR    = "assets/raw/multimodal_only_samples"
OUTPUT_DIR  = "data/splits/hmm_windows"
WINDOW_DAYS = 60
MIN_SAMPLES = 9_000
SEED        = 42
LABEL_COL   = "6_way_label"   # stratification key


# ── Helpers ───────────────────────────────────────────────────────────────────

def stratified_subsample(
    df: pd.DataFrame,
    n: int,
    label_col: str,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Draw exactly `n` rows from `df`, preserving class proportions of
    `label_col`.  Uses floor allocation with largest-remainder tie-breaking
    so the total is always exactly n.
    """
    counts = df[label_col].value_counts()
    proportions = counts / counts.sum()

    # Floor allocation
    alloc = (proportions * n).apply(np.floor).astype(int)
    remainder = n - alloc.sum()

    # Distribute remainder to classes with largest fractional parts
    frac = (proportions * n) - alloc
    top_classes = frac.nlargest(remainder).index
    alloc[top_classes] += 1

    assert alloc.sum() == n, f"Allocation sum {alloc.sum()} != {n}"

    parts = []
    for cls, k in alloc.items():
        pool = df[df[label_col] == cls]
        sampled = pool.sample(n=int(k), replace=False, random_state=rng.integers(2**31))
        parts.append(sampled)

    return pd.concat(parts).sample(frac=1, random_state=rng.integers(2**31)).reset_index(drop=True)


def largest_contiguous_run(mask: list[bool]) -> tuple[int, int]:
    """
    Returns (start_idx, end_idx) inclusive of the longest contiguous True run
    in `mask`.  Ties broken by earliest start.
    """
    best_start, best_len = 0, 0
    cur_start, cur_len   = 0, 0

    for i, val in enumerate(mask):
        if val:
            if cur_len == 0:
                cur_start = i
            cur_len += 1
            if cur_len > best_len:
                best_len  = cur_len
                best_start = cur_start
        else:
            cur_len = 0

    if best_len == 0:
        raise ValueError("No window meets the minimum sample threshold.")

    return best_start, best_start + best_len - 1


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data_dir",    default=DATA_DIR)
    parser.add_argument("--output_dir",  default=OUTPUT_DIR)
    parser.add_argument("--window_days", type=int,   default=WINDOW_DAYS)
    parser.add_argument("--min_samples", type=int,   default=MIN_SAMPLES)
    parser.add_argument("--seed",        type=int,   default=SEED)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # ── 1. Load & merge ───────────────────────────────────────────────────
    files = {
        "train":    "multimodal_train.tsv",
        "validate": "multimodal_validate.tsv",
        "test":     "multimodal_test_public.tsv",
    }
    dfs = []
    for split, fname in files.items():
        path = os.path.join(args.data_dir, fname)
        df = pd.read_csv(path, sep="\t", low_memory=False)
        df["original_split"] = split
        dfs.append(df)

    data = pd.concat(dfs, ignore_index=True)
    print(f"Total rows after merge: {len(data):,}")

    # ── 2. Filter, parse, sort ────────────────────────────────────────────
    data = data[data["hasImage"] == True].copy()
    print(f"Rows with images: {len(data):,}")

    data["created_utc"] = pd.to_numeric(data["created_utc"], errors="coerce")
    data = data.dropna(subset=["created_utc"])
    data = data.sort_values("created_utc").reset_index(drop=True)
    data["created_dt"] = pd.to_datetime(data["created_utc"], unit="s", utc=True)

    print(f"Date range: {data['created_dt'].min().date()} → {data['created_dt'].max().date()}")
    print(f"Final row count: {len(data):,}\n")

    # ── 3. Build fixed-width 60-day windows over the full date range ──────
    window_td = timedelta(days=args.window_days)
    origin    = data["created_dt"].min().normalize()  # midnight of first day

    # Assign each row a window index
    data["window_idx"] = (
        (data["created_dt"] - origin) // pd.Timedelta(days=args.window_days)
    ).astype(int)

    all_window_idxs = sorted(data["window_idx"].unique())
    print(f"Total windows in date range: {len(all_window_idxs)}")
    print(f"Window width: {args.window_days} days\n")

    # ── 4. Count samples per window, find qualifying contiguous run ───────
    window_counts = data.groupby("window_idx").size()

    # Build ordered mask: True if window meets threshold
    # (include every integer index from min to max, even if count=0)
    idx_min, idx_max = all_window_idxs[0], all_window_idxs[-1]
    all_idxs = list(range(idx_min, idx_max + 1))
    counts_full = [int(window_counts.get(i, 0)) for i in all_idxs]
    meets_threshold = [c >= args.min_samples for c in counts_full]

    run_start_local, run_end_local = largest_contiguous_run(meets_threshold)

    # Map local indices back to global window indices
    qualifying_global_idxs = all_idxs[run_start_local : run_end_local + 1]
    qualifying_counts = counts_full[run_start_local : run_end_local + 1]

    n_windows = len(qualifying_global_idxs)
    n_subsample = min(qualifying_counts)

    # Compute actual date boundaries for the qualifying run
    run_start_dt = origin + pd.Timedelta(days=qualifying_global_idxs[0]  * args.window_days)
    run_end_dt   = origin + pd.Timedelta(days=(qualifying_global_idxs[-1] + 1) * args.window_days)

    print(f"Qualifying contiguous run: {n_windows} windows")
    print(f"  Global window indices : {qualifying_global_idxs[0]} → {qualifying_global_idxs[-1]}")
    print(f"  Date range            : {run_start_dt.date()} → {(run_end_dt - pd.Timedelta(days=1)).date()}")
    print(f"  Raw counts range      : min={min(qualifying_counts):,}, max={max(qualifying_counts):,}")
    print(f"  Subsampling N         : {n_subsample:,}  (= min raw count)\n")

    # ── 5 & 6. Subsample each qualifying window and save ──────────────────
    manifest_rows = []

    for local_i, global_idx in enumerate(qualifying_global_idxs):
        window_data = data[data["window_idx"] == global_idx].copy()

        w_start = origin + pd.Timedelta(days=global_idx       * args.window_days)
        w_end   = origin + pd.Timedelta(days=(global_idx + 1) * args.window_days) - pd.Timedelta(seconds=1)

        raw_n = len(window_data)
        sampled = stratified_subsample(window_data, n_subsample, LABEL_COL, rng)

        fname = f"HMM_window_{local_i:03d}.tsv"
        out_path = os.path.join(args.output_dir, fname)
        sampled.to_csv(out_path, sep="\t", index=False)

        # Class distribution after subsampling
        class_dist = sampled[LABEL_COL].value_counts().sort_index().to_dict()

        manifest_rows.append({
            "window_local_idx":  local_i,
            "window_global_idx": global_idx,
            "start_date":        w_start.date().isoformat(),
            "end_date":          w_end.date().isoformat(),
            "raw_n":             raw_n,
            "sampled_n":         len(sampled),
            "filename":          fname,
            **{f"cls_{k}": v for k, v in class_dist.items()},
        })

        print(f"  [{local_i:03d}] {w_start.date()} → {w_end.date()}"
              f"  raw={raw_n:,}  sampled={len(sampled):,}"
              f"  saved → {fname}")

    # ── 7. Save manifest ──────────────────────────────────────────────────
    manifest = pd.DataFrame(manifest_rows)
    manifest_path = os.path.join(args.output_dir, "HMM_windows_manifest.csv")
    manifest.to_csv(manifest_path, index=False)

    print(f"\nManifest saved → {manifest_path}")
    print(f"\nDone. {n_windows} windows × {n_subsample:,} samples each"
          f" saved to: {args.output_dir}")

    # ── 8. Sanity check: class distributions ─────────────────────────────
    print(f"\n── {LABEL_COL} distribution check (first vs last window) ──")
    for label in [manifest_rows[0], manifest_rows[-1]]:
        cls_cols = {k: v for k, v in label.items() if k.startswith("cls_")}
        total = sum(cls_cols.values())
        dist  = {k: f"{v/total:.3f}" for k, v in cls_cols.items()}
        print(f"  Window {label['window_local_idx']:03d} ({label['start_date']}): {dist}")


if __name__ == "__main__":
    main()