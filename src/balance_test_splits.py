"""
src/balance_test_splits.py

Generates class-balanced versions of the Multi test splits for Experiment 2.

For each test split, we subsample so that the class distribution matches
the training set's distribution (by proportion). This controls for class
distribution shift, leaving only content shift as the variable — i.e. the
"Exp. 2 Balanced" condition in Stepanova & Ross (2023).

The limiting factor is whichever class is most "underrepresented" relative
to what the training distribution demands. All other classes are downsampled
to match.

Usage:
    # Default: balance all Multi_test*.tsv against Multi_train.tsv
    python src/balance_test_splits.py

    # Override paths explicitly
    python src/balance_test_splits.py \
        --train  data/splits/Multi_train.tsv \
        --tests  data/splits/Multi_test1.tsv data/splits/Multi_test2.tsv \
        --output_dir data/splits \
        --n_way 2

    # Run for 6-way labels
    python src/balance_test_splits.py --n_way 6

    # Fix the random seed for reproducibility
    python src/balance_test_splits.py --seed 42

Outputs:
    One new TSV per input test split, written to --output_dir with the
    suffix "_balanced" appended to the stem, e.g.:
        Multi_test1_balanced.tsv
        Multi_test2_balanced.tsv
        ...

    A summary table is printed to stdout.
"""

import os
import glob
import argparse
import numpy as np
import pandas as pd


# ── Config ────────────────────────────────────────────────────────────────────

SPLITS_DIR = "data/splits"

LABEL_COLS = {
    2: "2_way_label",
    6: "6_way_label",
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_class_counts(df: pd.DataFrame, label_col: str) -> pd.Series:
    """Returns a Series of {label_value: count}, sorted by label."""
    return df[label_col].value_counts().sort_index()


def get_class_proportions(counts: pd.Series) -> pd.Series:
    """Converts counts to proportions (sum to 1.0)."""
    return counts / counts.sum()


def subsample_to_distribution(
    df: pd.DataFrame,
    label_col: str,
    target_proportions: pd.Series,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Subsample `df` so its class distribution matches `target_proportions`.

    Strategy
    --------
    For each class c, the maximum number of samples we could keep while
    respecting the target proportion is:

        budget_c = floor(n_c / p_c)

    where n_c is the count of class c in the test split and p_c is its
    target proportion. The global budget is min(budget_c) over all classes.
    We then keep floor(budget * p_c) samples from each class, drawn without
    replacement.

    This is the tightest valid subsample: every class is represented at
    exactly the target proportion, and we retain as many rows as possible.

    Parameters
    ----------
    df               : DataFrame with a label column
    label_col        : name of the label column
    target_proportions : Series indexed by class label, summing to ~1.0
    rng              : numpy random Generator for reproducibility

    Returns
    -------
    Balanced (subsampled) DataFrame, index reset.
    """
    present_classes = df[label_col].unique()
    missing = set(target_proportions.index) - set(present_classes)
    if missing:
        print(f"  Warning: classes {missing} absent from this test split. "
              f"Proportions will be renormalised over present classes.")
        target_proportions = target_proportions[
            target_proportions.index.isin(present_classes)
        ]
        target_proportions = target_proportions / target_proportions.sum()

    counts = get_class_counts(df, label_col)
    # Only consider classes that appear in the target
    counts = counts[counts.index.isin(target_proportions.index)]

    # Compute the global budget (how many total rows we can afford)
    budgets = {}
    for cls, n_c in counts.items():
        p_c = target_proportions[cls]
        if p_c > 0:
            budgets[cls] = int(np.floor(n_c / p_c))
    global_budget = min(budgets.values())

    # Compute per-class target counts
    target_counts = {
        cls: int(np.floor(global_budget * target_proportions[cls]))
        for cls in target_proportions.index
        if cls in counts.index
    }

    # Sample from each class without replacement
    sampled_frames = []
    for cls, n_keep in target_counts.items():
        class_df = df[df[label_col] == cls]
        n_keep   = min(n_keep, len(class_df))   # safety guard
        sampled_frames.append(
            class_df.sample(n=n_keep, replace=False, random_state=int(rng.integers(2**31)))
        )

    balanced = pd.concat(sampled_frames, ignore_index=True)
    # Shuffle so classes are interleaved (not all one class then another)
    balanced = balanced.sample(frac=1.0, random_state=int(rng.integers(2**31))).reset_index(drop=True)
    return balanced


def print_distribution(label: str, counts: pd.Series) -> None:
    total = counts.sum()
    parts = ", ".join(
        f"cls {c}: {n:,} ({100*n/total:.1f}%)" for c, n in counts.items()
    )
    print(f"    {label}: total={total:,}  [{parts}]")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate class-balanced test splits for Experiment 2."
    )
    parser.add_argument(
        "--train", default=os.path.join(SPLITS_DIR, "Multi_train.tsv"),
        help="Training TSV whose class distribution is the target."
    )
    parser.add_argument(
        "--tests", nargs="+", default=None,
        help="Test TSVs to balance. Default: all Multi_test*.tsv in --splits_dir."
    )
    parser.add_argument(
        "--splits_dir", default=SPLITS_DIR,
        help="Directory to scan for Multi_test*.tsv when --tests is not given."
    )
    parser.add_argument(
        "--output_dir", default=None,
        help="Where to write balanced TSVs. Default: same directory as each input."
    )
    parser.add_argument(
        "--n_way", type=int, choices=[2, 6], default=2,
        help="Which label column to use for balancing (2 or 6)."
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="Random seed for reproducibility."
    )
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    # ── Resolve test paths ───────────────────────────────────────────────
    if args.tests:
        test_paths = args.tests
    else:
        pattern    = os.path.join(args.splits_dir, "Multi_test*.tsv")
        test_paths = sorted(
            p for p in glob.glob(pattern)
            if "_balanced" not in os.path.basename(p)
        )

    if not test_paths:
        print(f"No test TSVs found. Did you run prepare_splits.py?")
        return

    label_col = LABEL_COLS[args.n_way]
    print(f"Label column  : {label_col}  ({args.n_way}-way)")
    print(f"Training split: {args.train}")
    print(f"Random seed   : {args.seed}")
    print(f"Test splits   : {len(test_paths)} file(s)\n")

    # ── Load training distribution ───────────────────────────────────────
    print("Loading training split...")
    train_df = pd.read_csv(args.train, sep="\t", low_memory=False)
    train_counts = get_class_counts(train_df, label_col)
    train_props  = get_class_proportions(train_counts)

    print(f"  {len(train_df):,} rows")
    print_distribution("train", train_counts)
    print()

    # ── Process each test split ──────────────────────────────────────────
    summary_rows = []

    for test_path in test_paths:
        stem     = os.path.splitext(os.path.basename(test_path))[0]
        out_dir = args.output_dir or os.path.dirname(test_path) or "."
        out_path = os.path.join(out_dir, f"{stem}_balanced_{args.n_way}way.tsv")

        print(f"── {stem} ──")
        test_df = pd.read_csv(test_path, sep="\t", low_memory=False)
        print(f"  Original : {len(test_df):,} rows")
        print_distribution("before", get_class_counts(test_df, label_col))

        balanced = subsample_to_distribution(test_df, label_col, train_props, rng)
        bal_counts = get_class_counts(balanced, label_col)
        print(f"  Balanced : {len(balanced):,} rows")
        print_distribution("after ", bal_counts)

        os.makedirs(out_dir, exist_ok=True)
        balanced.to_csv(out_path, sep="\t", index=False)
        print(f"  Saved → {out_path}\n")

        summary_rows.append({
            "split":           stem,
            "original_rows":   len(test_df),
            "balanced_rows":   len(balanced),
            "retention_pct":   round(100 * len(balanced) / len(test_df), 1),
        })

    # ── Summary table ────────────────────────────────────────────────────
    print("── Summary ──")
    summary = pd.DataFrame(summary_rows)
    print(summary.to_string(index=False))
    print(
        "\nBalanced TSVs are ready. Pass them to train.py via --test to run "
        "the Exp. 2 Balanced condition."
    )


if __name__ == "__main__":
    main()