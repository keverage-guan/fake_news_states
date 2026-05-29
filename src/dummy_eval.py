"""
src/dummy_eval.py

Evaluates a dummy (hard majority) classifier for Experiment 2.

The dummy always predicts the most frequent class in the training set,
without using any content features. Any change in its performance across
test splits therefore reflects only class distribution shift — not content
shift. This isolates the "class distribution shift" condition described in
Stepanova & Ross (2023), Section 5.

How the dummy works
-------------------
The dummy predicts the majority class (argmax of training proportions)
for every sample. Metrics follow directly:

    micro F1 (= accuracy) = proportion of the majority class in the test set
                            (the only samples it gets right)

    Per-class F1:
        majority class c*:  F1 = 2 * p_c* / (p_c* + 1)
                            where p_c* = test proportion of c*
                            (precision = p_c*, recall = 1.0)
        all other classes:  F1 = 0  (recall = 0, precision undefined → 0)

    macro F1 = mean of per-class F1s over all classes in train ∪ test

Sanity check: a simulated version (predicting majority class for every
sample) is also computed and should match the analytical result exactly.

Usage:
    # Default: evaluate all Multi_test*.tsv (unbalanced) against Multi_train
    python src/dummy_eval.py

    # Explicit paths
    python src/dummy_eval.py \\
        --train data/splits/Multi_train.tsv \\
        --tests data/splits/Multi_test1.tsv data/splits/Multi_test2.tsv \\
        --n_way 2

    # Run for 6-way labels
    python src/dummy_eval.py --n_way 6

    # Save results to JSON
    python src/dummy_eval.py --n_way 2 --output runs/exp2_dummy_2way.json
    python src/dummy_eval.py --n_way 6 --output runs/exp2_dummy_6way.json

Outputs:
    Prints a results table to stdout.
    Optionally writes a JSON file with per-split metrics.
"""

import os
import glob
import json
import argparse
import numpy as np
import pandas as pd


# ── Config ────────────────────────────────────────────────────────────────────

SPLITS_DIR = "data/splits"

LABEL_COLS = {
    2: "2_way_label",
    6: "6_way_label",
}


# ── Metrics ───────────────────────────────────────────────────────────────────

def majority_class_metrics(
    train_props: pd.Series,
    test_props: pd.Series,
) -> tuple[float, float]:
    """
    Compute micro F1 and macro F1 for a hard majority classifier.

    The classifier always predicts majority_class = train_props.idxmax().

    micro F1 = accuracy = test proportion of the majority class
               (correct only when true label == majority class)

    Per-class F1:
        majority class c*:
            precision = p_c*  (of all predictions = c*, fraction truly c*)
            recall    = 1.0   (all true c* samples are predicted c*)
            F1        = 2 * p_c* / (p_c* + 1)
        all other classes:
            recall    = 0.0   (never predicted → no true positives)
            F1        = 0.0

    macro F1 = mean F1 over all classes in train_props ∪ test_props.

    Parameters
    ----------
    train_props : pd.Series
        Class proportions in the training set (index = class labels).
    test_props : pd.Series
        Class proportions in the test set (index = class labels).

    Returns
    -------
    (micro_f1, macro_f1) : (float, float)
    """
    majority = train_props.idxmax()

    # micro F1 = accuracy = fraction of test samples that are majority class
    micro_f1 = float(test_props.get(majority, 0.0))

    # macro F1: average per-class F1
    all_classes = sorted(train_props.index)
    f1s = []
    for c in all_classes:
        if c == majority:
            p = float(test_props.get(c, 0.0))   # precision of majority class
            # recall = 1.0, so F1 = 2*precision*1 / (precision + 1)
            f1s.append(2 * p / (p + 1) if (p + 1) > 0 else 0.0)
        else:
            f1s.append(0.0)
    macro_f1 = float(np.mean(f1s))

    return micro_f1, macro_f1


def simulated_micro_f1(
    majority_class,
    test_labels: pd.Series,
) -> float:
    """
    Simulate the hard majority classifier and compute accuracy.
    Should match majority_class_metrics micro F1 exactly (not just
    approximately, since there is no randomness).
    """
    preds = np.full(len(test_labels), fill_value=majority_class)
    return float(np.mean(preds == test_labels.values))


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_proportions(df: pd.DataFrame, label_col: str) -> pd.Series:
    counts = df[label_col].value_counts().sort_index()
    return counts / counts.sum()


def load_tsv(path: str) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", low_memory=False)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Hard majority dummy classifier evaluation for Experiment 2."
    )
    parser.add_argument(
        "--train", default=os.path.join(SPLITS_DIR, "Multi_train.tsv"),
        help="Training TSV whose majority class the dummy always predicts."
    )
    parser.add_argument(
        "--tests", nargs="+", default=None,
        help="Test TSVs to evaluate. Default: all Multi_test*.tsv "
             "(excluding _balanced) in --splits_dir."
    )
    parser.add_argument(
        "--splits_dir", default=SPLITS_DIR,
    )
    parser.add_argument(
        "--n_way", type=int, choices=[2, 6], default=2,
    )
    parser.add_argument(
        "--output", default=None,
        help="Optional path to write results JSON."
    )
    args = parser.parse_args()

    label_col = LABEL_COLS[args.n_way]

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

    print(f"Dummy classifier (hard majority) — {args.n_way}-way  "
          f"(label: {label_col})")
    print(f"Training split : {args.train}")
    print(f"Test splits    : {len(test_paths)} file(s)\n")

    # ── Training distribution ────────────────────────────────────────────
    train_df    = load_tsv(args.train)
    train_props = load_proportions(train_df, label_col)
    majority    = train_props.idxmax()

    print("Training class distribution:")
    for cls, p in train_props.items():
        marker = " ← majority (always predicted)" if cls == majority else ""
        print(f"  class {cls}: {p:.4f} ({p*100:.1f}%){marker}")
    print()

    # ── Evaluate on each test split ──────────────────────────────────────
    results = {}
    rows    = []

    for test_path in test_paths:
        stem    = os.path.splitext(os.path.basename(test_path))[0]
        test_df = load_tsv(test_path)

        test_counts = test_df[label_col].value_counts().sort_index()
        test_props  = test_counts / test_counts.sum()

        micro_f1, macro_f1 = majority_class_metrics(train_props, test_props)
        micro_f1_sim        = simulated_micro_f1(majority, test_df[label_col])

        dist_str = "  |  ".join(
            f"cls {c}: {test_props.get(c, 0):.3f}" for c in train_props.index
        )

        print(f"── {stem} ──")
        print(f"  n={len(test_df):,}  distribution: {dist_str}")
        print(f"  micro F1 (analytical) : {micro_f1:.4f}")
        print(f"  micro F1 (simulated)  : {micro_f1_sim:.4f}  "
              f"[should match analytical exactly]")
        print(f"  macro F1 (analytical) : {macro_f1:.4f}")
        print()

        results[stem] = {
            "n_samples":           len(test_df),
            "majority_class_tsv": int(majority), 
            "majority_class_model": int({0: 1, 1: 0}.get(majority, majority)),
            "micro_f1":            round(micro_f1, 5),
            "macro_f1":            round(macro_f1, 5),
            "test_class_props":    {
                str(c): round(float(v), 5) for c, v in test_props.items()
            },
        }
        rows.append({
            "split":    stem,
            "n":        len(test_df),
            "micro_f1": round(micro_f1, 4),
            "macro_f1": round(macro_f1, 4),
        })

    # ── Summary table ────────────────────────────────────────────────────
    print("── Summary ──")
    summary = pd.DataFrame(rows)
    print(summary.to_string(index=False))

    # ── Optional JSON output ─────────────────────────────────────────────
    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        payload = {
            "n_way":             args.n_way,
            "label_col":         label_col,
            "train_split":       args.train,
            "majority_class_tsv":   int(majority),
            "majority_class_model": int({0: 1, 1: 0}.get(majority, majority)),
            "train_class_props": {
                str(c): round(float(v), 5) for c, v in train_props.items()
            },
            "results":           results,
        }
        with open(args.output, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\nResults saved → {args.output}")


if __name__ == "__main__":
    main()