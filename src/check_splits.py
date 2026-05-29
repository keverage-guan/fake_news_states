"""
check_multi_splits.py

Run after prepare_splits.py to report row counts and percentage of the
total multimodal dataset in each Multiple Test split.

Usage:
    python check_multi_splits.py
"""

import os
import pandas as pd

SPLITS_DIR = "data/splits"

MULTI_SPLITS = [
    "Multi_train",
    "Multi_val",
    "Multi_test1",
    "Multi_test2",
    "Multi_test3",
    "Multi_test4",
    "Multi_test5",
]

rows = []
for name in MULTI_SPLITS:
    path = os.path.join(SPLITS_DIR, f"{name}.tsv")
    if not os.path.exists(path):
        print(f"Missing: {path}  (did prepare_splits.py finish successfully?)")
        raise SystemExit(1)
    df = pd.read_csv(path, sep="\t", low_memory=False)
    df["created_dt"] = pd.to_datetime(
        pd.to_numeric(df["created_utc"], errors="coerce"), unit="s", utc=True
    )
    start = df["created_dt"].min().date()
    end   = df["created_dt"].max().date()
    rows.append({
        "split": name,
        "n":     len(df),
        "start": start,
        "end":   end,
        "days":  (end - start).days + 1,
    })

total = sum(r["n"] for r in rows)

print(f"\n{'Split':<15} {'Rows':>8} {'% of Total':>11} {'Days':>6}  {'Date Range'}")
print("-" * 68)
for r in rows:
    pct = 100 * r["n"] / total
    print(f"{r['split']:<15} {r['n']:>8,} {pct:>10.2f}% {r['days']:>6}  {r['start']} → {r['end']}")
print("-" * 68)
print(f"{'TOTAL':<15} {total:>8,} {'100.00%':>11}")

# Warn if any split is empty
empty = [r["split"] for r in rows if r["n"] == 0]
if empty:
    print(f"\nWARNING: empty splits: {empty}")