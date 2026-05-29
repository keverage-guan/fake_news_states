#!/usr/bin/env python3
"""
show_60day_windows.py

Prints a table of example counts per 60-day window and saves it to plots/60day_windows.csv.
Tries data/splits/*.tsv first, then assets/raw/multimodal_only_samples/.

Usage:
    python show_60day_windows.py
"""

import os, sys, glob
import pandas as pd

WINDOW_DAYS = 60
PLOTS_DIR   = "plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

def load_data():
    split_dir = "data/splits"
    if os.path.isdir(split_dir):
        tsv_files = glob.glob(os.path.join(split_dir, "*.tsv"))
        if tsv_files:
            dfs = [pd.read_csv(f, sep="\t", low_memory=False,
                               usecols=lambda c: c in {"id", "created_utc"})
                   for f in tsv_files]
            data = pd.concat(dfs, ignore_index=True)
            if "id" in data.columns:
                data = data.drop_duplicates(subset="id")
            return data

    raw_dir = "assets/raw/multimodal_only_samples"
    raw_files = [
        os.path.join(raw_dir, "multimodal_train.tsv"),
        os.path.join(raw_dir, "multimodal_validate.tsv"),
        os.path.join(raw_dir, "multimodal_test_public.tsv"),
    ]
    if all(os.path.exists(p) for p in raw_files):
        dfs = [pd.read_csv(p, sep="\t", low_memory=False,
                           usecols=lambda c: c in {"id", "created_utc", "hasImage"})
               for p in raw_files]
        data = pd.concat(dfs, ignore_index=True)
        if "hasImage" in data.columns:
            data = data[data["hasImage"] == True]
        return data

    sys.exit("Could not find data. Run from your project root.")

# ── Load & parse timestamps ────────────────────────────────────────────────────
data = load_data()
data["created_utc"] = pd.to_numeric(data["created_utc"], errors="coerce")
data = data.dropna(subset=["created_utc"])
data["created_dt"] = pd.to_datetime(data["created_utc"], unit="s", utc=True)
data = data.sort_values("created_dt").reset_index(drop=True)

print(f"Loaded {len(data):,} rows  |  "
      f"{data['created_dt'].min().date()} → {data['created_dt'].max().date()}\n")

# ── Build 60-day windows ───────────────────────────────────────────────────────
ts = data["created_dt"].dt.tz_localize(None)
t0 = ts.min().normalize()
t1 = ts.max().normalize() + pd.Timedelta(days=1)

total_days = (t1 - t0).days
n_windows  = total_days // WINDOW_DAYS + (1 if total_days % WINDOW_DAYS else 0)

rows = []
for i in range(n_windows):
    w_start = t0 + pd.Timedelta(days=i * WINDOW_DAYS)
    w_end   = t0 + pd.Timedelta(days=(i + 1) * WINDOW_DAYS)
    if w_end > t1:
        w_end = t1
    count = int(((ts >= w_start) & (ts < w_end)).sum())
    rows.append({
        "Window": i + 1,
        "Start":  w_start.date(),
        "End":    (w_end - pd.Timedelta(days=1)).date(),
        "Days":   (w_end - w_start).days,
        "Count":  count,
    })

table = pd.DataFrame(rows)

# ── Print table ────────────────────────────────────────────────────────────────
col_w = {"Window": 6, "Start": 12, "End": 12, "Days": 5, "Count": 8}
header = (f"{'Win':>6}  {'Start':<12}  {'End':<12}  {'Days':>5}  {'Count':>8}")
sep    = "-" * len(header)

print(header)
print(sep)
for _, r in table.iterrows():
    print(f"{int(r.Window):>6}  {str(r.Start):<12}  {str(r.End):<12}  "
          f"{int(r.Days):>5}  {int(r.Count):>8,}")

print(sep)
print(f"{'TOTAL':>6}  {'':12}  {'':12}  {'':>5}  {table['Count'].sum():>8,}")
print(f"\nMean per window: {table['Count'].mean():,.0f}   "
      f"Std: {table['Count'].std():,.0f}   "
      f"Min: {table['Count'].min():,}   "
      f"Max: {table['Count'].max():,}")

# ── Save CSV ───────────────────────────────────────────────────────────────────
out_csv = os.path.join(PLOTS_DIR, "60day_windows.csv")
table.to_csv(out_csv, index=False)
print(f"\nSaved -> {out_csv}")