import pandas as pd
import numpy as np
import os
from datetime import date, timedelta

# ── 0. Config ──────────────────────────────────────────────────────────────
DATA_DIR   = "assets/raw/multimodal_only_samples"   # folder containing the three TSVs
OUTPUT_DIR = "data/splits"                          # where split CSVs will be saved
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── 1. Load & merge ────────────────────────────────────────────────────────
files = {
    "train":    "multimodal_train.tsv",
    "validate": "multimodal_validate.tsv",
    "test":     "multimodal_test_public.tsv",
}

dfs = []
for split, fname in files.items():
    path = os.path.join(DATA_DIR, fname)
    df   = pd.read_csv(path, sep="\t", low_memory=False)
    df["original_split"] = split          # keep track of provenance
    dfs.append(df)

data = pd.concat(dfs, ignore_index=True)
print(f"Total rows after merge: {len(data)}")

# ── 2. Filter to multimodal only (hasImage == True) ────────────────────────
data = data[data["hasImage"] == True].copy()
print(f"Rows with images: {len(data)}")

# ── 3. Parse timestamp & sort ──────────────────────────────────────────────
data["created_utc"] = pd.to_numeric(data["created_utc"], errors="coerce")
data = data.dropna(subset=["created_utc"])
data = data.sort_values("created_utc").reset_index(drop=True)

data["created_dt"] = pd.to_datetime(data["created_utc"], unit="s", utc=True)
print(f"Date range: {data['created_dt'].min()} → {data['created_dt'].max()}")
print(f"Final row count: {len(data)}\n")

# ── helper ─────────────────────────────────────────────────────────────────
def save(df, name):
    out = os.path.join(OUTPUT_DIR, f"{name}.tsv")
    df.to_csv(out, sep="\t", index=False)
    print(f"  Saved {name}.tsv  ({len(df):,} rows, "
          f"{df['created_dt'].min().date()} → {df['created_dt'].max().date()})")

# ══════════════════════════════════════════════════════════════════════════
# SPLIT A — Original (OG): use the authors' random split as-is
# ══════════════════════════════════════════════════════════════════════════
print("── Split A: Original (OG) ──")
save(data[data["original_split"] == "train"],    "OG_train")
save(data[data["original_split"] == "validate"], "OG_val")
save(data[data["original_split"] == "test"],     "OG_test")

# ══════════════════════════════════════════════════════════════════════════
# SPLIT B — Temporal: three consecutive chunks (82.56 / 8.72 / 8.72 %)
# ══════════════════════════════════════════════════════════════════════════
print("\n── Split B: Temporal ──")
n = len(data)
# Use round() rather than int() so that two successive truncations don't
# compound into a noticeable boundary error.
train_end = round(n * 0.8256)
val_end   = train_end + round(n * 0.0872)

temp_train = data.iloc[:train_end]
temp_val   = data.iloc[train_end:val_end]
temp_test  = data.iloc[val_end:]

save(temp_train, "Temporal_train")
save(temp_val,   "Temporal_val")
save(temp_test,  "Temporal_test")

# ══════════════════════════════════════════════════════════════════════════
# SPLIT C — Multiple Test Splits (5 test windows)
#
# Train  up to and including 2017-05-31          (paper: 06.2008–05.2017)
# Val    155 days  starting 2017-06-01           (paper: 06.2017–11.2017)
# Test1  153 days  following val                 (paper: 11.2017–04.2018)
# Test2  155 days  following test1               (paper: 05.2018–09.2018)
# Test3  155 days  following test2               (paper: 09.2018–02.2019)
# Test4  152 days  following test3               (paper: 02.2019–07.2019)
# Test5  128 days  following test4               (paper: 07.2019-11.2019)
# ══════════════════════════════════════════════════════════════════════════
print("\n── Split C: Multiple Test Splits ──")

MULTI_TRAIN_END = "2017-05-31"

# Day counts per window (inclusive of both endpoints)
DURATIONS = {
    "Multi_val":   155,
    "Multi_test1": 153,
    "Multi_test2": 155,
    "Multi_test3": 155,
    "Multi_test4": 152,
    "Multi_test5": 128, 
}

# Build start/end date pairs from the day counts
cursor = date(2017, 6, 1)
boundaries = {"Multi_train": ("2008-06-01", MULTI_TRAIN_END)}
for name, days in DURATIONS.items():
    end = cursor + timedelta(days=days - 1)          # inclusive end
    boundaries[name] = (cursor.isoformat(), end.isoformat())
    cursor = end + timedelta(days=1)                 # next window starts day after

for name, (start, end) in boundaries.items():
    mask = (data["created_dt"] >= pd.Timestamp(start, tz="UTC")) & \
           (data["created_dt"] < pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1))
    save(data[mask], name)

# ══════════════════════════════════════════════════════════════════════════
# Quick sanity check — class distributions
# ══════════════════════════════════════════════════════════════════════════
print("\n── 2-way label distribution (Temporal train) ──")
print(temp_train["2_way_label"].value_counts(normalize=True).round(4))

print("\n── 6-way label distribution (Temporal train) ──")
print(temp_train["6_way_label"].value_counts(normalize=True).round(4))

print("\nDone. All splits saved to:", OUTPUT_DIR)