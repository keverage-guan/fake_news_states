#!/usr/bin/env python3
"""
Extract the fields needed for sentiment analysis from the Yelp dataset.

Input:
    assets/raw/yelp_academic_dataset_review.json

Output:
    data/yelp_sentiment.jsonl

Each output line contains:
{
    "id": "...",
    "timestamp": "...",
    "sentiment": <star rating>,
    "text": "..."
}
"""

from pathlib import Path
import json

RAW_DIR = Path("assets/raw")
OUT_DIR = Path("data")

INPUT_FILE = RAW_DIR / "yelp_academic_dataset_review.json"
OUTPUT_FILE = OUT_DIR / "yelp_sentiment.jsonl"

OUT_DIR.mkdir(parents=True, exist_ok=True)

count = 0

with INPUT_FILE.open("r", encoding="utf-8") as fin, \
     OUTPUT_FILE.open("w", encoding="utf-8") as fout:

    for line in fin:
        review = json.loads(line)

        processed = {
            "id": review["review_id"],
            "timestamp": review["date"],
            "sentiment": review["stars"],
            "text": review["text"].strip(),
        }

        fout.write(json.dumps(processed, ensure_ascii=False) + "\n")
        count += 1

print(f"Processed {count:,} reviews.")
print(f"Saved to {OUTPUT_FILE}")