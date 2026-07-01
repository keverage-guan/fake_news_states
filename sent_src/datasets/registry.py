"""
datasets/registry.py

A DatasetSpec captures everything that varies between datasets, so the dataset
class, model, trainer, and weight extractor stay dataset-agnostic. Adding a new
dataset means adding a spec here, not editing the pipeline.

What varies between Fakeddit and Yelp:
  - modality: Fakeddit is text+image (2816-dim); Yelp is text-only (768-dim)
  - file format: Fakeddit rows are TSV; Yelp rows are JSONL
  - label scheme: Fakeddit has 2-way/6-way string labels; Yelp has 1-5 star ratings
  - column names: clean_title vs text; id/6_way_label vs id/sentiment

Everything downstream keys off `input_dim` (derived, never hardcoded) and the
label scheme in the spec.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Tuple

import pandas as pd


# ── Embedding dimensions per encoder ──────────────────────────────────────────
TEXT_DIM  = 768    # DistilRoBERTa / sentence-transformer text encoder
IMAGE_DIM = 2048   # ResNet-50 penultimate layer

EMBEDDING_DIMS = {"text": TEXT_DIM, "image": IMAGE_DIM}


@dataclass(frozen=True)
class DatasetSpec:
    """
    Immutable description of a dataset's shape and label scheme.

    Fields
    ------
    name          : short identifier, also used for embedding-cache prefixes.
    file_format   : "tsv" or "jsonl" — how a split file is read into a DataFrame.
    modalities    : ordered tuple of modality names, e.g. ("text", "image") or
                    ("text",). The order fixes the concatenation order the model
                    sees, so it must match what __getitem__ returns.
    text_col      : column holding raw text (used when encoding on the fly).
    id_col        : column holding the row id (used to locate images / cache).
    image_dir     : where images live (None for text-only datasets).
    embedding_dir : where precomputed embedding .npy caches live.
    label_schemes : maps an integer "n_way" selector to a LabelScheme describing
                    which column to read and how to turn it into a 0..K-1 target.
    """

    name: str
    file_format: str
    modalities: Tuple[str, ...]
    text_col: str
    id_col: str
    embedding_dir: str
    label_schemes: Dict[int, "LabelScheme"]
    image_dir: Optional[str] = None

    # ── Derived ───────────────────────────────────────────────────────────────
    @property
    def input_dim(self) -> int:
        """Total concatenated embedding width. Derived, never hardcoded."""
        return sum(EMBEDDING_DIMS[m] for m in self.modalities)

    def embedding_dim(self, modality: str) -> int:
        return EMBEDDING_DIMS[modality]

    def label_scheme(self, n_way: int) -> "LabelScheme":
        if n_way not in self.label_schemes:
            raise ValueError(
                f"Dataset '{self.name}' has no {n_way}-way label scheme. "
                f"Available: {sorted(self.label_schemes)}"
            )
        return self.label_schemes[n_way]

    def read_split(self, path: str) -> pd.DataFrame:
        if self.file_format == "tsv":
            return pd.read_csv(path, sep="\t", low_memory=False)
        elif self.file_format == "jsonl":
            return pd.read_json(path, lines=True)
        raise ValueError(f"Unknown file_format {self.file_format!r} for {self.name}")


@dataclass(frozen=True)
class LabelScheme:
    """
    Describes how to derive an integer target column from a raw label column.

    column        : source column name.
    num_classes   : K — number of output classes.
    str_map       : optional {lowercased-string: int} mapping for string labels.
    numeric_remap : optional {int: int} applied when the source column is already
                    numeric (e.g. Fakeddit's 2-way 0=True/1=Fake -> 0=Fake/1=True,
                    or Yelp's 1-5 stars -> 0-4).
    """
    column: str
    num_classes: int
    str_map: Optional[Dict[str, int]] = None
    numeric_remap: Optional[Dict[int, int]] = None


# ── Fakeddit ──────────────────────────────────────────────────────────────────

_FAKEDDIT_6WAY = {
    "true":                0,
    "satire":              1,
    "false connection":    2,
    "imposter content":    3,
    "manipulated content": 4,
    "misleading content":  5,
}
_FAKEDDIT_2WAY = {"true": 1, "fake": 0}

FAKEDDIT = DatasetSpec(
    name="fakeddit",
    file_format="tsv",
    modalities=("text", "image"),
    text_col="clean_title",
    id_col="id",
    image_dir="data/images/public_image_set",
    embedding_dir="data/embeddings",
    label_schemes={
        2: LabelScheme("2_way_label", 2, str_map=_FAKEDDIT_2WAY,
                       # TSV encodes 0=True,1=Fake; remap to 0=Fake,1=True
                       numeric_remap={0: 1, 1: 0}),
        6: LabelScheme("6_way_label", 6, str_map=_FAKEDDIT_6WAY,
                       numeric_remap=None),
    },
)


# ── Yelp sentiment ────────────────────────────────────────────────────────────
# JSONL rows: {"id", "timestamp", "sentiment": <1-5 star>, "text"}
# Star ratings 1..5 map to classes 0..4.
YELP = DatasetSpec(
    name="yelp",
    file_format="jsonl",
    modalities=("text",),
    text_col="text",
    id_col="id",
    image_dir=None,
    embedding_dir="data/embeddings",
    label_schemes={
        5: LabelScheme("sentiment", 5,
                       numeric_remap={1: 0, 2: 1, 3: 2, 4: 3, 5: 4}),
        # Convenience 2-way: 1-2 stars -> negative(0), 4-5 -> positive(1),
        # 3 stars dropped (mapped to None by absence from the remap).
        2: LabelScheme("sentiment", 2,
                       numeric_remap={1: 0, 2: 0, 4: 1, 5: 1}),
    },
)


# ── Registry ──────────────────────────────────────────────────────────────────

_REGISTRY: Dict[str, DatasetSpec] = {
    FAKEDDIT.name: FAKEDDIT,
    YELP.name: YELP,
}


def get_spec(name: str) -> DatasetSpec:
    key = name.lower()
    if key not in _REGISTRY:
        raise KeyError(
            f"Unknown dataset '{name}'. Registered: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[key]


def register(spec: DatasetSpec) -> None:
    _REGISTRY[spec.name] = spec