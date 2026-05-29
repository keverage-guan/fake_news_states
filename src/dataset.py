"""
src/dataset.py

FakedditDataset: loads a split TSV, reads images and text,
returns pre-computed embeddings (if cache exists) or computes on the fly.

Expected directory layout:
    data/
        splits/          ← TSV files produced by prepare_splits.py
        images/          ← extracted from public_images.tar.bz2
        embeddings/      ← created by precompute_embeddings.py (optional but fast)

Each TSV row must have at minimum:
    id, clean_title, 2_way_label, 6_way_label
"""

import os
import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError
Image.MAX_IMAGE_PIXELS = None  # disable decompression bomb check for this dataset

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

# ── Constants ─────────────────────────────────────────────────────────────────

IMAGE_DIR      = "data/images/public_image_set"
EMBEDDING_DIR  = "data/embeddings"

# ImageNet stats used by the paper (He et al. 2016 / PyTorch default)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# Map string label columns → integer targets
LABEL_COLS = {
    2: "2_way_label",
    6: "6_way_label",
}

# 6-way string → int
SIX_WAY_MAP = {
    "true":                 0,
    "satire":               1,
    "false connection":     2,
    "imposter content":     3,
    "manipulated content":  4,
    "misleading content":   5,
}

TWO_WAY_MAP = {
    "true":  1,
    "fake":  0,
}


# ── Image transform (paper: resize → random crop → normalize) ─────────────────

def get_image_transform(train: bool = True) -> T.Compose:
    if train:
        return T.Compose([
            T.Resize(256),
            T.RandomCrop(224),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    else:
        return T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])


# ── Dataset ───────────────────────────────────────────────────────────────────

class FakedditDataset(Dataset):
    """
    Parameters
    ----------
    tsv_path : str
        Path to a split TSV (e.g. "data/splits/OG_train.tsv").
    n_way : int
        2 or 6 — which label column to use.
    train : bool
        Controls image augmentation (random vs. center crop).
    text_model : sentence_transformers.SentenceTransformer or None
        If None, expects pre-computed text embeddings in EMBEDDING_DIR.
    resnet : torch.nn.Module or None
        If None, expects pre-computed image embeddings in EMBEDDING_DIR.
        If provided, should be ResNet-50 with the final FC layer removed
        (returns 2048-dim vectors).
    device : str
        "cuda" or "cpu" — used when encoding on-the-fly.
    """

    def __init__(
        self,
        tsv_path: str,
        n_way: int = 2,
        train: bool = True,
        text_model=None,
        resnet=None,
        device: str = "cpu",
    ):
        assert n_way in (2, 6), "n_way must be 2 or 6"

        self.n_way       = n_way
        self.train       = train
        self.text_model  = text_model
        self.resnet      = resnet
        self.device      = device
        self.transform   = get_image_transform(train)

        # Derive a split name used for embedding cache filenames
        # e.g. "data/splits/OG_train.tsv" → "OG_train"
        self.split_name  = os.path.splitext(os.path.basename(tsv_path))[0]

        # Load metadata
        self.df = pd.read_csv(tsv_path, sep="\t", low_memory=False)
        self.df = self.df.reset_index(drop=True)

        # Normalise label column to lowercase string, map to int
        label_col = LABEL_COLS[n_way]
        label_map = TWO_WAY_MAP if n_way == 2 else SIX_WAY_MAP

        raw = self.df[label_col]

        # If the column is already numeric (integers 0..N), use it directly.
        # Otherwise treat as strings and map through the label dict.
        if pd.api.types.is_numeric_dtype(raw):
            if n_way == 2:
                # TSV encodes 0=True, 1=Fake; remap to 0=Fake, 1=True
                self.df["_label"] = raw.map({0: 1, 1: 0})
            else:
                self.df["_label"] = raw
        else:
            self.df["_label"] = (
                raw.astype(str).str.lower().str.strip().map(label_map)
            )

        # Drop rows whose label is NaN (unmapped strings or genuine NaNs)
        before = len(self.df)
        self.df = self.df.dropna(subset=["_label"]).reset_index(drop=True)
        after  = len(self.df)
        if before != after:
            print(f"[FakedditDataset] Dropped {before - after} rows with unmapped/missing labels.")

        self.df["_label"] = self.df["_label"].astype(int)

        # ── Load or locate pre-computed embeddings ────────────────────────
        self._text_embs  = self._try_load_text_embeddings()
        self._image_embs = self._try_load_image_embeddings()

        if self._text_embs is not None:
            assert len(self._text_embs) == len(self.df), (
                f"Text embedding cache has {len(self._text_embs)} rows "
                f"but split TSV has {len(self.df)} rows."
            )
        if self._image_embs is not None:
            assert len(self._image_embs) == len(self.df), (
                f"Image embedding cache has {len(self._image_embs)} rows "
                f"but split TSV has {len(self.df)} rows."
            )

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _text_cache_path(self) -> str:
        return os.path.join(EMBEDDING_DIR, f"{self.split_name}_text.npy")

    def _image_cache_path(self) -> str:
        return os.path.join(EMBEDDING_DIR, f"{self.split_name}_image.npy")

    def _try_load_text_embeddings(self):
        p = self._text_cache_path()
        if os.path.exists(p):
            print(f"[FakedditDataset] Loading cached text embeddings: {p}")
            return np.load(p)
        return None

    def _try_load_image_embeddings(self):
        p = self._image_cache_path()
        if os.path.exists(p):
            print(f"[FakedditDataset] Loading cached image embeddings: {p}")
            return np.load(p)
        return None

    def _load_image(self, post_id: str):
        """Try common extensions. Returns a PIL Image or None."""
        for ext in (".jpg", ".jpeg", ".png", ".gif"):
            path = os.path.join(IMAGE_DIR, f"{post_id}{ext}")
            if os.path.exists(path):
                try:
                    img = Image.open(path).convert("RGB")
                    return img
                except (UnidentifiedImageError, OSError):
                    return None
        return None

    # ── Dataset interface ─────────────────────────────────────────────────────

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row   = self.df.iloc[idx]
        label = torch.tensor(row["_label"], dtype=torch.long)

        # ── Text embedding ──────────────────────────────────────────────
        if self._text_embs is not None:
            text_emb = torch.tensor(self._text_embs[idx], dtype=torch.float32)
        elif self.text_model is not None:
            text = str(row["clean_title"]) if pd.notna(row["clean_title"]) else ""
            with torch.no_grad():
                text_emb = torch.tensor(
                    self.text_model.encode(text, show_progress_bar=False),
                    dtype=torch.float32,
                )
        else:
            raise RuntimeError(
                "No text embeddings found. Either run precompute_embeddings.py "
                "or pass a text_model to FakedditDataset."
            )

        # ── Image embedding ─────────────────────────────────────────────
        if self._image_embs is not None:
            img_emb = torch.tensor(self._image_embs[idx], dtype=torch.float32)
        else:
            img = self._load_image(str(row["id"]))
            if img is None:
                # Missing image: use a zero vector (consistent fallback)
                img_emb = torch.zeros(2048, dtype=torch.float32)
            else:
                tensor = self.transform(img).unsqueeze(0).to(self.device)
                if self.resnet is not None:
                    with torch.no_grad():
                        img_emb = self.resnet(tensor).squeeze(0).cpu()
                else:
                    raise RuntimeError(
                        "No image embeddings found. Either run precompute_embeddings.py "
                        "or pass a resnet model to FakedditDataset."
                    )

        return text_emb, img_emb, label

    # ── Utility ───────────────────────────────────────────────────────────────

    def get_class_weights(self) -> torch.Tensor:
        """
        Returns inverse-frequency weights for each class,
        useful for weighted cross-entropy loss.
        Shape: (n_way,)
        """
        counts = self.df["_label"].value_counts().sort_index()
        weights = 1.0 / counts.values.astype(float)
        weights = weights / weights.min()
        return torch.tensor(weights, dtype=torch.float32)

    def get_class_distribution(self) -> dict:
        """Returns {class_int: count} dict."""
        return self.df["_label"].value_counts().sort_index().to_dict()