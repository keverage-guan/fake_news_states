"""
src/precompute_embeddings.py

One-time script: encodes all split TSVs with distilroberta (text) and
ResNet-50 (images), saving .npy files to data/embeddings/.

Run once before training:
    python src/precompute_embeddings.py

Resumable: if interrupted mid-split, rerunning will pick up from the last
checkpoint within that split. Already-completed splits are skipped automatically
(use --force to redo them).
"""

import os
import glob
import argparse
import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm

import torch
import torchvision.models as models
import torchvision.transforms as T
from sentence_transformers import SentenceTransformer
from PIL import Image
Image.MAX_IMAGE_PIXELS = None  # disable decompression bomb check for this dataset

# ── Config ────────────────────────────────────────────────────────────────────

SPLITS_DIR       = "data/splits"
IMAGE_DIR        = "data/images/public_image_set"
EMBEDDING_DIR    = "data/embeddings"
CHECKPOINT_DIR   = "data/embeddings/checkpoints"
BATCH_SIZE       = 256    # for text encoding
IMG_BATCH        = 64     # for image encoding (adjust to VRAM)
CHECKPOINT_EVERY = 50     # save image checkpoint every N batches (~3200 images)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

IMAGE_TRANSFORM = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),   # deterministic for caching
    T.ToTensor(),
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_image(post_id: str):
    for ext in (".jpg", ".jpeg", ".png", ".gif"):
        path = os.path.join(IMAGE_DIR, f"{post_id}{ext}")
        if os.path.exists(path):
            try:
                return Image.open(path).convert("RGB")
            except (UnidentifiedImageError, OSError):
                return None
    return None


def build_resnet_encoder(device):
    """ResNet-50 with the final FC replaced by Identity → 2048-dim output."""
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    model.fc = torch.nn.Identity()
    model.eval()
    model.to(device)
    return model


# ── Main encoding logic ───────────────────────────────────────────────────────

def encode_text(titles: list, text_model: SentenceTransformer) -> np.ndarray:
    """Returns (N, 768) float32 array."""
    print("  Encoding text...")
    embeddings = text_model.encode(
        titles,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,
    )
    return embeddings.astype(np.float32)


def encode_images(post_ids: list, resnet, device, checkpoint_path: str = None) -> np.ndarray:
    """
    Returns (N, 2048) float32 array. Missing images → zero vector.
    Saves a checkpoint every CHECKPOINT_EVERY batches and resumes from it
    if interrupted.
    """
    print("  Encoding images...")
    all_embs    = []
    missing     = 0
    start_batch = 0

    # ── Resume from checkpoint if available ──────────────────────────────────
    if checkpoint_path and os.path.exists(checkpoint_path):
        data = np.load(checkpoint_path, allow_pickle=True).item()
        # Force each element back to a proper float32 array so we don't
        # end up with a numpy object array after list()
        all_embs    = [np.array(e, dtype=np.float32) for e in data["embs"]]
        missing     = int(data["missing"])
        start_batch = int(data["next_batch"])
        rows_done   = start_batch * IMG_BATCH
        print(f"  Resuming from batch {start_batch} ({rows_done:,} / {len(post_ids):,} rows done)")

    batches = list(range(0, len(post_ids), IMG_BATCH))

    for batch_num, i in enumerate(
        tqdm(batches[start_batch:], initial=start_batch, total=len(batches))
    ):
        batch_ids = post_ids[i : i + IMG_BATCH]
        tensors   = []
        indices   = []

        for j, pid in enumerate(batch_ids):
            img = load_image(str(pid))
            if img is not None:
                tensors.append(IMAGE_TRANSFORM(img))
                indices.append(j)
            else:
                missing += 1

        batch_embs = np.zeros((len(batch_ids), 2048), dtype=np.float32)

        if tensors:
            t = torch.stack(tensors).to(device)
            with torch.no_grad():
                out = resnet(t).cpu().numpy().astype(np.float32)
            for k, idx in enumerate(indices):
                batch_embs[idx] = out[k]

        all_embs.append(batch_embs)

        # ── Checkpoint ────────────────────────────────────────────────────
        # Use (batch_num + 1) % CHECKPOINT_EVERY, NOT actual_batch_num %
        # CHECKPOINT_EVERY. This ensures the interval is always relative to
        # batches processed in the current run, so it fires every 50 batches
        # regardless of where we resumed from.
        actual_batch_num = start_batch + batch_num + 1
        if checkpoint_path and (batch_num + 1) % CHECKPOINT_EVERY == 0:
            os.makedirs(CHECKPOINT_DIR, exist_ok=True)
            np.save(checkpoint_path, {
                "embs":       all_embs,
                "missing":    missing,
                "next_batch": actual_batch_num,
            })

    if missing:
        print(f"  Warning: {missing} images were missing/unreadable → zero vectors used.")

    # vstack FIRST, then delete checkpoint — so a crash here can't wipe the
    # checkpoint before the result is safely in memory
    result = np.vstack(all_embs)
    if checkpoint_path and os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
    return result


def process_split(tsv_path: str, text_model, resnet, device, force: bool = False):
    split_name  = os.path.splitext(os.path.basename(tsv_path))[0]
    text_out    = os.path.join(EMBEDDING_DIR,   f"{split_name}_text.npy")
    image_out   = os.path.join(EMBEDDING_DIR,   f"{split_name}_image.npy")
    img_ckpt    = os.path.join(CHECKPOINT_DIR,  f"{split_name}_image_ckpt.npy")

    if not force and os.path.exists(text_out) and os.path.exists(image_out):
        print(f"[{split_name}] Cache already exists, skipping. (Use --force to redo.)")
        return

    print(f"\n[{split_name}] Loading TSV...")
    df = pd.read_csv(tsv_path, sep="\t", low_memory=False)
    df = df.reset_index(drop=True)
    print(f"  {len(df):,} rows")

    titles   = [str(t) if pd.notna(t) else "" for t in df["clean_title"].tolist()]
    post_ids = df["id"].tolist()

    # Text
    if force or not os.path.exists(text_out):
        text_embs = encode_text(titles, text_model)
        np.save(text_out, text_embs)
        print(f"  Saved text embeddings → {text_out}  shape={text_embs.shape}")
    else:
        print(f"  Text cache exists, skipping.")

    # Images
    if force or not os.path.exists(image_out):
        img_embs = encode_images(post_ids, resnet, device, checkpoint_path=img_ckpt)
        np.save(image_out, img_embs)
        print(f"  Saved image embeddings → {image_out}  shape={img_embs.shape}")
    else:
        print(f"  Image cache exists, skipping.")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--splits", nargs="+", default=None,
        help="Specific split TSVs to process. Default: all in data/splits/."
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Recompute even if cache already exists."
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = parser.parse_args()

    os.makedirs(EMBEDDING_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    tsv_files = args.splits or sorted(glob.glob(os.path.join(SPLITS_DIR, "*.tsv")))
    if not tsv_files:
        print(f"No TSV files found in {SPLITS_DIR}. Did you run prepare_splits.py?")
        return

    print(f"Device: {args.device}")
    print(f"Found {len(tsv_files)} split(s) to process:")
    for f in tsv_files:
        print(f"  {f}")

    print("\nLoading text model (all-distilroberta-v1)...")
    text_model = SentenceTransformer("sentence-transformers/all-distilroberta-v1")

    print("Loading ResNet-50...")
    resnet = build_resnet_encoder(args.device)

    for tsv_path in tsv_files:
        process_split(tsv_path, text_model, resnet, args.device, force=args.force)

    print("\nDone. All embeddings saved to", EMBEDDING_DIR)


if __name__ == "__main__":
    main()