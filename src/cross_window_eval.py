"""
src/cross_window_eval.py

Step 4 (of 6): Cross-window F1 evaluation — intentionally HMM-agnostic.

For a single training window (--train_window_idx), load all seed models
and evaluate each on every valid test window.  Saves one row file:

    data/hmm_perf/<n_way>way/rows/row_NNN.npz

After all array tasks finish, run src/merge_cross_window.py to assemble
the full N×N F1 matrix.  HMM state labels are joined *later* in
src/within_across_states.py, keeping the two pipelines independent.

Designed to be run as a SLURM array job; see cross_window_eval.slurm.

Usage
-----
    python src/cross_window_eval.py \\
        --runs_dir         runs/hmm_windows/6way \\
        --splits_dir       data/splits/hmm_windows \\
        --output_dir       data/hmm_perf/6way \\
        --n_way            6 \\
        --hidden_size      1024 \\
        --seed_agg         mean \\
        --batch_size       512 \\
        --num_workers      4 \\
        --device           cuda \\
        --train_window_idx 0
"""

import os
import sys
import time
import glob
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score

sys.path.insert(0, os.path.dirname(__file__))
from train import MultimodalMLP
from dataset import FakedditDataset


# ── helpers ───────────────────────────────────────────────────────────────────

def elapsed(t0: float) -> str:
    s = time.time() - t0
    return f"{s:.1f}s" if s < 60 else f"{s / 60:.1f}min"


def find_valid_windows(runs_dir: str) -> list[int]:
    """
    Return sorted list of window indices that have at least one trained
    seed model (best_model.pt) under runs_dir/window_XXX/seed_Y/.
    """
    dirs = sorted(glob.glob(os.path.join(runs_dir, "window_???")))
    valid = []
    for d in dirs:
        name = os.path.basename(d)
        try:
            idx = int(name.split("_")[1])
        except (IndexError, ValueError):
            continue
        if glob.glob(os.path.join(d, "seed_*", "best_model.pt")):
            valid.append(idx)
    return sorted(valid)


def load_seed_models(
    runs_dir: str,
    window_idx: int,
    hidden_size: int,
    n_way: int,
    device: torch.device,
) -> dict[int, torch.nn.Module]:
    """
    Load every seed model for a given window.
    Returns {seed_idx: model (eval mode, on device)}.
    """
    window_dir = os.path.join(runs_dir, f"window_{window_idx:03d}")
    models: dict[int, torch.nn.Module] = {}

    for sd in sorted(glob.glob(os.path.join(window_dir, "seed_*"))):
        pt = os.path.join(sd, "best_model.pt")
        if not os.path.isfile(pt):
            continue
        try:
            seed_idx = int(os.path.basename(sd).split("_")[1])
        except (IndexError, ValueError):
            continue

        ckpt = torch.load(pt, map_location=device)
        state_dict = ckpt.get("model_state", ckpt)   # handle both formats

        model = MultimodalMLP(input_dim=2816,
                              hidden_size=hidden_size,
                              num_classes=n_way)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        models[seed_idx] = model

    return models


@torch.no_grad()
def eval_model_on_window(
    model: torch.nn.Module,
    window_tsv: str,
    n_way: int,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> tuple[float, np.ndarray]:
    """
    Evaluate one model on one window.
    Returns (macro_f1, per_class_f1 of shape (n_way,)).
    """
    ds = FakedditDataset(window_tsv, n_way=n_way, train=False)
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    all_preds, all_labels = [], []
    for text_emb, img_emb, labels in dl:
        text_emb = text_emb.to(device)
        img_emb  = img_emb.to(device)
        logits   = model(text_emb, img_emb)
        preds    = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.tolist())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)
    labels_range = list(range(n_way))

    macro_f1  = float(f1_score(all_labels, all_preds,
                                average="macro", zero_division=0))
    per_class = f1_score(all_labels, all_preds, average=None,
                          zero_division=0,
                          labels=labels_range).astype(np.float32)
    return macro_f1, per_class


# ── plot helpers (imported by merge_cross_window.py) ─────────────────────────

def plot_heatmap(
    f1_matrix: np.ndarray,
    valid_ids: list[int],
    output_path: str,
) -> None:
    """Plain F1 heatmap — no HMM state annotations."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(valid_ids)

    # Mask diagonal (within-distribution) so it doesn't bias the colormap.
    # The saved matrix is untouched; this copy is plot-only.
    plot_mat = f1_matrix.copy().astype(float)
    np.fill_diagonal(plot_mat, np.nan)

    fig, ax = plt.subplots(figsize=(max(6, n * 0.4), max(5, n * 0.4)))
    im  = ax.imshow(plot_mat, aspect="auto", cmap="RdYlGn",
                    vmin=np.nanmin(plot_mat), vmax=np.nanmax(plot_mat))
    plt.colorbar(im, ax=ax, label="Macro F1")

    tick_labels = [f"W{i:03d}" for i in valid_ids]
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(tick_labels, rotation=90, fontsize=7)
    ax.set_yticklabels(tick_labels, fontsize=7)
    ax.set_xlabel("Test window", fontsize=11)
    ax.set_ylabel("Train window", fontsize=11)
    ax.set_title("Cross-window macro F1", fontsize=13)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_path}")


def _save_and_plot(
    f1_matrix: np.ndarray,
    f1_per_class: np.ndarray,
    f1_per_seed_cube: np.ndarray,
    valid_ids: list[int],
    args,           # needs: output_dir, n_way
    t0: float,
) -> None:
    """
    Persist the assembled F1 matrix and generate plots.

    Deliberately HMM-agnostic: no state_seq / window_ids / k parameters.
    HMM state labels are joined downstream in within_across_states.py.
    """
    import pandas as pd
    os.makedirs(args.output_dir, exist_ok=True)

    # ── NPZ ──────────────────────────────────────────────────────────────
    npz_path = os.path.join(args.output_dir, "cross_window_f1.npz")
    np.savez(
        npz_path,
        f1_matrix        = f1_matrix,
        f1_per_class     = f1_per_class,
        f1_per_seed_cube = f1_per_seed_cube,
        valid_ids        = np.array(valid_ids, dtype=np.int32),
        n_way            = np.int32(args.n_way),
    )
    print(f"  Saved: {npz_path}")

    # ── CSV ───────────────────────────────────────────────────────────────
    csv_path = os.path.join(args.output_dir, "cross_window_f1.csv")
    pd.DataFrame(
        f1_matrix,
        index   = [f"train_{i:03d}" for i in valid_ids],
        columns = [f"test_{j:03d}"  for j in valid_ids],
    ).to_csv(csv_path)
    print(f"  Saved: {csv_path}")

    # ── Heatmap ───────────────────────────────────────────────────────────
    plot_heatmap(f1_matrix, valid_ids,
                 os.path.join(args.output_dir, "heatmap_f1.png"))

    print(f"  Total wall time: {elapsed(t0)}")


# ── main ─────────────────────────────────────────────────────────────────────

def main(args: argparse.Namespace) -> None:
    t0     = time.time()
    device = torch.device(
        args.device if (args.device == "cpu" or not torch.cuda.is_available())
        else args.device
    )
    rows_dir = os.path.join(args.output_dir, "rows")
    os.makedirs(rows_dir, exist_ok=True)

    print("=" * 60)
    print(f"  src/cross_window_eval.py  —  train window {args.train_window_idx}")
    print("=" * 60)
    print(f"  Device : {device}")

    # ── 1. Discover valid windows ─────────────────────────────────────────
    valid_ids = find_valid_windows(args.runs_dir)
    print(f"\n  Valid windows ({len(valid_ids)}): {valid_ids}")

    if args.train_window_idx not in valid_ids:
        print(f"  [skip] Window {args.train_window_idx} has no trained models.")
        return

    row_idx = valid_ids.index(args.train_window_idx)
    n_valid = len(valid_ids)

    # ── 2. Load seed models ───────────────────────────────────────────────
    print(f"\n[2] Loading models for window {args.train_window_idx:03d} ...")
    models = load_seed_models(
        args.runs_dir, args.train_window_idx,
        args.hidden_size, args.n_way, device,
    )
    if not models:
        print(f"  ERROR: no best_model.pt found for window {args.train_window_idx:03d}")
        return

    seed_ids = sorted(models.keys())
    n_seeds  = len(seed_ids)
    print(f"  Loaded {n_seeds} seed models: seeds {seed_ids}")

    # ── 3. Evaluate on every window ───────────────────────────────────────
    row_f1          = np.full(n_valid, np.nan, dtype=np.float32)
    row_per_class   = np.full((n_valid, args.n_way), np.nan, dtype=np.float32)
    row_f1_per_seed = np.full((n_seeds, n_valid), np.nan, dtype=np.float32)

    print(f"\n[3] Evaluating on {n_valid} windows ...")
    for j, test_win in enumerate(valid_ids):
        tsv = os.path.join(args.splits_dir, f"HMM_window_{test_win:03d}.tsv")
        if not os.path.isfile(tsv):
            print(f"  [warn] TSV missing: {tsv} — skipping")
            continue

        seed_f1s, seed_pcs = [], []
        for si, seed_idx in enumerate(seed_ids):
            f1, pc = eval_model_on_window(
                models[seed_idx], tsv, args.n_way,
                args.batch_size, args.num_workers, device,
            )
            row_f1_per_seed[si, j] = f1
            seed_f1s.append(f1)
            seed_pcs.append(pc)

        if seed_f1s:
            agg = np.mean if args.seed_agg == "mean" else np.median
            row_f1[j]      = float(agg(seed_f1s))
            row_per_class[j] = agg(seed_pcs, axis=0)

        marker = " ← (self)" if test_win == args.train_window_idx else ""
        print(f"    test W{test_win:03d}: F1 = {row_f1[j]:.4f}{marker}")

    # ── 4. Save row file ──────────────────────────────────────────────────
    row_path = os.path.join(rows_dir, f"row_{row_idx:03d}.npz")
    np.savez(
        row_path,
        row_idx         = np.int32(row_idx),
        valid_ids       = np.array(valid_ids, dtype=np.int32),
        row_f1          = row_f1,
        row_per_class   = row_per_class,
        row_f1_per_seed = row_f1_per_seed,
    )
    print(f"\n  Saved: {row_path}")
    print(f"  Elapsed: {elapsed(t0)}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Cross-window F1 evaluation for one training window. "
            "HMM-agnostic: does not require decode_npz. "
            "Run merge_cross_window.py after all rows are done, "
            "then within_across_states.py to join with HMM decode."
        )
    )
    p.add_argument("--runs_dir",
                   default="runs/hmm_windows/6way",
                   help="Root dir containing window_XXX/seed_Y/best_model.pt")
    p.add_argument("--splits_dir",
                   default="data/splits/hmm_windows",
                   help="Dir containing HMM_window_XXX.tsv files")
    p.add_argument("--output_dir",
                   default="data/hmm_perf/6way",
                   help="Output root; row files go to output_dir/rows/")
    p.add_argument("--n_way",            type=int, default=6, choices=[2, 6])
    p.add_argument("--hidden_size",      type=int, default=1024)
    p.add_argument("--seed_agg",         default="mean",
                   choices=["mean", "median"],
                   help="How to aggregate F1 across seeds")
    p.add_argument("--batch_size",       type=int, default=512)
    p.add_argument("--num_workers",      type=int, default=4)
    p.add_argument("--device",           default="cuda")
    p.add_argument("--train_window_idx", type=int, required=True,
                   help="Which window's models to evaluate (SLURM_ARRAY_TASK_ID)")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())