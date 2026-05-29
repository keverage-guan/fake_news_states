"""
plot_exp1_confusion.py

Reproduces the Experiment 1 confusion matrices from Stepanova & Ross (2023):
  - Figure 3: OG vs Temporal 2-way  (side-by-side)
  - Figure 4: Original 6-way
  - Figure 5: Temporal 6-way

Matrices are normalised over true labels (rows sum to 1), so diagonal
entries are per-class recall — matching the paper's convention.

Usage:
    python plot_exp1_confusion.py
    python plot_exp1_confusion.py --runs_dir runs --output_dir plots
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from dataset import FakedditDataset


# ── Model (must match src/train.py) ──────────────────────────────────────────

class MultimodalMLP(nn.Module):
    def __init__(self, input_dim=2816, hidden_size=512, num_classes=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, text_emb, img_emb):
        return self.net(torch.cat([text_emb, img_emb], dim=1))


# ── Run config (read directly from best_per_experiment.csv) ──────────────────
#
# experiment              run_name                    n  lr       hidden
# runs/exp1_OG_2way       OG_2way_n4096_lr0.001       2  0.001    4096
# runs/exp1_OG_6way       OG_6way_n1024_lr0.0001      6  0.0001   1024
# runs/exp1_temporal_2way Temporal_2way_n16384_lr0.001 2  0.001   16384
# runs/exp1_temporal_6way Temporal_6way_n1024_lr0.001  6  0.001   1024

EXP1_CONFIGS = {
    "OG_2way": dict(
        exp_dir     = "exp1_OG_2way",
        run_name    = "OG_2way_n4096_lr0.001",
        num_classes = 2,
        hidden_size = 4096,
        test_tsv    = "data/splits/OG_test.tsv",
    ),
    "Temporal_2way": dict(
        exp_dir     = "exp1_temporal_2way",
        run_name    = "Temporal_2way_n16384_lr0.001",
        num_classes = 2,
        hidden_size = 16384,
        test_tsv    = "data/splits/Temporal_test.tsv",
    ),
    "OG_6way": dict(
        exp_dir     = "exp1_OG_6way",
        run_name    = "OG_6way_n1024_lr0.0001",
        num_classes = 6,
        hidden_size = 1024,
        test_tsv    = "data/splits/OG_test.tsv",
    ),
    "Temporal_6way": dict(
        exp_dir     = "exp1_temporal_6way",
        run_name    = "Temporal_6way_n1024_lr0.001",
        num_classes = 6,
        hidden_size = 1024,
        test_tsv    = "data/splits/Temporal_test.tsv",
    ),
}

TWO_WAY_LABELS = ["False", "True"]
SIX_WAY_LABELS = ["True", "Satire", "False Con.", "Impost.", "Manip.", "Mislead."]


# ── Inference ─────────────────────────────────────────────────────────────────

def get_predictions(model_path, test_tsv, num_classes, hidden_size, device):
    ds     = FakedditDataset(test_tsv, n_way=num_classes, train=False)
    loader = DataLoader(ds, batch_size=512, shuffle=False, num_workers=4)

    model = MultimodalMLP(2816, hidden_size, num_classes).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for text_emb, img_emb, labels in loader:
            logits = model(text_emb.to(device), img_emb.to(device))
            all_preds.append(logits.argmax(1).cpu().numpy())
            all_labels.append(labels.numpy())

    return np.concatenate(all_labels), np.concatenate(all_preds)


# ── Plotting ──────────────────────────────────────────────────────────────────

def normalise_cm(y_true, y_pred, labels):
    """Row-normalised confusion matrix (rows = true classes → recall on diagonal)."""
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    row_sums = cm.sum(axis=1, keepdims=True)
    return np.where(row_sums > 0, cm / row_sums, 0.0)


def draw_cm(ax, cm_norm, display_labels, title):
    n   = len(display_labels)
    im  = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(n));  ax.set_xticklabels(display_labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(n));  ax.set_yticklabels(display_labels, fontsize=8)
    ax.set_xlabel("Predicted", fontsize=9)
    ax.set_ylabel("Actual",    fontsize=9)
    ax.set_title(title, fontsize=10, fontweight="bold")
    for i in range(n):
        for j in range(n):
            val   = cm_norm[i, j]
            color = "white" if val > 0.5 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7, color=color)
    return im


# ── Figure 3: side-by-side 2-way ─────────────────────────────────────────────

def figure3(runs_dir, output_dir, device):
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))
    fig.suptitle("Figure 3: 2-way Confusion Matrices (OG vs Temporal)", fontsize=11)

    for ax, key, title in zip(axes,
                               ["OG_2way", "Temporal_2way"],
                               ["ORIGINAL 2-WAY", "TEMPORAL 2-WAY"]):
        cfg        = EXP1_CONFIGS[key]
        model_path = os.path.join(runs_dir, cfg["exp_dir"], cfg["run_name"], "best_model.pt")
        y_true, y_pred = get_predictions(
            model_path, cfg["test_tsv"], cfg["num_classes"], cfg["hidden_size"], device
        )
        cm_norm = normalise_cm(y_true, y_pred, labels=[0, 1])
        draw_cm(ax, cm_norm, TWO_WAY_LABELS, title)

    plt.tight_layout()
    out = os.path.join(output_dir, "figure3_exp1_2way_confusion.png")
    plt.savefig(out, bbox_inches="tight", dpi=150)
    print(f"Saved: {out}")
    plt.close()


# ── Figures 4 & 5: 6-way ─────────────────────────────────────────────────────

def figures4_5(runs_dir, output_dir, device):
    specs = [
        ("OG_6way",       "Figure 4: ORIGINAL 6-WAY",  "figure4_exp1_6way_OG_confusion.png"),
        ("Temporal_6way", "Figure 5: TEMPORAL 6-WAY",   "figure5_exp1_6way_Temporal_confusion.png"),
    ]
    for key, suptitle, fname in specs:
        cfg        = EXP1_CONFIGS[key]
        model_path = os.path.join(runs_dir, cfg["exp_dir"], cfg["run_name"], "best_model.pt")
        y_true, y_pred = get_predictions(
            model_path, cfg["test_tsv"], cfg["num_classes"], cfg["hidden_size"], device
        )
        cm_norm = normalise_cm(y_true, y_pred, labels=list(range(6)))

        fig, ax = plt.subplots(figsize=(6, 5))
        fig.suptitle(suptitle, fontsize=11)
        im = draw_cm(ax, cm_norm, SIX_WAY_LABELS, suptitle.split(":")[1].strip())
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        plt.tight_layout()

        out = os.path.join(output_dir, fname)
        plt.savefig(out, bbox_inches="tight", dpi=150)
        print(f"Saved: {out}")
        plt.close()


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Generate Experiment 1 confusion matrices.")
    p.add_argument("--runs_dir",   default="runs",  help="Root runs directory.")
    p.add_argument("--output_dir", default="plots", help="Where to save PNGs.")
    p.add_argument("--device",     default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Device:     {args.device}")
    print(f"Runs dir:   {args.runs_dir}")
    print(f"Output dir: {args.output_dir}\n")
    figure3(args.runs_dir, args.output_dir, args.device)
    figures4_5(args.runs_dir, args.output_dir, args.device)
    print("\nDone.")


if __name__ == "__main__":
    main()