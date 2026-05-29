"""
plot_exp2_figure6.py

Reproduces Figure 6 from Stepanova & Ross (2023):
    Per-class F1, Precision, and Recall across 5 temporal test splits
    for three evaluation conditions:
        Normal   — trained model on unbalanced test splits
        Balanced — trained model on class-distribution-matched test splits
        Dummy    — hard majority classifier

Layout: 3 rows (F1, Precision, Recall) × 6 columns
        (Normal 2-way | Balanced 2-way | Dummy 2-way |
         Normal 6-way | Balanced 6-way | Dummy 6-way)

Usage:
    python plot_exp2_figure6.py
    python plot_exp2_figure6.py --runs_dir runs --output_dir plots

Run paths read from best_per_experiment.csv:
    runs/exp2_multi_2way / Multi_2way_n4096_lr0.0001 / best_model.pt  (hidden=4096)
    runs/exp2_multi_6way / Multi_6way_n4096_lr0.0001 / best_model.pt  (hidden=4096)

Balanced split filenames must match the output of balance_test_splits.py,
e.g. data/splits/Multi_test1_balanced_2way.tsv (2-way)
     data/splits/Multi_test1_balanced_6way.tsv (6-way)
Adjust BALANCED_PATTERN below if yours differ.
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from dataset import FakedditDataset


# ── Model ─────────────────────────────────────────────────────────────────────

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


# ── Run config ────────────────────────────────────────────────────────────────

EXP2_CONFIGS = {
    2: dict(
        exp_dir     = "exp2_multi_2way",
        run_name    = "Multi_2way_n4096_lr0.0001",
        hidden_size = 4096,
        label_col   = "2_way_label",
    ),
    6: dict(
        exp_dir     = "exp2_multi_6way",
        run_name    = "Multi_6way_n4096_lr0.0001",
        hidden_size = 4096,
        label_col   = "6_way_label",
    ),
}

# Adjust this pattern if balance_test_splits.py wrote different filenames.
# {t} = test number (1–5), {n} = n_way (2 or 6)
BALANCED_PATTERN = "data/splits/Multi_test{t}_balanced_{n}way.tsv"

TWO_WAY_CLASSES = {0: "False", 1: "True"}
SIX_WAY_CLASSES = {
    0: "True", 1: "Satire", 2: "False Con.",
    3: "Impost.", 4: "Manip.", 5: "Mislead.",
}

CLASS_COLORS_2 = {0: "#e41a1c", 1: "#377eb8"}
CLASS_COLORS_6 = {
    0: "#377eb8", 1: "#ff7f00", 2: "#4daf4a",
    3: "#984ea3", 4: "#e41a1c", 5: "#a65628",
}


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


def per_class_prf(y_true, y_pred, classes):
    """Returns {class_idx: {f1, precision, recall}}."""
    labels = sorted(classes.keys())
    p, r, f, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    return {c: {"f1": f[i], "precision": p[i], "recall": r[i]}
            for i, c in enumerate(labels)}


# ── Dummy metrics (analytical, no model needed) ───────────────────────────────

def dummy_per_class_prf(train_tsv, test_tsv, n_way):
    """Hard majority classifier: always predicts the most frequent training class."""
    label_col   = EXP2_CONFIGS[n_way]["label_col"]
    classes     = TWO_WAY_CLASSES if n_way == 2 else SIX_WAY_CLASSES

    train_df    = pd.read_csv(train_tsv, sep="\t", low_memory=False)
    test_df     = pd.read_csv(test_tsv,  sep="\t", low_memory=False)
    train_counts = train_df[label_col].value_counts()
    test_counts  = test_df[label_col].value_counts()
    majority     = int(train_counts.idxmax())
    n_test       = len(test_df)

    result = {}
    for c in classes:
        if c == majority:
            prec = test_counts.get(majority, 0) / n_test
            rec  = 1.0
            f1   = 2 * prec / (prec + 1) if (prec + 1) > 0 else 0.0
        else:
            prec, rec, f1 = 0.0, 0.0, 0.0
        result[c] = {"f1": f1, "precision": prec, "recall": rec}
    return result


# ── Collect all per-class metrics ─────────────────────────────────────────────

def collect_data(runs_dir, device):
    """
    Returns nested dict:
        data[n_way][condition][metric][class_idx] = [t1, t2, t3, t4, t5]
    """
    SPLITS_DIR = "data/splits"
    TRAIN_TSV  = os.path.join(SPLITS_DIR, "Multi_train.tsv")

    data = {}
    for n_way, cfg in EXP2_CONFIGS.items():
        classes    = TWO_WAY_CLASSES if n_way == 2 else SIX_WAY_CLASSES
        model_path = os.path.join(runs_dir, cfg["exp_dir"], cfg["run_name"], "best_model.pt")

        data[n_way] = {
            cond: {m: {c: [] for c in classes} for m in ("f1", "precision", "recall")}
            for cond in ("normal", "balanced", "dummy")
        }

        for t in range(1, 6):
            normal_tsv   = os.path.join(SPLITS_DIR, f"Multi_test{t}.tsv")
            balanced_tsv = BALANCED_PATTERN.format(t=t, n=n_way)

            # Normal
            y_true, y_pred = get_predictions(
                model_path, normal_tsv, n_way, cfg["hidden_size"], device
            )
            for c, vals in per_class_prf(y_true, y_pred, classes).items():
                for m in ("f1", "precision", "recall"):
                    data[n_way]["normal"][m][c].append(vals[m])

            # Balanced
            y_true, y_pred = get_predictions(
                model_path, balanced_tsv, n_way, cfg["hidden_size"], device
            )
            for c, vals in per_class_prf(y_true, y_pred, classes).items():
                for m in ("f1", "precision", "recall"):
                    data[n_way]["balanced"][m][c].append(vals[m])

            # Dummy (analytical)
            for c, vals in dummy_per_class_prf(TRAIN_TSV, normal_tsv, n_way).items():
                for m in ("f1", "precision", "recall"):
                    data[n_way]["dummy"][m][c].append(vals[m])

    return data


# ── Plot ──────────────────────────────────────────────────────────────────────

def make_figure6(data, output_dir):
    metrics    = ["f1", "precision", "recall"]
    col_specs  = [
        (2, "normal"),   (2, "balanced"),   (2, "dummy"),
        (6, "normal"),   (6, "balanced"),   (6, "dummy"),
    ]
    col_titles = [
        "2-way\nNormal", "2-way\nBalanced", "2-way\nDummy",
        "6-way\nNormal", "6-way\nBalanced", "6-way\nDummy",
    ]
    row_titles = ["F1", "Precision", "Recall"]
    x = np.arange(1, 6)

    fig, axes = plt.subplots(3, 6, figsize=(15, 7), sharex=True, sharey="row")
    fig.suptitle(
        "Figure 6: Per-class F1 / Precision / Recall across 5 Test Splits (Exp. 2)",
        fontsize=12
    )

    for col_idx, (n_way, condition) in enumerate(col_specs):
        classes = TWO_WAY_CLASSES if n_way == 2 else SIX_WAY_CLASSES
        colors  = CLASS_COLORS_2  if n_way == 2 else CLASS_COLORS_6

        for row_idx, metric in enumerate(metrics):
            ax = axes[row_idx, col_idx]

            for c, label in classes.items():
                ax.plot(x, data[n_way][condition][metric][c],
                        marker="o", markersize=3.5,
                        color=colors[c], label=label, linewidth=1.2)

            ax.set_xlim(0.5, 5.5)
            ax.set_ylim(-0.05, 1.05)
            ax.set_xticks(x)
            ax.tick_params(axis="both", labelsize=7)
            ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.5)

            if row_idx == 0:
                ax.set_title(col_titles[col_idx], fontsize=8, fontweight="bold")
            if col_idx == 0:
                ax.set_ylabel(row_titles[row_idx], fontsize=9)
            if row_idx == 2:
                ax.set_xlabel("Test Split", fontsize=8)

    # Legends below the figure
    handles_2 = [mlines.Line2D([], [], color=CLASS_COLORS_2[c], marker="o",
                                markersize=4, label=lbl)
                 for c, lbl in TWO_WAY_CLASSES.items()]
    handles_6 = [mlines.Line2D([], [], color=CLASS_COLORS_6[c], marker="o",
                                markersize=4, label=lbl)
                 for c, lbl in SIX_WAY_CLASSES.items()]

    leg1 = fig.legend(handles=handles_2, title="2-way",
                      loc="lower left",  bbox_to_anchor=(0.01, -0.05),
                      ncol=2, fontsize=7, title_fontsize=7)
    leg2 = fig.legend(handles=handles_6, title="6-way",
                      loc="lower right", bbox_to_anchor=(0.99, -0.05),
                      ncol=6, fontsize=7, title_fontsize=7)
    fig.add_artist(leg1)

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    out = os.path.join(output_dir, "figure6_exp2_per_class_metrics.png")
    plt.savefig(out, bbox_inches="tight", dpi=150)
    print(f"Saved: {out}")
    plt.close()


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Reproduce Figure 6 (Exp. 2 per-class metrics).")
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
    print("Collecting predictions...")
    data = collect_data(args.runs_dir, args.device)
    print("Plotting Figure 6...")
    make_figure6(data, args.output_dir)
    print("\nDone.")


if __name__ == "__main__":
    main()