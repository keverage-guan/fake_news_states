"""
src/train.py

Trains the ensemble MLP classifier described in Stepanova & Ross (2023):
    concat(DistilRoBERTa 768-dim, ResNet-50 2048-dim) → hidden layer n → num_classes

Usage (single run, with validation):
    python src/train.py \
        --train data/splits/OG_train.tsv \
        --val   data/splits/OG_val.tsv   \
        --test  data/splits/OG_test.tsv  \
        --num_classes 2                  \
        --hidden_size 16384              \
        --lr 0.0001                      \
        --run_name OG_2way_n16384_lr1e-4

Usage (HMM window mode — no val, fixed epochs, explicit seed):
    python src/train.py \
        --train data/splits/hmm_windows/HMM_window_000.tsv \
        --no_val                         \
        --num_classes 6                  \
        --hidden_size 1024               \
        --lr 0.0001                      \
        --max_epochs 20                  \
        --seed 3                         \
        --run_name HMM_w000_s3_6way

Outputs (written to --output_dir/--run_name/):
    best_model.pt    ← state dict of best val epoch (or final epoch if --no_val)
    checkpoint.pt    ← full training state for resume (deleted on clean finish)
    metrics.json     ← train/val curves + final test scores

Resuming:
    Re-run the exact same command. If checkpoint.pt exists in the run dir,
    training resumes from where it left off automatically.
"""

import os
import json
import random
import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import sys
sys.path.insert(0, os.path.dirname(__file__))
from dataset import FakedditDataset


# ── Reproducibility ───────────────────────────────────────────────────────────

def seed_everything(seed: int) -> None:
    """Set all relevant RNG seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Deterministic cuDNN ops (slight perf cost; safe to remove if speed matters)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False




class MultimodalMLP(nn.Module):
    """
    One hidden layer feed-forward network.
    Input:  concat(text 768-dim, image 2048-dim) = 2816-dim
    Output: num_classes logits
    """
    def __init__(self, input_dim: int = 2816, hidden_size: int = 512, num_classes: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, text_emb: torch.Tensor, img_emb: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([text_emb, img_emb], dim=1))


# ── Checkpointing ─────────────────────────────────────────────────────────────

def save_checkpoint(path: str, payload: dict) -> None:
    """
    Atomic checkpoint write: save to <path>.tmp, then os.replace() into <path>.
    On POSIX this is a single syscall and cannot leave a partial file behind.
    """
    tmp = path + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)   # atomic on POSIX; overwrites on Windows too


def load_checkpoint(path: str, model, optimizer, device):
    """
    Load checkpoint and restore model + optimizer in-place.
    Returns (start_epoch, best_val_acc, epochs_no_impr, history).
    """
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    return (
        ckpt["epoch"] + 1,          # resume from next epoch
        ckpt["best_val_acc"],
        ckpt["epochs_no_impr"],
        ckpt["history"],
    )


# ── F1 helpers (no sklearn) ───────────────────────────────────────────────────

def compute_f1(labels: list, preds: list, num_classes: int):
    """
    Returns (micro_f1, macro_f1) using only the Python stdlib.

    In multiclass single-label classification:
      micro F1  = accuracy  (TP_total / N, since sum FP == sum FN == total wrong)
      macro F1  = unweighted mean of per-class F1 scores

    No external dependencies — avoids the sklearn/pyarrow/GLIBCXX issue on
    older HPC nodes.
    """
    tp = [0] * num_classes
    fp = [0] * num_classes
    fn = [0] * num_classes

    for true, pred in zip(labels, preds):
        if pred == true:
            tp[true] += 1
        else:
            fp[pred] += 1
            fn[true] += 1

    # Per-class F1
    f1s = []
    for c in range(num_classes):
        denom = 2 * tp[c] + fp[c] + fn[c]
        f1s.append(2 * tp[c] / denom if denom > 0 else 0.0)

    macro_f1 = sum(f1s) / num_classes

    # Micro F1 == accuracy in multiclass single-label setting
    total_tp = sum(tp)
    total_fp = sum(fp)
    total_fn = sum(fn)
    denom    = 2 * total_tp + total_fp + total_fn
    micro_f1 = 2 * total_tp / denom if denom > 0 else 0.0

    return micro_f1, macro_f1


# ── Evaluation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, loader, device, num_classes):
    model.eval()
    criterion = nn.CrossEntropyLoss()

    all_preds, all_labels = [], []
    total_loss, total_samples = 0.0, 0

    for text_emb, img_emb, labels in loader:
        text_emb, img_emb, labels = (
            text_emb.to(device), img_emb.to(device), labels.to(device)
        )
        logits = model(text_emb, img_emb)
        total_loss    += criterion(logits, labels).item() * labels.size(0)
        total_samples += labels.size(0)
        all_preds.extend(logits.argmax(1).cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    avg_loss = total_loss / total_samples
    accuracy = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
    micro_f1, macro_f1 = compute_f1(all_labels, all_preds, num_classes)
    return accuracy, micro_f1, macro_f1, avg_loss


# ── Training loop ─────────────────────────────────────────────────────────────

def train(args):
    device = torch.device(args.device)

    # ── Seed ─────────────────────────────────────────────────────────────
    # Must happen before model init and DataLoader creation so that weight
    # initialisation and batch ordering are fully determined by the seed.
    if args.seed is not None:
        seed_everything(args.seed)
        print(f"Seed: {args.seed}")

    # ── Output directory ─────────────────────────────────────────────────
    run_dir       = os.path.join(args.output_dir, args.run_name)
    os.makedirs(run_dir, exist_ok=True)
    best_ckpt     = os.path.join(run_dir, "best_model.pt")
    resume_ckpt   = os.path.join(run_dir, "checkpoint.pt")
    metrics_path  = os.path.join(run_dir, "metrics.json")

    # ── Skip if already complete ──────────────────────────────────────────
    # A finished run deletes checkpoint.pt and writes metrics.json.
    # If metrics.json exists and --force was not passed, nothing to do.
    if os.path.exists(metrics_path) and not args.force:
        print(f"Run already complete ({metrics_path} exists). Skipping.")
        print(f"Pass --force to retrain from scratch.")
        with open(metrics_path) as f:
            return json.load(f)


    # ── Datasets & loaders ───────────────────────────────────────────────
    print("\nLoading splits...")
    train_ds = FakedditDataset(args.train, n_way=args.num_classes, train=True)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=(args.device == "cuda"),
    )

    if not args.no_val:
        val_ds = FakedditDataset(args.val, n_way=args.num_classes, train=False)
        val_loader = DataLoader(
            val_ds, batch_size=args.batch_size * 2, shuffle=False,
            num_workers=args.num_workers, pin_memory=(args.device == "cuda"),
        )
        print(f"  Train: {len(train_ds):,}  Val: {len(val_ds):,}")
    else:
        val_loader = None
        print(f"  Train: {len(train_ds):,}  Val: (disabled -- fixed-epoch mode)")

    # ── Model, loss, optimizer ───────────────────────────────────────────
    model = MultimodalMLP(
        input_dim=2816, hidden_size=args.hidden_size, num_classes=args.num_classes
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # ── Resume from checkpoint if present ────────────────────────────────
    start_epoch    = 1
    best_val_acc   = -1.0
    epochs_no_impr = 0
    history        = []

    if os.path.exists(resume_ckpt):
        print(f"\nResuming from checkpoint: {resume_ckpt}")
        start_epoch, best_val_acc, epochs_no_impr, history = load_checkpoint(
            resume_ckpt, model, optimizer, device
        )
        print(f"  Resuming at epoch {start_epoch}  "
              f"(best val acc so far: {best_val_acc:.4f})")
    else:
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\nModel: 2816 → {args.hidden_size} → {args.num_classes}  "
              f"({params:,} params)")

    if args.no_val:
        print(f"\nTraining epochs {start_epoch}–{args.max_epochs}  "
              f"(fixed-epoch mode, no early stopping)\n")
    else:
        print(f"\nTraining epochs {start_epoch}–{args.max_epochs}  "
              f"(early stopping patience={args.patience})\n")

    # ── Epoch loop ───────────────────────────────────────────────────────
    for epoch in range(start_epoch, args.max_epochs + 1):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        for text_emb, img_emb, labels in train_loader:
            text_emb, img_emb, labels = (
                text_emb.to(device), img_emb.to(device), labels.to(device)
            )
            optimizer.zero_grad()
            logits = model(text_emb, img_emb)
            loss   = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss    += loss.item() * labels.size(0)
            train_correct += (logits.argmax(1) == labels).sum().item()
            train_total   += labels.size(0)

        train_loss /= train_total
        train_acc   = train_correct / train_total

        if not args.no_val:
            # ── Val eval + early stopping ──────────────────────────────
            val_acc, val_micro_f1, val_macro_f1, val_loss = evaluate(
                model, val_loader, device, args.num_classes
            )

            row = {
                "epoch":        epoch,
                "train_loss":   round(train_loss, 5),
                "train_acc":    round(train_acc, 5),
                "val_loss":     round(val_loss, 5),
                "val_acc":      round(val_acc, 5),
                "val_micro_f1": round(val_micro_f1, 5),
                "val_macro_f1": round(val_macro_f1, 5),
            }
            history.append(row)

            improved = val_acc > best_val_acc
            print(
                f"Epoch {epoch:02d} | "
                f"train loss={train_loss:.4f} acc={train_acc:.4f} | "
                f"val loss={val_loss:.4f} acc={val_acc:.4f} "
                f"micro_f1={val_micro_f1:.4f} macro_f1={val_macro_f1:.4f}"
                + (" ✓" if improved else "")
            )

            if improved:
                best_val_acc   = val_acc
                epochs_no_impr = 0
                save_checkpoint(best_ckpt, model.state_dict())
            else:
                epochs_no_impr += 1

        else:
            # ── No-val mode: just log train metrics, save every epoch ──
            row = {
                "epoch":      epoch,
                "train_loss": round(train_loss, 5),
                "train_acc":  round(train_acc, 5),
            }
            history.append(row)

            print(
                f"Epoch {epoch:02d} | "
                f"train loss={train_loss:.4f} acc={train_acc:.4f}"
            )

            # Overwrite best_model.pt each epoch; final epoch = the saved weights
            save_checkpoint(best_ckpt, model.state_dict())

        # ── Save resume checkpoint (atomic) ───────────────────────────
        # Written every epoch so a mid-epoch crash loses at most one epoch.
        save_checkpoint(resume_ckpt, {
            "epoch":           epoch,
            "model_state":     model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "best_val_acc":    best_val_acc,
            "epochs_no_impr":  epochs_no_impr,
            "history":         history,
        })

        if not args.no_val and epochs_no_impr >= args.patience:
            print(f"\nEarly stopping after epoch {epoch} "
                  f"({args.patience} epochs without improvement).")
            break

    # ── Test evaluation ──────────────────────────────────────────────────
    test_results = {}
    if args.test:
        print(f"\nEvaluating best model on test set(s)...")
        # Load best weights (not necessarily the last epoch's)
        model.load_state_dict(torch.load(best_ckpt, map_location=device))

        for test_path in args.test:
            split_name = os.path.splitext(os.path.basename(test_path))[0]
            test_ds    = FakedditDataset(test_path, n_way=args.num_classes, train=False)
            test_loader = DataLoader(
                test_ds, batch_size=args.batch_size * 2,
                shuffle=False, num_workers=args.num_workers,
            )
            acc, micro_f1, macro_f1, loss = evaluate(model, test_loader, device, args.num_classes)
            test_results[split_name] = {
                "accuracy":  round(acc, 5),
                "micro_f1":  round(micro_f1, 5),
                "macro_f1":  round(macro_f1, 5),
                "loss":      round(loss, 5),
                "n_samples": len(test_ds),
            }
            print(
                f"  [{split_name}]  "
                f"acc={acc:.4f}  micro_f1={micro_f1:.4f}  macro_f1={macro_f1:.4f}"
            )

    # ── Save metrics.json ────────────────────────────────────────────────
    metrics = {
        "run_name":     args.run_name,
        "num_classes":  args.num_classes,
        "hidden_size":  args.hidden_size,
        "lr":           args.lr,
        "batch_size":   args.batch_size,
        "seed":         args.seed,
        "no_val":       args.no_val,
        "best_val_acc": round(best_val_acc, 5) if not args.no_val else None,
        "epochs_run":   len(history),
        "train_split":  args.train,
        "val_split":    args.val if not args.no_val else None,
        "test_splits":  args.test,
        "history":      history,
        "test_results": test_results,
    }
    tmp = metrics_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(metrics, f, indent=2)
    os.replace(tmp, metrics_path)

    # ── Clean up resume checkpoint on successful finish ───────────────────
    # Done last: if anything above crashes, checkpoint.pt is still intact.
    if os.path.exists(resume_ckpt):
        os.remove(resume_ckpt)

    print(f"\nDone. Outputs in {run_dir}/")
    if args.no_val:
        print(f"  best_model.pt  (final epoch weights, epoch {len(history)})")
    else:
        print(f"  best_model.pt  (val acc={best_val_acc:.4f})")
    print(f"  metrics.json")
    return metrics


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train the Fakeddit multimodal MLP.")

    # Data
    p.add_argument("--train",        required=True)
    p.add_argument("--val",          default=None,
                   help="Validation TSV. Required unless --no_val is set.")
    p.add_argument("--no_val",       action="store_true",
                   help="Disable validation and early stopping. Train for exactly "
                        "--max_epochs epochs and save final weights. Intended for "
                        "the HMM window experiment where weight vectors must be a "
                        "clean function of training data, not val-set selection.")
    p.add_argument("--test",         nargs="+", default=None,
                   help="One or more test TSVs (multiple used for Exp. 2).")
    p.add_argument("--num_classes",  type=int, choices=[2, 6], default=2)

    # Model
    p.add_argument("--hidden_size",  type=int, required=True,
                   help="Width of the single hidden layer.")

    # Optimisation
    p.add_argument("--lr",           type=float, required=True,
                   help="Adam learning rate.")
    p.add_argument("--batch_size",   type=int, default=256)
    p.add_argument("--max_epochs",   type=int, default=20)
    p.add_argument("--patience",     type=int, default=4,
                   help="Early-stopping patience (epochs without val-acc improvement). "
                        "Ignored when --no_val is set.")

    # Reproducibility
    p.add_argument("--seed",         type=int, default=None,
                   help="Global RNG seed (random, numpy, torch). Required for "
                        "reproducible HMM window runs.")

    # Infra
    p.add_argument("--device",       default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--num_workers",  type=int, default=4)

    # Output
    p.add_argument("--output_dir",   default="runs")
    p.add_argument("--run_name",     default=None,
                   help="Auto-generated from hyperparams if omitted.")
    p.add_argument("--force",        action="store_true",
                   help="Retrain from scratch even if metrics.json already exists.")

    args = p.parse_args()

    # Validate val / no_val
    if args.no_val and args.val is not None:
        p.error("--val and --no_val are mutually exclusive.")
    if not args.no_val and args.val is None:
        p.error("--val is required unless --no_val is set.")

    if args.run_name is None:
        stem      = os.path.splitext(os.path.basename(args.train))[0]
        split_tag = stem.replace("_train", "")
        args.run_name = f"{split_tag}_{args.num_classes}way_n{args.hidden_size}_lr{args.lr}"

    return args


if __name__ == "__main__":
    args = parse_args()
    print("=" * 60)
    print(f"Run:          {args.run_name}")
    print(f"Classes:      {args.num_classes}-way")
    print(f"Hidden size:  {args.hidden_size}")
    print(f"LR:           {args.lr}")
    print(f"Seed:         {args.seed}")
    print(f"Val:          {'disabled (fixed-epoch mode)' if args.no_val else args.val}")
    print(f"Device:       {args.device}")
    print("=" * 60)
    train(args)