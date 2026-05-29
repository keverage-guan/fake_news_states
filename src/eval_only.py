"""
src/eval_only.py

Loads a trained best_model.pt and evaluates it on one or more test TSVs
without any retraining. Used for the Exp. 2 Balanced condition.

Typical workflow
----------------
1. The Exp. 2 Normal sweep finishes and you identify the best (n, lr) pair.
2. Point this script at that run's best_model.pt and the balanced TSVs.

Usage
-----
# Evaluate a specific model checkpoint on balanced splits
python src/eval_only.py \\
    --model  runs/exp2_multi_2way/Multi_2way_n2048_lr0.0001/best_model.pt \\
    --num_classes 2 \\
    --hidden_size 2048 \\
    --tests  data/splits/Multi_test1_balanced_2way.tsv \\
             data/splits/Multi_test2_balanced_2way.tsv \\
             data/splits/Multi_test3_balanced_2way.tsv \\
             data/splits/Multi_test4_balanced_2way.tsv \\
             data/splits/Multi_test5_balanced_2way.tsv \\
    --output runs/exp2_balanced_2way.json

# Or: auto-find the best run in a sweep directory by val accuracy
python src/eval_only.py \\
    --sweep_dir  runs/exp2_multi_2way \\
    --num_classes 2 \\
    --tests  data/splits/Multi_test1_balanced_2way.tsv ... \\
    --output runs/exp2_balanced_2way.json

# 6-way version
python src/eval_only.py \\
    --sweep_dir  runs/exp2_multi_6way \\
    --num_classes 6 \\
    --tests  data/splits/Multi_test1_balanced_6way.tsv ... \\
    --output runs/exp2_balanced_6way.json

Outputs
-------
    Prints per-split metrics to stdout.
    Writes a JSON file to --output (if provided) with:
        - model provenance (path, hidden_size, val_acc of the chosen run)
        - per-split accuracy, micro F1, macro F1, n_samples
"""

import os
import sys
import json
import glob
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(__file__))
from dataset import FakedditDataset


# ── Model (must match train.py exactly) ───────────────────────────────────────

class MultimodalMLP(nn.Module):
    def __init__(self, input_dim: int = 2816, hidden_size: int = 512, num_classes: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, text_emb: torch.Tensor, img_emb: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([text_emb, img_emb], dim=1))


# ── F1 (copied from train.py — no sklearn dependency) ─────────────────────────

def compute_f1(labels: list, preds: list, num_classes: int):
    tp = [0] * num_classes
    fp = [0] * num_classes
    fn = [0] * num_classes
    for true, pred in zip(labels, preds):
        if pred == true:
            tp[true] += 1
        else:
            fp[pred] += 1
            fn[true] += 1
    f1s = []
    for c in range(num_classes):
        denom = 2 * tp[c] + fp[c] + fn[c]
        f1s.append(2 * tp[c] / denom if denom > 0 else 0.0)
    macro_f1 = sum(f1s) / num_classes
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


# ── Sweep-directory helpers ───────────────────────────────────────────────────

def find_best_run(sweep_dir: str) -> tuple[str, dict]:
    """
    Scan all metrics.json files under sweep_dir and return the path to the
    best_model.pt whose run achieved the highest best_val_acc.

    Returns (best_model_path, metrics_dict).
    """
    pattern = os.path.join(sweep_dir, "**", "metrics.json")
    candidates = glob.glob(pattern, recursive=True)
    if not candidates:
        raise FileNotFoundError(
            f"No metrics.json files found under {sweep_dir}. "
            "Has the sweep finished?"
        )

    best_val_acc  = -1.0
    best_model_pt = None
    best_metrics  = None

    for mpath in candidates:
        with open(mpath) as f:
            m = json.load(f)
        val_acc = m.get("best_val_acc", -1.0)
        if val_acc > best_val_acc:
            best_val_acc  = val_acc
            best_model_pt = os.path.join(os.path.dirname(mpath), "best_model.pt")
            best_metrics  = m

    if not os.path.exists(best_model_pt):
        raise FileNotFoundError(
            f"Best run's best_model.pt not found: {best_model_pt}"
        )

    return best_model_pt, best_metrics


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate a saved best_model.pt on test splits (no retraining)."
    )

    # Model source: either explicit path or sweep dir auto-discovery
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--model",
        help="Direct path to a best_model.pt checkpoint."
    )
    src.add_argument(
        "--sweep_dir",
        help="Sweep output directory. Auto-selects the run with highest best_val_acc."
    )

    # Architecture — required when using --model; auto-read from metrics.json
    # when using --sweep_dir
    p.add_argument(
        "--hidden_size", type=int, default=None,
        help="Hidden layer width. Required with --model; auto-detected with --sweep_dir."
    )

    p.add_argument("--num_classes",  type=int, choices=[2, 6], required=True)
    p.add_argument(
        "--tests", nargs="+", required=True,
        help="One or more test TSVs to evaluate (e.g. the balanced splits)."
    )
    p.add_argument("--batch_size",   type=int, default=512)
    p.add_argument("--num_workers",  type=int, default=4)
    p.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu"
    )
    p.add_argument(
        "--output", default=None,
        help="Optional path to write results JSON."
    )
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    # ── Resolve model path and architecture ──────────────────────────────
    provenance = {}

    if args.sweep_dir:
        print(f"Scanning sweep directory: {args.sweep_dir}")
        model_path, sweep_metrics = find_best_run(args.sweep_dir)
        hidden_size = sweep_metrics["hidden_size"]
        print(f"  Best run : {sweep_metrics['run_name']}")
        print(f"  Val acc  : {sweep_metrics['best_val_acc']:.4f}")
        print(f"  Model    : {model_path}")
        provenance = {
            "source":       "sweep_dir",
            "sweep_dir":    args.sweep_dir,
            "run_name":     sweep_metrics["run_name"],
            "best_val_acc": sweep_metrics["best_val_acc"],
            "hidden_size":  hidden_size,
            "lr":           sweep_metrics.get("lr"),
        }
    else:
        model_path = args.model
        if args.hidden_size is None:
            raise ValueError("--hidden_size is required when using --model.")
        hidden_size = args.hidden_size
        print(f"Model : {model_path}")
        provenance = {
            "source":      "explicit",
            "model_path":  model_path,
            "hidden_size": hidden_size,
        }

    # ── Load model ───────────────────────────────────────────────────────
    print(f"\nLoading model (2816 → {hidden_size} → {args.num_classes})...")
    model = MultimodalMLP(
        input_dim=2816, hidden_size=hidden_size, num_classes=args.num_classes
    ).to(device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    print("  Loaded successfully.")

    # ── Evaluate on each test split ──────────────────────────────────────
    print(f"\nEvaluating on {len(args.tests)} test split(s)...\n")

    results = {}
    rows    = []

    for test_path in args.tests:
        stem = os.path.splitext(os.path.basename(test_path))[0]

        test_ds = FakedditDataset(test_path, n_way=args.num_classes, train=False)
        test_loader = DataLoader(
            test_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=(args.device == "cuda"),
        )

        acc, micro_f1, macro_f1, loss = evaluate(
            model, test_loader, device, args.num_classes
        )

        print(
            f"  [{stem}]  n={len(test_ds):,}  "
            f"acc={acc:.4f}  micro_f1={micro_f1:.4f}  macro_f1={macro_f1:.4f}"
        )

        results[stem] = {
            "n_samples": len(test_ds),
            "accuracy":  round(acc, 5),
            "micro_f1":  round(micro_f1, 5),
            "macro_f1":  round(macro_f1, 5),
            "loss":      round(loss, 5),
        }
        rows.append({
            "split":    stem,
            "n":        len(test_ds),
            "micro_f1": round(micro_f1, 4),
            "macro_f1": round(macro_f1, 4),
        })

    # ── Summary table ────────────────────────────────────────────────────
    print("\n── Summary ──")
    col_w = max(len(r["split"]) for r in rows)
    print(f"  {'split':<{col_w}}   n          micro_f1   macro_f1")
    print(f"  {'-'*col_w}   ---------  ---------  ---------")
    for r in rows:
        print(
            f"  {r['split']:<{col_w}}   {r['n']:<9,}  "
            f"{r['micro_f1']:.4f}     {r['macro_f1']:.4f}"
        )

    # ── Optional JSON output ─────────────────────────────────────────────
    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        payload = {
            "num_classes": args.num_classes,
            "model":       provenance,
            "results":     results,
        }
        tmp = args.output + ".tmp"
        with open(tmp, "w") as f:
            json.dump(payload, f, indent=2)
        os.replace(tmp, args.output)
        print(f"\nResults saved → {args.output}")


if __name__ == "__main__":
    main()