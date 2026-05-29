"""
src/collect_results.py

Aggregates all metrics.json files from one or more experiment directories
into two outputs:

  1. results_summary.csv  — written into each experiment directory,
                            all runs ranked by best_val_acc

  2. best_per_experiment.csv — written to --output, one row per experiment
                               directory showing only the best run (by
                               best_val_acc, matching the paper's model
                               selection criterion)

Usage:
    python src/collect_results.py \\
        runs/exp1_OG_2way \\
        runs/exp1_OG_6way \\
        runs/exp1_temporal_2way \\
        runs/exp1_temporal_6way \\
        --output runs/best_per_experiment.csv

Console output:
    A compact table of the best run per experiment directory.
"""

import os
import csv
import json
import argparse


# ── Helpers ───────────────────────────────────────────────────────────────────

def collect_metrics(exp_dir: str) -> list[dict]:
    """
    Recursively scan exp_dir for metrics.json files.
    Returns a list of dicts, one per completed run, sorted by best_val_acc desc.
    """
    results = []
    for root, _, files in os.walk(exp_dir):
        if "metrics.json" in files:
            path = os.path.join(root, "metrics.json")
            try:
                with open(path) as f:
                    m = json.load(f)
                m["_source_dir"] = exp_dir
                results.append(m)
            except (json.JSONDecodeError, OSError) as e:
                print(f"  Warning: could not read {path}: {e}")

    results.sort(key=lambda m: m.get("best_val_acc", -1), reverse=True)
    return results


def build_rows(m: dict) -> list[dict]:
    """
    Flatten one metrics.json into one CSV row per test split.
    Each row gets a 'test_split' column instead of per-split column prefixes.
    """
    base = {
        "run_name":     m.get("run_name", ""),
        "num_classes":  m.get("num_classes", ""),
        "hidden_size":  m.get("hidden_size", ""),
        "lr":           m.get("lr", ""),
        "best_val_acc": m.get("best_val_acc", ""),
        "epochs_run":   m.get("epochs_run", ""),
    }
    test_results = m.get("test_results", {})
    if not test_results:
        return [{**base, "test_split": "", "micro_f1": "", "macro_f1": "", "acc": ""}]
    rows = []
    for split, res in test_results.items():
        rows.append({
            **base,
            "test_split": split,
            "micro_f1":   res.get("micro_f1", ""),
            "macro_f1":   res.get("macro_f1", ""),
            "acc":        res.get("accuracy", ""),
        })
    return rows


FIELDNAMES = [
    "run_name", "num_classes", "hidden_size", "lr",
    "best_val_acc", "epochs_run", "test_split",
    "micro_f1", "macro_f1", "acc",
]


def write_csv(path: str, rows: list[dict], fieldnames: list[str]) -> None:
    """Atomic CSV write."""
    tmp = path + ".tmp"
    with open(tmp, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


# ── Main ──────────────────────────────────────────────────────────────────────

def find_exp_dirs(runs_dir: str) -> list[str]:
    """
    Return all immediate subdirectories of runs_dir that contain at least
    one metrics.json somewhere inside them, sorted alphabetically.
    """
    if not os.path.isdir(runs_dir):
        return []
    found = []
    for entry in sorted(os.scandir(runs_dir), key=lambda e: e.name):
        if not entry.is_dir():
            continue
        for _, _, files in os.walk(entry.path):
            if "metrics.json" in files:
                found.append(entry.path)
                break
    return found


def main():
    p = argparse.ArgumentParser(
        description="Collect training results into summary CSVs."
    )
    p.add_argument(
        "--runs_dir", default="runs",
        help="Root directory to scan for experiment subdirs (default: runs/)."
    )
    p.add_argument(
        "--output", default=None,
        help="Path for the combined best-run CSV. "
             "Defaults to <runs_dir>/best_per_experiment.csv."
    )
    args = p.parse_args()

    if args.output is None:
        args.output = os.path.join(args.runs_dir, "best_per_experiment.csv")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    exp_dirs = find_exp_dirs(args.runs_dir)
    if not exp_dirs:
        print(f"No experiment directories found under {args.runs_dir}/")
        return
    print(f"Found {len(exp_dirs)} experiment dir(s) under {args.runs_dir}/:")
    for d in exp_dirs:
        print(f"  {d}")

    best_rows = []   # one per experiment dir (may be multiple rows if multiple splits)

    print()

    for exp_dir in exp_dirs:
        if not os.path.isdir(exp_dir):
            print(f"  Warning: {exp_dir} is not a directory — skipping.")
            continue

        metrics_list = collect_metrics(exp_dir)

        if not metrics_list:
            print(f"  {exp_dir}: no completed runs found.")
            continue

        n_runs = len(metrics_list)
        rows   = []
        for m in metrics_list:
            rows.extend(build_rows(m))

        # ── Per-experiment results_summary.csv ───────────────────────────
        summary_path = os.path.join(exp_dir, "results_summary.csv")
        write_csv(summary_path, rows, FIELDNAMES)

        # ── Best run (top of sorted list) ────────────────────────────────
        best      = metrics_list[0]
        best_rows_exp = build_rows(best)
        for brow in best_rows_exp:
            brow["experiment"] = exp_dir
        best_rows.extend(best_rows_exp)

        # ── Console output ────────────────────────────────────────────────
        print(f"  {exp_dir}  ({n_runs} runs)")
        print(f"    best: {best['run_name']}")
        print(f"      hidden={best['hidden_size']}  lr={best['lr']}"
              f"  val_acc={best['best_val_acc']:.4f}"
              f"  epochs={best['epochs_run']}")
        for split, res in best.get("test_results", {}).items():
            if res:
                print(f"      [{split}]  "
                      f"micro_f1={res.get('micro_f1','?'):.4f}  "
                      f"macro_f1={res.get('macro_f1','?'):.4f}  "
                      f"acc={res.get('accuracy','?'):.4f}")
        print(f"    → {summary_path}  ({n_runs} runs, {len(rows)} rows)\n")

    # ── Combined best_per_experiment.csv ─────────────────────────────────
    if best_rows:
        combined_fields = ["experiment"] + FIELDNAMES
        write_csv(args.output, best_rows, combined_fields)
        print(f"  Combined best-run table → {args.output}  ({len(best_rows)} rows)")

    print()


if __name__ == "__main__":
    main()