"""
src/plot_state_timeline.py

Publication-quality timeline figure with two panels:

  Top panel:    Colour-coded state bar — one cell per 60-day window,
                coloured by HMM state.  X-axis is calendar date.
                State labels annotated inside or above each run.

  Bottom panel: Stacked area chart of 6-way class proportions across
                windows, with the same calendar x-axis.

Both panels share the x-axis so state boundaries line up exactly with
shifts in class composition.

Outputs  (<output_dir>/)
-------
  plot_state_timeline.png   (300 dpi, publication-ready)
  window_class_props.csv    per-window class proportions used for the plot

Usage
-----
  python src/plot_state_timeline.py

  python src/plot_state_timeline.py \
      --decode_npz  data/hmm_hmm/6way/final_decode_k7.npz \
      --manifest    data/splits/hmm_windows/HMM_windows_manifest.csv \
      --output_dir  data/timeline/k7 \
      --width_inches 14 \
      --dpi 300
"""

import os
import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.dates as mdates



# ── Label maps ────────────────────────────────────────────────────────────────

CLASS_NAMES = [
    "True",
    "Satire",
    "False Connection",
    "Imposter Content",
    "Manipulated Content",
    "Misleading Content",
]
CLS_COLS = [f"cls_{i}" for i in range(6)]

# Colourblind-friendly class palette (Okabe-Ito inspired)
CLASS_COLORS = [
    "#009E73",   # True              — teal
    "#56B4E9",   # Satire            — sky blue
    "#E69F00",   # False Connection  — orange
    "#CC79A7",   # Imposter Content  — mauve
    "#D55E00",   # Manipulated       — vermillion
    "#0072B2",   # Misleading        — deep blue
]

# State palette (tab10-ish, consistent with other scripts)
STATE_PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c",
    "#d62728", "#9467bd", "#8c564b",
    "#17becf", "#e377c2"
]


# ── Args ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--decode_npz",   default="data/hmm_hmm/6way/final_decode_k6.npz")
    p.add_argument("--manifest",     default="data/splits/hmm_windows/HMM_windows_manifest.csv")
    p.add_argument("--output_dir",   default="data/timeline/k6")
    p.add_argument("--width_inches", type=float, default=14.0)
    p.add_argument("--dpi",          type=int,   default=300)
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Load ───────────────────────────────────────────────────────────────
    dec        = np.load(args.decode_npz, allow_pickle=True)
    state_seq  = dec["state_seq"].astype(int)
    window_ids = dec["window_ids"].astype(int)
    k          = int(dec.get("k", state_seq.max() + 1))
    N          = len(state_seq)

    manifest = pd.read_csv(args.manifest, parse_dates=["start_date", "end_date"])
    # Align to the window_ids from decode and preserve their order
    manifest = (manifest
                .set_index("window_local_idx")
                .loc[window_ids]
                .reset_index()
                .sort_values("window_local_idx")
                .reset_index(drop=True))
    # Re-derive state_seq in the same sorted order as manifest
    win_to_state = dict(zip(window_ids, state_seq))
    state_seq    = np.array([win_to_state[w] for w in manifest["window_local_idx"].values])

    for c in CLS_COLS:
        if c not in manifest.columns:
            manifest[c] = 0
        else:
            manifest[c] = manifest[c].fillna(0).astype(int)

    # ── Per-window class proportions ───────────────────────────────────────
    class_counts = manifest[CLS_COLS].values.astype(float)   # (N, 6)
    row_totals   = class_counts.sum(axis=1, keepdims=True)
    class_props  = class_counts / np.where(row_totals > 0, row_totals, 1)

    # Window midpoints for the area chart x-axis
    midpoints = manifest["start_date"] + (manifest["end_date"] - manifest["start_date"]) / 2

    # Save per-window proportions
    prop_df = pd.DataFrame(class_props, columns=CLASS_NAMES)
    prop_df.insert(0, "window_id",  window_ids)
    prop_df.insert(1, "state",      state_seq)
    prop_df.insert(2, "start_date", manifest["start_date"].values)
    prop_df.insert(3, "end_date",   manifest["end_date"].values)
    prop_path = os.path.join(args.output_dir, "window_class_props.csv")
    prop_df.to_csv(prop_path, index=False)
    print(f"Saved: {prop_path}")

    # ── Figure layout ──────────────────────────────────────────────────────
    # Two panels: state bar (thin) + stacked area (main)
    fig, axes = plt.subplots(
        2, 1,
        figsize=(args.width_inches, 6),
        gridspec_kw={"height_ratios": [1, 5], "hspace": 0.08},
    )
    ax_bar   = axes[0]
    ax_stack = axes[1]

    # Shared x limits: span from first window start to last window end
    x_min = mdates.date2num(manifest["start_date"].min())
    x_max = mdates.date2num(manifest["end_date"].max())

    # ── Top panel: state colour bar ────────────────────────────────────────
    for i in range(N):
        x0  = mdates.date2num(manifest["start_date"].iloc[i])
        x1  = mdates.date2num(manifest["end_date"].iloc[i])
        s   = state_seq[i]
        ax_bar.barh(0, x1 - x0, left=x0, height=1.0,
                    color=STATE_PALETTE[s % len(STATE_PALETTE)],
                    edgecolor="white", linewidth=0.4)

    # Annotate each contiguous run of the same state with its label
    # Find runs
    run_starts = [0]
    for i in range(1, N):
        if state_seq[i] != state_seq[i - 1]:
            run_starts.append(i)
    run_starts.append(N)

    for r in range(len(run_starts) - 1):
        i_start = run_starts[r]
        i_end   = run_starts[r + 1] - 1
        s       = state_seq[i_start]
        x_left  = mdates.date2num(manifest["start_date"].iloc[i_start])
        x_right = mdates.date2num(manifest["end_date"].iloc[i_end])
        x_mid   = (x_left + x_right) / 2
        ax_bar.text(x_mid, 0.5, f"S{s}",
                    ha="center", va="center", fontsize=8,
                    fontweight="bold", color="black")

    ax_bar.set_xlim(x_min, x_max)
    ax_bar.set_ylim(0, 1)
    ax_bar.set_yticks([])
    ax_bar.set_xticks([])
    ax_bar.set_ylabel("State", fontsize=9, labelpad=4)
    ax_bar.spines[["top", "right", "bottom", "left"]].set_visible(False)

    # State legend
    state_patches = [
        mpatches.Patch(color=STATE_PALETTE[s], label=f"State {s}")
        for s in range(k)
    ]
    ax_bar.legend(handles=state_patches, loc="upper right",
                  fontsize=7, ncol=k, framealpha=0.85,
                  bbox_to_anchor=(1.0, 1.5))

    # ── Bottom panel: stacked area chart ───────────────────────────────────
    x_dates = mdates.date2num(midpoints.values)

    # Sort classes by mean proportion descending so dominant classes are at
    # the bottom of the stack (easier to read)
    mean_props  = class_props.mean(axis=0)
    order       = np.argsort(mean_props)[::-1]   # descending

    # Build cumulative stack
    stack_bottom = np.zeros(N)
    for ci in order:
        y = class_props[:, ci]
        ax_stack.fill_between(
            x_dates, stack_bottom, stack_bottom + y,
            color=CLASS_COLORS[ci],
            alpha=0.85,
            label=CLASS_NAMES[ci],
            step=None,
        )
        # Thin border between classes
        ax_stack.plot(x_dates, stack_bottom + y,
                      color="white", lw=0.3, alpha=0.5)
        stack_bottom += y

    # Vertical dashed lines at state boundaries
    for r in range(1, len(run_starts) - 1):
        i_start = run_starts[r]
        x_bound = mdates.date2num(manifest["start_date"].iloc[i_start])
        ax_stack.axvline(x_bound, color="black", lw=1.0,
                         linestyle="--", alpha=0.5, zorder=5)

    ax_stack.set_xlim(x_min, x_max)
    ax_stack.set_ylim(0, 1)
    ax_stack.set_ylabel("Class proportion", fontsize=11)
    ax_stack.set_xlabel("Date", fontsize=11)

    # Date formatting
    ax_stack.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax_stack.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.setp(ax_stack.xaxis.get_majorticklabels(), rotation=35, ha="right",
             fontsize=8)

    ax_stack.grid(True, axis="y", alpha=0.25, linestyle="--")
    ax_stack.spines[["top", "right"]].set_visible(False)

    # Class legend — two rows so it doesn't crowd the plot
    class_patches = [
        mpatches.Patch(color=CLASS_COLORS[ci], label=CLASS_NAMES[ci])
        for ci in order
    ]
    ax_stack.legend(handles=class_patches,
                    loc="upper left",
                    bbox_to_anchor=(0.0, -0.22),
                    fontsize=8, ncol=3, framealpha=0.85)

    fig.suptitle("HMM state sequence and class composition over time",
                 fontsize=13, y=1.01)

    out_path = os.path.join(args.output_dir, "plot_state_timeline.png")
    fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")
    print("Done.")


if __name__ == "__main__":
    main()