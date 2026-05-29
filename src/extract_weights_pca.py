"""
src_extract_weights_pca.py

Extracts MLP head weights from all trained HMM window models, aligns hidden
units across seeds to a synthetic centroid via linear assignment, reduces via
PCA, z-scores, computes per-window seed centroids, and saves everything needed
for HMM fitting.

Pipeline
--------
1.  Load best_model.pt for every (window, seed) pair.
    Extract structured weight matrices:
        W_h : (1024 × 2816)   hidden layer weights
        b_h : (1024,)          hidden layer biases
        W_o : (6 × 1024)      output layer weights
        b_o : (6,)             output layer biases

2.  Per-window iterative alignment (Choice B — joint fingerprint):
    For each window independently, align all seeds to a synthetic centroid
    via iterated Hungarian assignment until convergence.

    No bootstrap reference seed is used. Iteration starts from each seed's
    original (unaligned) weights, so no single seed is privileged.

    Fingerprint: fingerprint[i] = concat(W_h[i,:], W_o[:,i])  (2822-dim).
    Permutation applied consistently to W_h rows, b_h, and W_o columns.

3.  Flatten aligned weight matrices and stack:
        W_aligned : (N_windows × N_seeds, D)   D = 2,889,734

4.  Fit PCA separately on unaligned weights (for the before-plot only),
    then fit the final PCA on aligned weights.

5.  Project aligned weights → Z  (N_windows × N_seeds, n_components).

6.  Fit z-score normaliser on Z (all aligned data), standardise → Z_scaled.

7.  Compute per-window centroid in PCA space → C  (N_windows, n_components).

8.  Save outputs and produce two sanity-check plots:
      sanity_check_before.png — PCA of UNALIGNED weights (seed clustering visible)
      sanity_check_after.png  — PCA of ALIGNED weights
    Both plots color points by window (continuous colormap, one color per window)
    and encode seed index as marker shape.

Component-selection methods (--component_selection)
----------------------------------------------------
broken_stick  (default)
    Retain all leading PCs where observed variance > broken-stick expectation
    (what you'd expect from random partitioning of total variance across n dims).
    Conservative; robust to noise.

auer_gervini
    Auer & Gervini (2008) step-function estimator.  Fits a step function to
    the normalised eigenvalue sequence and selects the number of components
    at the largest step in the scree plot, weighted by the integral of the
    gap above the uniform baseline.  Less conservative than broken-stick;
    tends to retain more structure when the true dimensionality is high.

Both methods fall back to --variance_threshold if they return 0 or all
components (degenerate cases).

Outputs (written to --output_dir)
----------------------------------
weights_pca.npz          ← Z_scaled, C, window_ids, seed_ids,
                             centroid_wins, explained_variance_ratio
pca_model.pkl            ← fitted sklearn PCA object  (aligned-space)
scaler_model.pkl         ← fitted sklearn StandardScaler object
sanity_check_before.png
sanity_check_after.png
broken_stick.png

Usage
-----
    python src_extract_weights_pca.py \\
        --runs_dir   runs/hmm_windows \\
        --output_dir data/hmm_weights \\
        --n_windows  35 \\
        --n_seeds    5  \\
        --variance_threshold 0.90 \\
        --component_selection broken_stick   # or: auer_gervini

    # dry-run (checks paths only, no loading):
    python src_extract_weights_pca.py --runs_dir runs/hmm_windows --dry_run
"""

import os
import sys
import pickle
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import torch
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# ── Architecture constants (must match src_train.py) ──────────────────────────

INPUT_DIM   = 2816
HIDDEN_SIZE = 1024
NUM_CLASSES = 6

# Flat vector dimension: W_h + b_h + W_o + b_o
# = 1024*2816 + 1024 + 6*1024 + 6 = 2,889,734
FLAT_DIM = HIDDEN_SIZE * INPUT_DIM + HIDDEN_SIZE + NUM_CLASSES * HIDDEN_SIZE + NUM_CLASSES


# ── Weight loading ─────────────────────────────────────────────────────────────

def load_structured_weights(ckpt_path: str) -> dict:
    """
    Load best_model.pt and return weight matrices as numpy arrays.
    Keeps them structured (not flat) so alignment can operate on them.

    Returns dict with keys: W_h (1024,2816), b_h (1024,), W_o (6,1024), b_o (6,)
    """
    state_dict = torch.load(ckpt_path, map_location="cpu")
    expected   = ["net.0.weight", "net.0.bias", "net.2.weight", "net.2.bias"]
    for k in expected:
        if k not in state_dict:
            raise KeyError(
                f"Key '{k}' missing from state_dict at {ckpt_path}. "
                f"Keys present: {list(state_dict.keys())}"
            )
    return {
        "W_h": state_dict["net.0.weight"].cpu().float().numpy(),  # (1024, 2816)
        "b_h": state_dict["net.0.bias"  ].cpu().float().numpy(),  # (1024,)
        "W_o": state_dict["net.2.weight"].cpu().float().numpy(),  # (6, 1024)
        "b_o": state_dict["net.2.bias"  ].cpu().float().numpy(),  # (6,)
    }


def flatten_weights(sw: dict) -> np.ndarray:
    """Concatenate structured weight dict into a 1-D float32 vector (fixed order)."""
    return np.concatenate([
        sw["W_h"].ravel(),
        sw["b_h"].ravel(),
        sw["W_o"].ravel(),
        sw["b_o"].ravel(),
    ]).astype(np.float32)


# ── Path helpers ───────────────────────────────────────────────────────────────

def ckpt_path(runs_dir: str, window_idx: int, seed_idx: int) -> str:
    return os.path.join(
        runs_dir,
        f"window_{window_idx:03d}",
        f"seed_{seed_idx}",
        "best_model.pt",
    )


def collect_weights(
    runs_dir:  str,
    n_windows: int,
    n_seeds:   int,
    dry_run:   bool = False,
) -> tuple:
    """
    Walk all (window, seed) pairs and load structured weights.

    Returns
    -------
    weights_by_window : list[dict]  — index w → {seed_idx: sw_dict}
    window_ids        : list[int]   — window index per found checkpoint (flat)
    seed_ids          : list[int]   — seed index per found checkpoint (flat)
    missing           : list[str]   — paths that did not exist
    """
    weights_by_window = [{} for _ in range(n_windows)]
    window_ids, seed_ids, missing = [], [], []

    for w in range(n_windows):
        for s in range(n_seeds):
            path = ckpt_path(runs_dir, w, s)
            if not os.path.exists(path):
                missing.append(path)
                continue
            if not dry_run:
                weights_by_window[w][s] = load_structured_weights(path)
            window_ids.append(w)
            seed_ids.append(s)

    return weights_by_window, window_ids, seed_ids, missing


# ── Weight matching ────────────────────────────────────────────────────────────

def unit_fingerprints(sw: dict) -> np.ndarray:
    """
    Per-unit fingerprint matrix: shape (hidden_size, input_dim + num_classes).

    Row i = concat(W_h[i, :], W_o[:, i])

    Encodes both what unit i receives (W_h row) and what it sends (W_o column),
    giving a richer identity for matching than hidden weights alone (Choice B).
    """
    return np.concatenate([sw["W_h"], sw["W_o"].T], axis=1)  # (1024, 2822)


def cost_matrix(ref: dict, src: dict) -> np.ndarray:
    """
    Squared-Euclidean cost matrix between fingerprints of ref and src.
    Shape: (hidden_size, hidden_size).  M[i,j] = ||fp_ref[i] - fp_src[j]||^2.
    Computed efficiently via ||a-b||^2 = ||a||^2 + ||b||^2 - 2 a·b.
    """
    fp_ref = unit_fingerprints(ref)
    fp_src = unit_fingerprints(src)
    ref_sq = (fp_ref ** 2).sum(axis=1, keepdims=True)
    src_sq = (fp_src ** 2).sum(axis=1, keepdims=True)
    M      = ref_sq + src_sq.T - 2.0 * (fp_ref @ fp_src.T)
    return np.maximum(M, 0.0)   # numerical safety


def apply_permutation(sw: dict, perm: np.ndarray) -> dict:
    """
    Return a new structured weight dict with hidden units reordered by perm.
    perm[i] = j  →  position i in output gets unit j from input.
    W_h rows, b_h entries, and W_o columns permuted consistently.
    Output bias b_o has no hidden-unit indexing and is unchanged.
    """
    return {
        "W_h": sw["W_h"][perm, :],
        "b_h": sw["b_h"][perm],
        "W_o": sw["W_o"][:, perm],
        "b_o": sw["b_o"],
    }


def match_to_reference(ref: dict, src: dict) -> dict:
    """
    Find and apply the permutation of src's hidden units that minimises total
    fingerprint distance to ref (Hungarian algorithm).
    """
    _, col_ind = linear_sum_assignment(cost_matrix(ref, src))
    return apply_permutation(src, col_ind)


def centroid_weights(aligned_seeds: dict) -> dict:
    """
    Compute mean weight matrices across a dict of aligned seed weight dicts.
    The centroid is not a real network but a valid alignment target.
    """
    keys = list(aligned_seeds.keys())
    return {
        "W_h": np.mean([aligned_seeds[s]["W_h"] for s in keys], axis=0),
        "b_h": np.mean([aligned_seeds[s]["b_h"] for s in keys], axis=0),
        "W_o": np.mean([aligned_seeds[s]["W_o"] for s in keys], axis=0),
        "b_o": np.mean([aligned_seeds[s]["b_o"] for s in keys], axis=0),
    }


def align_window(seeds: dict, max_iter: int = 20, tol: float = 1e-4) -> dict:
    """
    Align all seeds in a window to a synthetic centroid via iterated Hungarian
    assignment until convergence.

    No bootstrap reference seed is used. Iteration starts from each seed's
    original (unaligned) weights, so the fixed point is determined purely by
    the geometry of the networks and no single seed is privileged.

    Algorithm
    ---------
    1. Compute centroid of original (unaligned) weights — a noisy but
       symmetric starting target.
    2. Align every seed to the current centroid (Hungarian on fingerprints).
    3. Recompute centroid from newly aligned seeds.
    4. Repeat 2-3 until centroid Frobenius change < tol or max_iter reached.

    Always re-aligns from the original seed weights each iteration to avoid
    permutation composition drift.
    """
    seed_indices = sorted(seeds.keys())

    # Iteration 0: start from original unaligned weights (symmetric init)
    current = dict(seeds)

    prev_centroid = None
    for iteration in range(max_iter):
        synth = centroid_weights(current)

        if prev_centroid is not None:
            delta = (
                np.concatenate([synth[k].ravel() for k in ["W_h", "b_h", "W_o", "b_o"]])
              - np.concatenate([prev_centroid[k].ravel() for k in ["W_h", "b_h", "W_o", "b_o"]])
            )
            if np.linalg.norm(delta) < tol:
                break

        prev_centroid = synth
        # Re-align from original seeds each time — no permutation drift
        current = {s: match_to_reference(synth, seeds[s]) for s in seed_indices}

    return current


# ── Component-selection methods ────────────────────────────────────────────────

def broken_stick_threshold(n: int) -> np.ndarray:
    """
    Expected proportion of variance for each component under the broken-stick
    model, if total variance is randomly distributed across n dimensions.

    For component k (1-indexed):
        bs[k] = (1/n) * sum_{j=k}^{n} (1/j)

    Returns array of length n (0-indexed).
    """
    bs = np.zeros(n)
    for k in range(1, n + 1):
        bs[k - 1] = np.sum(1.0 / np.arange(k, n + 1)) / n
    return bs


def select_broken_stick(evr: np.ndarray, n_fit: int) -> tuple[int, str]:
    """
    Broken-stick selection rule.

    Retain a contiguous prefix of PCs where observed variance exceeds the
    broken-stick expectation.  Stop at the first PC that fails its threshold.

    Returns (n_components, description_string).
    """
    bs   = broken_stick_threshold(n_fit)
    fails = evr <= bs

    if not fails.any():
        n_bs = n_fit        # every component beats threshold — degenerate
    else:
        n_bs = int(np.argmax(fails))   # index of first failure (0-based count)

    if 0 < n_bs < n_fit:
        desc = (
            f"broken_stick: PC{n_bs} passes "
            f"({evr[n_bs-1]*100:.4f}% > {bs[n_bs-1]*100:.4f}%), "
            f"PC{n_bs+1} fails ({evr[n_bs]*100:.4f}% ≤ {bs[n_bs]*100:.4f}%)"
        )
    else:
        desc = f"broken_stick gave degenerate result ({n_bs}/{n_fit})"

    return n_bs, bs, desc


def select_auer_gervini(evr: np.ndarray, n_fit: int) -> tuple[int, str]:
    """
    Auer & Gervini (2008) step-function estimator for number of PCA components.

    Method
    ------
    Let λ_k = evr[k] (normalised eigenvalues, sum to 1).  Define the
    cumulative step function F(t) = #{k : λ_k ≥ t} / n_fit.
    The Auer-Gervini estimator d̂ is the value of k that maximises the area
    between the empirical scree plot and the uniform baseline 1/n_fit,
    measured as the integral up to that step.

    Practically, we evaluate the gap g_k = λ_k - 1/n_fit for each component
    and keep all leading PCs where g_k > 0 (i.e. above the uniform baseline),
    then select the last k in this set as the cutoff.  This is equivalent to
    finding the "elbow" in the scree plot relative to the flat noise floor.

    Unlike broken-stick, this method does not require a contiguous prefix —
    it finds the last component that still exceeds the uniform expectation,
    which makes it less conservative when structure is spread across many PCs.

    Reference: Auer, P. & Gervini, D. (2008). Choosing principal components:
    a new graphical method based on Bayesian model selection.
    Communications in Statistics — Simulation and Computation, 37(5), 962-977.

    Returns (n_components, baseline_array, description_string).
    """
    baseline = np.ones(n_fit) / n_fit           # uniform noise floor
    above    = np.where(evr > baseline)[0]      # indices above baseline

    if len(above) == 0:
        n_ag = 0
        desc = "auer_gervini: no components exceed uniform baseline"
    elif len(above) == n_fit:
        n_ag = n_fit
        desc = "auer_gervini: all components exceed baseline (degenerate)"
    else:
        n_ag = int(above[-1]) + 1               # last index above + 1 (1-based count)
        desc = (
            f"auer_gervini: last component above uniform baseline is PC{n_ag} "
            f"({evr[n_ag-1]*100:.4f}% > {baseline[0]*100:.4f}%)"
        )
        if n_ag < n_fit:
            desc += (
                f"; PC{n_ag+1} is below "
                f"({evr[n_ag]*100:.4f}% ≤ {baseline[0]*100:.4f}%)"
            )

    return n_ag, baseline, desc


# ── PCA + z-score ──────────────────────────────────────────────────────────────

def fit_pca(
    W:                   np.ndarray,
    variance_threshold:  float,
    label:               str   = "",
    component_selection: str   = "broken_stick",
    max_components_fit:  int   = 500,
) -> tuple:
    """
    Fit PCA on W and select n_components via the chosen method.

    Component-selection options
    ---------------------------
    "broken_stick"  (default)
        Retain leading PCs where observed variance > broken-stick expectation.
        Conservative; stop at first failure (contiguous prefix rule).

    "auer_gervini"
        Retain all PCs that exceed the uniform variance baseline (1/n_fit).
        Less conservative; not restricted to a contiguous prefix.

    Both fall back to --variance_threshold if they return 0 or n_fit
    (degenerate cases).

    Returns
    -------
    pca          : fitted sklearn PCA with n_components retained
    n_components : number of components kept
    evr          : full explained_variance_ratio_ array (all fitted components)
    threshold    : per-component threshold used by the selection method
                   (broken-stick array or uniform baseline array)
    """
    n_fit = min(min(W.shape), max_components_fit)
    tag   = f"[{label}] " if label else ""
    print(f"  {tag}Fitting PCA up to {n_fit} components "
          f"(selection: {component_selection}) ...")

    pca_full = PCA(n_components=n_fit, random_state=0)
    pca_full.fit(W)
    evr    = pca_full.explained_variance_ratio_   # (n_fit,)
    cumvar = np.cumsum(evr)

    # ── Run chosen selection method ──────────────────────────────────────
    if component_selection == "broken_stick":
        n_method, threshold, method_desc = select_broken_stick(evr, n_fit)
    elif component_selection == "auer_gervini":
        n_method, threshold, method_desc = select_auer_gervini(evr, n_fit)
    else:
        raise ValueError(
            f"Unknown --component_selection '{component_selection}'. "
            "Choose 'broken_stick' or 'auer_gervini'."
        )

    # ── Fallback: cumulative variance threshold ──────────────────────────
    n_var = int(np.searchsorted(cumvar, variance_threshold) + 1)
    n_var = min(n_var, n_fit)

    if n_method == 0 or n_method == n_fit:
        n_components = n_var
        used_method  = (
            f"variance threshold ({variance_threshold*100:.0f}%) — "
            f"{component_selection} gave degenerate {n_method}/{n_fit}"
        )
    else:
        n_components = n_method
        used_method  = component_selection

    cum_kept = float(cumvar[n_components - 1]) * 100
    print(f"  {tag}n_components = {n_components}  ({cum_kept:.1f}% variance)  "
          f"[method: {used_method}]")
    print(f"  {tag}{method_desc}")

    pca = PCA(n_components=n_components, random_state=0)
    pca.fit(W)
    return pca, n_components, evr, threshold


def fit_scaler(Z: np.ndarray) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(Z)
    return scaler


# ── Centroids in PCA space ─────────────────────────────────────────────────────

def compute_centroids(
    Z_scaled:   np.ndarray,
    window_ids: list,
    seed_ids:   list,
    n_windows:  int,
    n_seeds:    int,
) -> tuple:
    """
    Average z-scored PCA vectors across seeds, per window.

    Returns
    -------
    C             : (n_valid_windows, n_components)  centroid matrix
    centroid_wins : list of window indices with ≥ 1 seed
    """
    n_components = Z_scaled.shape[1]
    accum  = np.zeros((n_windows, n_components), dtype=np.float64)
    counts = np.zeros(n_windows, dtype=int)

    for i, (w, _) in enumerate(zip(window_ids, seed_ids)):
        accum[w]  += Z_scaled[i]
        counts[w] += 1

    incomplete = [(w, counts[w]) for w in range(n_windows)
                  if 0 < counts[w] < n_seeds]
    if incomplete:
        print(f"  WARNING: {len(incomplete)} window(s) have < {n_seeds} seeds:")
        for w, cnt in incomplete:
            print(f"    window_{w:03d}: {cnt}/{n_seeds} seeds")

    valid = [w for w in range(n_windows) if counts[w] > 0]
    C     = (accum[valid] / counts[valid, None]).astype(np.float32)
    return C, valid


# ── Sanity-check plots ─────────────────────────────────────────────────────────

# One marker shape per seed (up to 5 seeds)
SEED_MARKERS = ['o', 's', '^', 'D', 'P']   # circle, square, triangle, diamond, plus-star


def scatter_plot(
    Z:          np.ndarray,
    window_ids: list,
    seed_ids:   list,
    n_windows:  int,
    title:      str,
    out_path:   str,
) -> None:
    """
    2-D scatter of PC1 vs PC2.

    Color  — window index, sampled evenly from a continuous colormap so all
             n_windows colors are distinct (no palette recycling).
    Marker — seed index (circle, square, triangle, diamond, plus-star).
    """
    color_values  = np.linspace(0.05, 0.95, n_windows)
    cmap_cont     = cm.get_cmap("nipy_spectral")
    window_colors = {w: cmap_cont(color_values[w]) for w in range(n_windows)}

    fig, ax = plt.subplots(figsize=(11, 7))

    for i, (w, s) in enumerate(zip(window_ids, seed_ids)):
        ax.scatter(Z[i, 0], Z[i, 1],
                   color=window_colors[w],
                   marker=SEED_MARKERS[s % len(SEED_MARKERS)],
                   s=45, alpha=0.80, linewidths=0.3,
                   edgecolors="none", zorder=2)

    sm = plt.cm.ScalarMappable(
             cmap=plt.matplotlib.colors.ListedColormap(
                 [window_colors[w] for w in range(n_windows)]),
             norm=plt.Normalize(vmin=-0.5, vmax=n_windows - 0.5))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02,
                        ticks=np.arange(0, n_windows, max(1, n_windows // 10)))
    cbar.set_label("Window index", fontsize=9)

    unique_seeds = sorted(set(seed_ids))
    seed_handles = [
        plt.Line2D([0], [0],
                   marker=SEED_MARKERS[s % len(SEED_MARKERS)],
                   color="0.35", linestyle="none",
                   markersize=7, label=f"seed {s}")
        for s in unique_seeds
    ]
    ax.legend(handles=seed_handles, fontsize=8, loc="upper left",
              title="Seed", title_fontsize=8, framealpha=0.7)

    ax.set_xlabel("PC 1 (z-scored)", fontsize=11)
    ax.set_ylabel("PC 2 (z-scored)", fontsize=11)
    ax.set_title(title, fontsize=12)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Plot saved → {out_path}")


def plot_component_selection(
    evr:       np.ndarray,
    threshold: np.ndarray,
    n_kept:    int,
    method:    str,
    out_path:  str,
    max_show:  int = 60,
) -> None:
    """
    Two-panel plot (linear + log): observed variance vs per-component threshold.
    Works for both broken-stick and Auer-Gervini baselines.
    Vertical dashed line at n_kept; shaded region shows retained components.
    """
    n_show  = min(len(evr), max_show)
    ranks   = np.arange(1, n_show + 1)
    method_label = {
        "broken_stick": "Broken-stick expectation %",
        "auer_gervini": "Uniform baseline % (Auer-Gervini)",
    }.get(method, "Threshold %")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        f"PCA Component Selection — {method.replace('_', ' ').title()} Analysis",
        fontsize=13, fontweight="bold",
    )

    for ax, yscale in zip(axes, ["linear", "log"]):
        ax.plot(ranks, evr[:n_show] * 100, "o-", color="#2563EB",
                linewidth=1.8, markersize=4, label="Observed variance %")
        ax.plot(ranks, threshold[:n_show] * 100, "s--", color="#DC2626",
                linewidth=1.5, markersize=4, label=method_label)
        ax.axvline(n_kept, color="#16A34A", linestyle="--", linewidth=1.5,
                   label=f"Cutoff: PC {n_kept}")
        kept = min(n_kept, n_show)
        ax.fill_between(
            ranks[:kept],
            evr[:kept] * 100, threshold[:kept] * 100,
            where=(evr[:kept] > threshold[:kept]),
            alpha=0.12, color="#2563EB", label="Retained region",
        )
        ax.set_xlabel("PC rank", fontsize=11)
        ax.set_ylabel("Variance explained (%)", fontsize=11)
        ax.set_title(f"{'Linear' if yscale == 'linear' else 'Log'} scale",
                     fontsize=11)
        ax.set_yscale(yscale)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, n_show + 1)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved -> {out_path}")


def make_plots(
    Z_before:   np.ndarray,
    Z_after:    np.ndarray,
    window_ids: list,
    seed_ids:   list,
    output_dir: str,
    n_windows:  int,
    method:     str = "broken_stick",
) -> None:
    method_tag = method.replace("_", " ").title()
    scatter_plot(
        Z_before, window_ids, seed_ids, n_windows,
        f"BEFORE alignment: MLP weights in PCA space\n"
        f"(dominant clustering is by seed — permutation symmetry)"
        f"  [{method_tag}]",
        os.path.join(output_dir, "sanity_check_before.png"),
    )
    scatter_plot(
        Z_after, window_ids, seed_ids, n_windows,
        f"AFTER alignment: MLP weights in PCA space\n"
        f"(seeds within each window should now cluster tightly)"
        f"  [{method_tag}]",
        os.path.join(output_dir, "sanity_check_after.png"),
    )


# ── Main ───────────────────────────────────────────────────────────────────────

def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    # ── 1. Load checkpoints ──────────────────────────────────────────────
    print(f"\n── Step 1: Loading checkpoints from {args.runs_dir} ──")
    weights_by_window, window_ids, seed_ids, missing = collect_weights(
        args.runs_dir, args.n_windows, args.n_seeds, dry_run=args.dry_run
    )

    n_found    = len(window_ids)
    n_expected = args.n_windows * args.n_seeds
    print(f"  Found : {n_found} / {n_expected} checkpoints")

    if missing:
        print(f"  Missing ({len(missing)}):")
        for p in missing:
            print(f"    {p}")

    if n_found == 0:
        print("ERROR: No checkpoints found. Check --runs_dir.")
        sys.exit(1)

    if args.dry_run:
        print("\nDry run complete. Exiting.")
        return

    # ── 2. Flatten unaligned weights (for before-plot only) ──────────────
    print(f"\n── Step 2: Flattening unaligned weights (for comparison plot) ──")
    W_before_rows = []
    for w in range(args.n_windows):
        for s in sorted(weights_by_window[w].keys()):
            W_before_rows.append(flatten_weights(weights_by_window[w][s]))
    W_before = np.stack(W_before_rows, axis=0).astype(np.float32)
    print(f"  Shape: {W_before.shape}  ({W_before.nbytes / 1e6:.0f} MB)")

    # ── 3. Iterative weight alignment per window ──────────────────────────
    print(f"\n── Step 3: Iterative alignment (no bootstrap reference) ──")
    aligned_by_window = {}
    for w in range(args.n_windows):
        sw = weights_by_window[w]
        if not sw:
            continue
        aligned_by_window[w] = align_window(sw)
        n_s      = len(sw)
        seed_str = ", ".join(str(s) for s in sorted(sw.keys()))
        print(f"  window_{w:03d}: {n_s} seed(s) [{seed_str}] aligned")

    # ── 4. Flatten aligned weights ────────────────────────────────────────
    print(f"\n── Step 4: Flattening aligned weights ──")
    W_after_rows = []
    for w in range(args.n_windows):
        if w not in aligned_by_window:
            continue
        for s in sorted(aligned_by_window[w].keys()):
            W_after_rows.append(flatten_weights(aligned_by_window[w][s]))
    W_after = np.stack(W_after_rows, axis=0).astype(np.float32)
    print(f"  Shape: {W_after.shape}")

    # ── 5. PCA — separate fits for before-plot and final use ──────────────
    print(f"\n── Step 5: Fitting PCA ──")

    # Before-plot: own PCA on unaligned weights so seed clustering is visible
    pca_before, _, _, _ = fit_pca(
        W_before, args.variance_threshold,
        label="unaligned", component_selection=args.component_selection,
    )
    Z_before_2d = pca_before.transform(W_before).astype(np.float32)

    # Final PCA: fit on aligned weights — saved and used for HMM
    pca, n_components, evr_full, threshold = fit_pca(
        W_after, args.variance_threshold,
        label="aligned", component_selection=args.component_selection,
    )
    Z_after = pca.transform(W_after).astype(np.float32)
    print(f"  Aligned projected shape: {Z_after.shape}")

    # ── 6. Z-score independently for before-plot and after ────────────────
    print(f"\n── Step 6: Z-scoring ──")

    scaler_before   = fit_scaler(Z_before_2d)
    Z_before_scaled = scaler_before.transform(Z_before_2d).astype(np.float32)

    scaler         = fit_scaler(Z_after)
    Z_after_scaled = scaler.transform(Z_after).astype(np.float32)
    print(f"  Z_after  mean≈{Z_after_scaled.mean():.4f}  "
          f"std≈{Z_after_scaled.std():.4f}  (should be ≈0, ≈1)")

    # ── 7. Centroids in aligned PCA space ────────────────────────────────
    print(f"\n── Step 7: Computing per-window centroids ──")
    C, centroid_wins = compute_centroids(
        Z_after_scaled, window_ids, seed_ids, args.n_windows, args.n_seeds
    )
    print(f"  Centroid matrix: {C.shape}  ({len(centroid_wins)} windows)")

    # ── 8. Save outputs ───────────────────────────────────────────────────
    print(f"\n── Step 8: Saving outputs → {args.output_dir} ──")

    npz_path = os.path.join(args.output_dir, "weights_pca.npz")
    np.savez(
        npz_path,
        Z_scaled                 = Z_after_scaled,
        C                        = C,
        window_ids               = np.array(window_ids,    dtype=np.int32),
        seed_ids                 = np.array(seed_ids,      dtype=np.int32),
        centroid_wins            = np.array(centroid_wins, dtype=np.int32),
        explained_variance_ratio = pca.explained_variance_ratio_,
        evr_full                 = evr_full,
        component_threshold      = threshold,
    )
    print(f"  weights_pca.npz  → {npz_path}")

    pca_path = os.path.join(args.output_dir, "pca_model.pkl")
    with open(pca_path, "wb") as f:
        pickle.dump(pca, f)
    print(f"  pca_model.pkl    → {pca_path}")

    scaler_path = os.path.join(args.output_dir, "scaler_model.pkl")
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    print(f"  scaler_model.pkl → {scaler_path}")

    # ── 9. Sanity-check plots ────────────────────────────────────────────
    print(f"\n── Step 9: Generating sanity-check plots ──")
    make_plots(
        Z_before_scaled, Z_after_scaled,
        window_ids, seed_ids,
        args.output_dir, args.n_windows,
        method = args.component_selection,
    )
    plot_component_selection(
        evr_full, threshold, n_components,
        method   = args.component_selection,
        out_path = os.path.join(args.output_dir, f"{args.component_selection}.png"),
    )

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Done.")
    print(f"  Checkpoints loaded    : {n_found} / {n_expected}")
    print(f"  Component selection   : {args.component_selection}")
    print(f"  PCA components        : {n_components}  "
          f"({pca.explained_variance_ratio_.cumsum()[-1]*100:.1f}% variance)")
    print(f"  Centroid matrix       : {C.shape}  → ready for HMM")
    print(f"  Outputs               : {args.output_dir}/")
    if missing:
        print(f"\n  WARNING: {len(missing)} missing checkpoints (listed above).")
    print(f"{'='*60}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Extract, align, and PCA-reduce MLP weights for HMM fitting."
    )
    p.add_argument("--runs_dir",    required=True,
                   help="Root of trained window runs, e.g. runs/hmm_windows")
    p.add_argument("--output_dir",  default="data/hmm_weights",
                   help="Where to write outputs (created if absent).")
    p.add_argument("--n_windows",   type=int, default=35)
    p.add_argument("--n_seeds",     type=int, default=5)
    p.add_argument("--variance_threshold", type=float, default=0.90,
                   help="Fallback cumulative PCA variance threshold if the "
                        "primary selection method gives a degenerate result.")
    p.add_argument(
        "--component_selection",
        choices=["broken_stick", "auer_gervini"],
        default="broken_stick",
        help=(
            "Method for selecting the number of PCA components. "
            "'broken_stick' (default): retain leading PCs that exceed the "
            "broken-stick expectation (conservative, contiguous-prefix rule). "
            "'auer_gervini': retain all PCs that exceed the uniform variance "
            "baseline 1/n — less conservative, allows gaps in the prefix."
        ),
    )
    p.add_argument("--dry_run",     action="store_true",
                   help="Check checkpoint paths only; skip loading and processing.")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print("=" * 60)
    print(f"runs_dir             : {args.runs_dir}")
    print(f"output_dir           : {args.output_dir}")
    print(f"n_windows            : {args.n_windows}")
    print(f"n_seeds              : {args.n_seeds}")
    print(f"variance_threshold   : {args.variance_threshold}")
    print(f"component_selection  : {args.component_selection}")
    print(f"dry_run              : {args.dry_run}")
    print("=" * 60)
    main(args)