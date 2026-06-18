"""
src/partial_jsd_transfer.py

Reviewer-requested control: does the within-vs-across-state advantage in
cross-window transfer survive after class-distribution divergence (JSD) is
partialled out, alongside temporal lag?

Motivation
----------
Two separate results in the paper could, in principle, be the *same* effect:

  (1) within_across_states.py  — within-state pairs transfer better than
      across-state pairs, even after temporal lag is controlled for.
  (2) state_pair_correlation.py — transfer falls as the Jensen-Shannon
      divergence between two states' 6-way class distributions rises.

The concern: maybe "sharing a state" predicts transfer *only* because windows
in the same state happen to have similar class distributions. If so, once we
hold JSD (and lag) fixed, the within/across indicator should add nothing.
This script tests exactly that, at the level the reviewer asked for — the
individual window pair.

Key design choice: JSD is computed PER WINDOW PAIR
--------------------------------------------------
In state_pair_correlation.py, JSD is computed between two *states'* aggregated
class distributions. If we reused that here, every within-state pair would get
JSD = JSD(state_s, state_s) = 0, making JSD a deterministic function of the
within/across indicator (perfect collinearity) — the regression could not
separate them. So we instead compute JSD between each *window's own* 6-way
class distribution (from the manifest cls_* counts). This is a continuous,
state-agnostic dissimilarity that varies both within and across states, so it
can genuinely compete with the indicator for explanatory power. This is also a
strictly stronger control: it uses the finest-grained class-distribution
information available, not the state-averaged version.

Two analyses, matching the reviewer's two suggested framings
------------------------------------------------------------
  A. Combined model (primary):
        f1  ~  beta0 + beta_jsd * JSD + beta_lag * lag + beta_state * same_state
     We report beta_state (the within-state advantage in macro-F1 units, holding
     JSD and lag fixed) and test it with a Freedman-Lane partial permutation
     test that is stratified by lag — the natural extension of the paper's
     existing distance-conditioned shuffle. Because window pairs are not
     independent (each window appears in many pairs; the two transfer
     directions of an unordered pair share covariates), we ALSO report
     cluster-robust standard errors clustered on the unordered window pair.

  B. Two-stage residual model (the reviewer's explicit alternative):
        step 1:  regress f1 ~ JSD + lag, take residuals
        step 2:  test same_state on the residuals
     We test step 2 two ways: (i) a simple within-minus-across mean residual
     gap with a label-shuffle null, and (ii) the paper's own harmonic-weighted,
     lag-stratified distance-conditioned gap statistic applied to the
     residualized F1 (so the inference machinery is identical to Figure 7,
     just run on JSD-residualized F1).

If beta_state stays positive and significant after partialling JSD and lag, the
latent-state structure carries transfer-relevant information *beyond* raw
class-distribution shift and temporal proximity.

Inputs
------
  --decode_npz   data/hmm_hmm/6way/final_decode_k7.npz   (state_seq, window_ids)
  --f1_npz       data/hmm_perf/6way/cross_window_f1.npz   (f1_matrix [, cube])
  --manifest     data/splits/hmm_windows/HMM_windows_manifest.csv  (cls_*)

Outputs  (--output_dir)
-----------------------
  window_pair_table.csv        one row per ordered (i, j) pair: f1, jsd, lag,
                               same_state, state_i, state_j
  partial_jsd_results.csv      coefficient table + permutation p-values
  partial_jsd_null.npz         null distributions for the indicator effect
  plot_partial_residual.png    JSD+lag-residualized F1, within vs across
  plot_indicator_null.png      Freedman-Lane null for beta_state

Usage
-----
  python src/partial_jsd_transfer.py \
      --decode_npz  data/hmm_hmm/6way/final_decode_k7.npz \
      --f1_npz      data/hmm_perf/6way/cross_window_f1.npz \
      --manifest    data/splits/hmm_windows/HMM_windows_manifest.csv \
      --output_dir  data/partial_jsd/6way/k7 \
      --n_permutations 10000 \
      --seed 42

  # Use one row per (train, test, seed) instead of the per-window mean
  # (more rows but heavily correlated; inference uses pair-clustered SEs /
  #  pair-respecting permutation either way):
  #   ... --use_seeds

Requirements: numpy, pandas, scipy, matplotlib, tqdm
              statsmodels (optional — only for the cluster-robust SE table;
              the permutation inference runs without it)
"""

import os
import argparse
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

warnings.filterwarnings("ignore", category=RuntimeWarning)


# ── Label maps (match characterize_states.py) ─────────────────────────────────

CLASS_NAMES = {
    0: "True",
    1: "Satire",
    2: "False Connection",
    3: "Imposter Content",
    4: "Manipulated Content",
    5: "Misleading Content",
}
CLS_COLS = [f"cls_{i}" for i in range(6)]


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--decode_npz", default="data/hmm_hmm/6way/final_decode_k7.npz",
                   help="HMM decode file (state_seq, window_ids) from fit_hmm_decode.py")
    p.add_argument("--f1_npz", default="data/hmm_perf/6way/cross_window_f1.npz",
                   help="Cross-window F1 matrix (and optional per-seed cube)")
    p.add_argument("--manifest",
                   default="data/splits/hmm_windows/HMM_windows_manifest.csv",
                   help="Window manifest with cls_0..cls_5 columns")
    p.add_argument("--output_dir", default="data/partial_jsd/6way/k7")
    p.add_argument("--n_permutations", type=int, default=10_000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--use_seeds", action="store_true",
                   help="Use one row per (train,test,seed) instead of per-window-mean F1")
    p.add_argument("--no_plots", action="store_true")
    return p.parse_args()


# ── I/O ─────────────────────────────────────────────────────────────────────

def load_inputs(decode_npz, f1_npz, manifest_path):
    dec        = np.load(decode_npz, allow_pickle=True)
    state_seq  = dec["state_seq"].astype(int)    # (N,)
    window_ids = dec["window_ids"].astype(int)   # (N,)
    k          = int(dec.get("k", state_seq.max() + 1))

    f1_data   = np.load(f1_npz, allow_pickle=True)
    f1_matrix = f1_data["f1_matrix"].astype(float)               # (N, N)
    f1_cube   = (f1_data["f1_per_seed_cube"].astype(float)
                 if "f1_per_seed_cube" in f1_data else None)     # (N, n_seeds, N)

    N = len(state_seq)
    assert f1_matrix.shape == (N, N), (
        f"state_seq length {N} != f1_matrix shape {f1_matrix.shape}")

    # Align manifest rows to the decode window order, fill missing class cols.
    manifest = pd.read_csv(manifest_path, parse_dates=["start_date", "end_date"])
    manifest = manifest.set_index("window_local_idx").loc[window_ids].reset_index()
    for c in CLS_COLS:
        if c not in manifest.columns:
            manifest[c] = 0
        else:
            manifest[c] = manifest[c].fillna(0).astype(int)

    n_seeds = f1_cube.shape[1] if f1_cube is not None else "n/a"
    print(f"Loaded: N={N} windows, k={k} states, n_seeds={n_seeds}")
    print(f"State sequence: {state_seq}")
    return state_seq, window_ids, k, f1_matrix, f1_cube, manifest


# ── Per-window class proportions ──────────────────────────────────────────────

def window_class_props(manifest: pd.DataFrame) -> np.ndarray:
    """
    Returns (N, 6) array of per-window class proportions.

    NOTE: this is per *window*, not per state — the whole point of the
    window-pair-level JSD (see module docstring).
    """
    counts = manifest[CLS_COLS].values.astype(float)        # (N, 6)
    totals = counts.sum(axis=1, keepdims=True)
    return counts / np.where(totals > 0, totals, 1.0)


def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """
    Jensen-Shannon divergence (base-2) between two discrete distributions.
    Identical definition to state_pair_correlation.py.
    """
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)

    def kl(a, b):
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / b[mask])))

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


# ── Build window-pair table ───────────────────────────────────────────────────

def build_window_pair_table(state_seq: np.ndarray,
                            f1_matrix: np.ndarray,
                            f1_cube,
                            class_props: np.ndarray,
                            use_seeds: bool) -> pd.DataFrame:
    """
    One row per ordered off-diagonal window pair (i, j), i != j.

    Columns
    -------
      train_win, test_win   window indices (0..N-1)
      seed                  seed index, or -1 for the per-window mean
      lag                   |i - j|  (temporal distance)
      same_state            1 if windows i and j share an HMM state, else 0
      state_i, state_j      decoded states of the two windows
      jsd                   JSD between the two windows' OWN 6-way class
                            distributions (window-level, not state-level)
      pair_id               integer id of the unordered pair {i, j}, used as
                            the clustering / permutation-blocking unit
      f1                    macro-F1 of the i -> j transfer
    """
    N = len(state_seq)
    rows = []

    # Stable integer id for each unordered window pair {i, j}.
    def pair_id(i, j):
        a, b = (i, j) if i < j else (j, i)
        return a * N + b

    use_cube = use_seeds and (f1_cube is not None)
    if use_seeds and f1_cube is None:
        print("  [warn] --use_seeds set but no per-seed cube present; "
              "falling back to per-window-mean F1.")

    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            base = dict(
                train_win  = i,
                test_win   = j,
                lag        = abs(i - j),
                same_state = int(state_seq[i] == state_seq[j]),
                state_i    = int(state_seq[i]),
                state_j    = int(state_seq[j]),
                jsd        = js_divergence(class_props[i], class_props[j]),
                pair_id    = pair_id(i, j),
            )
            if use_cube:
                n_seeds = f1_cube.shape[1]
                for s in range(n_seeds):
                    v = f1_cube[i, s, j]
                    if np.isnan(v):
                        continue
                    rows.append({**base, "seed": s, "f1": float(v)})
            else:
                v = f1_matrix[i, j]
                if np.isnan(v):
                    continue
                rows.append({**base, "seed": -1, "f1": float(v)})

    df = pd.DataFrame(rows)
    df["jsd"] = df["jsd"].round(6)
    return df


# ── Plain numpy OLS helpers (used inside permutation loops) ───────────────────

def design_matrix(df: pd.DataFrame, cols) -> np.ndarray:
    """Build [1, col1, col2, ...] design matrix from the named columns."""
    X = np.column_stack([np.ones(len(df))] + [df[c].values.astype(float)
                                              for c in cols])
    return X


def ols_beta(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """OLS coefficient vector via least squares."""
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return beta


# ── A. Combined model + Freedman-Lane partial permutation ─────────────────────

def freedman_lane_indicator_test(df: pd.DataFrame,
                                 n_perm: int,
                                 rng: np.random.Generator):
    """
    Partial test of the within/across indicator in

        f1 ~ JSD + lag + same_state

    via the Freedman-Lane scheme, with residual permutation stratified by lag.

    Procedure
    ---------
      1. Observed statistic: fit the FULL model, record the same_state coef.
      2. Reduced model: fit f1 ~ JSD + lag; take fitted values + residuals.
      3. For each permutation, shuffle the reduced-model residuals WITHIN each
         lag stratum (so the lag structure and the paper's temporal-proximity
         control are preserved), rebuild a synthetic response
             y* = fitted_reduced + residual_permuted,
         refit the FULL model, and record the same_state coefficient.
      4. p (one-sided, within-state advantage) = fraction of null coefs
         >= observed; a two-sided p is also returned.

    Stratifying the residual permutation by lag matters here because the HMM
    states are contiguous temporal runs, so same_state==1 only occurs at small
    lags. Within-lag permutation makes the null compare within- and
    across-state pairs *at the same lag*, exactly as Figure 7 does.

    Returns dict with observed coef, full-model coefs, null array, p-values.
    """
    y = df["f1"].values.astype(float)

    X_full = design_matrix(df, ["jsd", "lag", "same_state"])
    beta_full = ols_beta(X_full, y)
    # Column order: [intercept, jsd, lag, same_state]
    obs_coef = float(beta_full[3])

    X_red = design_matrix(df, ["jsd", "lag"])
    beta_red = ols_beta(X_red, y)
    fitted_red = X_red @ beta_red
    resid_red = y - fitted_red

    # Pre-index residual positions by lag stratum.
    lag_vals = df["lag"].values
    strata = {d: np.where(lag_vals == d)[0] for d in np.unique(lag_vals)}

    null_coefs = np.empty(n_perm)
    for b in tqdm(range(n_perm), desc="Freedman-Lane (indicator)"):
        resid_perm = resid_red.copy()
        for idx in strata.values():
            if idx.size > 1:
                resid_perm[idx] = rng.permutation(resid_red[idx])
        y_star = fitted_red + resid_perm
        beta_star = ols_beta(X_full, y_star)
        null_coefs[b] = beta_star[3]

    p_one = float((null_coefs >= obs_coef).mean())             # within advantage
    p_two = float((np.abs(null_coefs) >= abs(obs_coef)).mean())

    return {
        "obs_coef":   obs_coef,
        "beta_full":  beta_full,        # [intercept, jsd, lag, same_state]
        "null_coefs": null_coefs,
        "p_one":      p_one,
        "p_two":      p_two,
    }


def cluster_robust_table(df: pd.DataFrame):
    """
    Fit  f1 ~ JSD + lag + same_state  with standard errors clustered on the
    unordered window pair, using statsmodels if available.

    Clustering on pair_id is the relevant correction because the two transfer
    directions (i->j and j->i), and — under --use_seeds — all per-seed rows of
    a given (i, j) cell, are not independent. Returns a tidy DataFrame, or None
    if statsmodels is not installed (the permutation test is the headline
    inference regardless).
    """
    try:
        import statsmodels.api as sm
    except Exception as e:                       # pragma: no cover
        print(f"  [info] statsmodels unavailable ({e}); "
              f"skipping cluster-robust SE table.")
        return None

    X = sm.add_constant(df[["jsd", "lag", "same_state"]].astype(float))
    y = df["f1"].astype(float)
    model = sm.OLS(y, X).fit(
        cov_type="cluster",
        cov_kwds={"groups": df["pair_id"].values},
    )
    out = pd.DataFrame({
        "term":     ["intercept", "jsd", "lag", "same_state"],
        "coef":     model.params.values,
        "se_clustered": model.bse.values,
        "t":        model.tvalues.values,
        "p_param":  model.pvalues.values,
    })
    return out


# ── B. Two-stage residual model (reviewer's explicit alternative) ─────────────

def residualize_on_jsd_and_lag(df: pd.DataFrame) -> np.ndarray:
    """Return F1 residuals after regressing f1 ~ JSD + lag (step 1)."""
    y = df["f1"].values.astype(float)
    X_red = design_matrix(df, ["jsd", "lag"])
    beta_red = ols_beta(X_red, y)
    return y - X_red @ beta_red


def simple_residual_gap_test(resid: np.ndarray,
                            same_state: np.ndarray,
                            n_perm: int,
                            rng: np.random.Generator):
    """
    Step 2, simple form: within-minus-across mean of the JSD+lag residuals,
    with a plain label-shuffle null on the same_state indicator.

    Because lag was already removed in step 1, a free (non-stratified) shuffle
    is defensible here; the lag-stratified variant below is the conservative
    belt-and-suspenders version.
    """
    w = resid[same_state == 1]
    a = resid[same_state == 0]
    obs_gap = float(w.mean() - a.mean())

    n_within = int(same_state.sum())
    null = np.empty(n_perm)
    for b in tqdm(range(n_perm), desc="Residual gap (free shuffle)"):
        perm = rng.permutation(same_state)
        null[b] = resid[perm == 1].mean() - resid[perm == 0].mean()

    p_one = float((null >= obs_gap).mean())
    return {"obs_gap": obs_gap, "null": null, "p_one": p_one,
            "n_within": n_within, "n_across": int((same_state == 0).sum())}


def distance_conditioned_gap(value: np.ndarray,
                            same_state: np.ndarray,
                            lag: np.ndarray) -> float:
    """
    The paper's harmonic-count-weighted, lag-stratified within-minus-across
    gap statistic (Eq. 4 in within_across_states.py), here computed on an
    arbitrary `value` array — we pass in the JSD+lag residuals.
    """
    gaps, weights = [], []
    for d in np.unique(lag):
        m = lag == d
        w = value[m & (same_state == 1)]
        a = value[m & (same_state == 0)]
        if w.size == 0 or a.size == 0:
            continue
        gaps.append(w.mean() - a.mean())
        weights.append(2 * w.size * a.size / (w.size + a.size))
    if not gaps:
        return 0.0
    return float(np.average(gaps, weights=np.array(weights)))


def distance_conditioned_residual_test(resid: np.ndarray,
                                       same_state: np.ndarray,
                                       lag: np.ndarray,
                                       n_perm: int,
                                       rng: np.random.Generator):
    """
    Step 2, paper-faithful form: apply the exact distance-conditioned label
    shuffle from within_across_states.py to the JSD-residualized F1.

    Within each lag stratum (that has both within- and across-state pairs),
    shuffle the same_state labels; the statistic is the harmonic-weighted mean
    of per-lag residual gaps. This is identical to the Figure-7 test except the
    response is F1 with the JSD trend removed.
    """
    obs = distance_conditioned_gap(resid, same_state, lag)

    # Precompute per-lag indices and harmonic weights for usable strata.
    strata, weights = {}, {}
    for d in np.unique(lag):
        idx = np.where(lag == d)[0]
        ss = same_state[idx]
        nw, na = int(ss.sum()), int((ss == 0).sum())
        if nw > 0 and na > 0:
            strata[d] = idx
            weights[d] = 2 * nw * na / (nw + na)
    total_w = sum(weights.values())

    null = np.empty(n_perm)
    for b in tqdm(range(n_perm), desc="Residual gap (lag-stratified)"):
        num = 0.0
        for d, idx in strata.items():
            labels = same_state[idx].copy()
            rng.shuffle(labels)
            vals = resid[idx]
            num += weights[d] * (vals[labels == 1].mean() - vals[labels == 0].mean())
        null[b] = num / total_w if total_w else 0.0

    p_one = float((null >= obs).mean())
    return {"obs_gap": obs, "null": null, "p_one": p_one}


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_partial_residual(resid, same_state, output_path):
    """Box/strip of JSD+lag-residualized F1 for within- vs across-state pairs."""
    fig, ax = plt.subplots(figsize=(5, 4.5))
    groups = [resid[same_state == 0], resid[same_state == 1]]
    ax.boxplot(groups, labels=["Across-state", "Within-state"],
               showmeans=True, widths=0.5)
    for i, g in enumerate(groups, start=1):
        x = rng_jitter(len(g), i)
        ax.scatter(x, g, s=6, alpha=0.25,
                   color="#DC2626" if i == 1 else "#2563EB")
    ax.axhline(0.0, color="black", lw=0.8, ls="--", alpha=0.6)
    ax.set_ylabel("F1 residual (after removing JSD + lag)")
    ax.set_title("JSD+lag-residualized transfer F1\nby state membership")
    ax.grid(True, axis="y", alpha=0.3, ls="--")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_path}")


def rng_jitter(n, center, width=0.12, seed=0):
    r = np.random.default_rng(seed + center)
    return center + r.uniform(-width, width, size=n)


def plot_indicator_null(null_coefs, obs_coef, p_one, output_path):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(null_coefs, bins=60, color="#94A3B8", edgecolor="white",
            lw=0.3, alpha=0.85, label="Freedman-Lane null")
    ax.axvline(obs_coef, color="#DC2626", lw=2.5,
               label=f"Observed same_state coef = {obs_coef:.4f}\n(p = {p_one:.4f})")
    ax.axvline(np.percentile(null_coefs, 95), color="black", lw=1.2, ls="--",
               alpha=0.7, label=f"Null 95th pct = {np.percentile(null_coefs,95):.4f}")
    ax.set_xlabel("same_state coefficient  (within − across F1, JSD & lag held fixed)")
    ax.set_ylabel("Count")
    ax.set_title("Partial test of state membership\n(JSD and lag partialled out)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, ls="--")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    print("=" * 64)
    print("Partialling JSD out of cross-window transfer (window-pair level)")
    print("=" * 64)

    state_seq, window_ids, k, f1_matrix, f1_cube, manifest = load_inputs(
        args.decode_npz, args.f1_npz, args.manifest)

    class_props = window_class_props(manifest)

    print("\nBuilding window-pair table ...")
    df = build_window_pair_table(state_seq, f1_matrix, f1_cube,
                                 class_props, args.use_seeds)
    obs_label = "per-seed" if (args.use_seeds and f1_cube is not None) else "per-window-mean"
    print(f"  {len(df):,} ordered off-diagonal observations ({obs_label})")
    print(f"  within-state: {int(df['same_state'].sum()):,}   "
          f"across-state: {int((df['same_state']==0).sum()):,}")
    print(f"  JSD range: [{df['jsd'].min():.4f}, {df['jsd'].max():.4f}]   "
          f"lag range: [{df['lag'].min()}, {df['lag'].max()}]")

    pair_path = os.path.join(args.output_dir, "window_pair_table.csv")
    df.to_csv(pair_path, index=False)
    print(f"  Saved: {pair_path}")

    # ── A. Combined model + Freedman-Lane partial permutation ───────────────
    print(f"\n[A] Combined model  f1 ~ JSD + lag + same_state")
    fl = freedman_lane_indicator_test(df, args.n_permutations, rng)
    b = fl["beta_full"]   # [intercept, jsd, lag, same_state]
    print(f"      intercept   = {b[0]:+.4f}")
    print(f"      beta_JSD    = {b[1]:+.4f}")
    print(f"      beta_lag    = {b[2]:+.4f}")
    print(f"      beta_state  = {b[3]:+.4f}   "
          f"(within-state F1 advantage, JSD & lag held fixed)")
    print(f"      Freedman-Lane p (one-sided, within>across) = {fl['p_one']:.4f}")
    print(f"      Freedman-Lane p (two-sided)                = {fl['p_two']:.4f}")

    crt = cluster_robust_table(df)
    if crt is not None:
        print("\n      Cluster-robust (clustered on unordered window pair):")
        print(crt.to_string(index=False,
                            float_format=lambda v: f"{v:+.4f}"))

    # ── B. Two-stage residual model ─────────────────────────────────────────
    print(f"\n[B] Two-stage: residualize f1 on JSD + lag, then test same_state")
    resid = residualize_on_jsd_and_lag(df)
    same_state = df["same_state"].values.astype(int)
    lag = df["lag"].values.astype(int)

    simple = simple_residual_gap_test(resid, same_state, args.n_permutations, rng)
    print(f"      Residual within−across gap (free shuffle)        = "
          f"{simple['obs_gap']:+.4f}   p = {simple['p_one']:.4f}")

    dcr = distance_conditioned_residual_test(
        resid, same_state, lag, args.n_permutations, rng)
    print(f"      Residual gap (lag-stratified, harmonic-weighted) = "
          f"{dcr['obs_gap']:+.4f}   p = {dcr['p_one']:.4f}")

    # ── Save results table ──────────────────────────────────────────────────
    rows = [
        {"analysis": "combined_model", "term": "beta_JSD",
         "estimate": b[1], "p_one_sided": np.nan, "p_two_sided": np.nan},
        {"analysis": "combined_model", "term": "beta_lag",
         "estimate": b[2], "p_one_sided": np.nan, "p_two_sided": np.nan},
        {"analysis": "combined_model", "term": "beta_same_state",
         "estimate": b[3], "p_one_sided": fl["p_one"], "p_two_sided": fl["p_two"]},
        {"analysis": "two_stage_residual_free", "term": "within_minus_across_gap",
         "estimate": simple["obs_gap"], "p_one_sided": simple["p_one"],
         "p_two_sided": np.nan},
        {"analysis": "two_stage_residual_lagstratified", "term": "distance_conditioned_gap",
         "estimate": dcr["obs_gap"], "p_one_sided": dcr["p_one"],
         "p_two_sided": np.nan},
    ]
    res_df = pd.DataFrame(rows)
    res_path = os.path.join(args.output_dir, "partial_jsd_results.csv")
    res_df.to_csv(res_path, index=False)
    print(f"\n  Saved: {res_path}")
    if crt is not None:
        crt.to_csv(os.path.join(args.output_dir,
                                "partial_jsd_cluster_robust.csv"), index=False)

    np.savez(
        os.path.join(args.output_dir, "partial_jsd_null.npz"),
        fl_null=fl["null_coefs"], fl_obs=fl["obs_coef"],
        fl_p_one=fl["p_one"], fl_p_two=fl["p_two"],
        resid_free_null=simple["null"], resid_free_obs=simple["obs_gap"],
        resid_lag_null=dcr["null"], resid_lag_obs=dcr["obs_gap"],
        beta_full=b,
    )
    print(f"  Saved: {os.path.join(args.output_dir, 'partial_jsd_null.npz')}")

    # ── Plots ───────────────────────────────────────────────────────────────
    if not args.no_plots:
        print("\nGenerating plots ...")
        plot_partial_residual(
            resid, same_state,
            os.path.join(args.output_dir, "plot_partial_residual.png"))
        plot_indicator_null(
            fl["null_coefs"], fl["obs_coef"], fl["p_one"],
            os.path.join(args.output_dir, "plot_indicator_null.png"))

    print("\nDone.")


if __name__ == "__main__":
    main()