"""
src/hmm_on_class_dist.py

Baseline (b): a segmentation derived from the class distribution directly.

Instead of fitting the HMM to the per-window weight centroids, we fit the SAME
kind of Gaussian HMM to the per-window 6-way class-distribution sequence. The
emission dimension is 6 (the class proportions) rather than the weight-PCA
dimension. Everything else mirrors src/fit_hmm_decode.py: per-dimension
z-scoring (HMMs are scale-sensitive), diagonal-covariance Gaussian emissions,
multiple random initialisations with the best training log-likelihood kept, and
a Viterbi decode.

This isolates whether the segmentation needs the model weights at all, or
whether the class distribution alone — which the weight-space states are already
known to track (rho_JSD = -0.824 in the paper) — reproduces the partition.

To make the comparison a clean head-to-head, the number of states defaults to
the SAME k as the weight HMM. Pass --match_decode <weight_hmm_decode>.npz to
copy k (and the window ordering) directly from the HMM you are comparing
against, or set --k explicitly.

Output: final_decode_classdist_k{k}.npz — drop-in for within_across_states.py,
check_equal_windows.py and state_pair_correlation.py. NOTE: for this baseline
the saved `means` are the windows' weight-PCA centroids per state (so the
weight-space correlation still runs), but the SEGMENTATION itself is from the
class distribution.

Usage
-----
  python src/hmm_on_class_dist.py \
      --weights_npz  data/hmm_weights/6way/weights_pca.npz \
      --manifest     data/splits/hmm_windows/HMM_windows_manifest.csv \
      --output_dir   data/baselines/classdist \
      --match_decode data/hmm_hmm/6way/final_decode_k7.npz \
      --n_inits 50 --n_iter 200 --seed 42

  # or set k directly
  python src/hmm_on_class_dist.py --k 7

Requirements: hmmlearn, numpy, pandas, matplotlib
"""

import os
import argparse
import warnings
import logging

import numpy as np
from hmmlearn import hmm

from segmentation_common import (
    load_weights_pca, build_centroid_sequence, load_class_distribution,
    load_window_dates, relabel_by_first_appearance, save_decode,
    print_segmentation, plot_timeline, CLASS_NAMES,
)

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", message=".*did not converge.*")
warnings.filterwarnings("ignore", category=DeprecationWarning)
# hmmlearn emits convergence / zero-transition notices through the logging
# module (not warnings); quiet it so the run output stays readable.
logging.getLogger("hmmlearn").setLevel(logging.ERROR)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights_npz", default="data/hmm_weights/weights_pca.npz",
                   help="weights_pca.npz (for the chronological window order and "
                        "the PCA centroids saved into the decode file)")
    p.add_argument("--manifest",    default="data/splits/hmm_windows/HMM_windows_manifest.csv")
    p.add_argument("--output_dir",  default="data/baselines/classdist")
    p.add_argument("--match_decode", default=None,
                   help="Optional HMM decode .npz to copy k from (head-to-head)")
    p.add_argument("--k",           type=int, default=7,
                   help="Number of states (ignored if --match_decode is given)")
    p.add_argument("--n_inits",     type=int, default=50)
    p.add_argument("--n_iter",      type=int, default=200)
    p.add_argument("--seed",        type=int, default=42)
    return p.parse_args()


def zscore(X):
    """Per-column z-score (mean 0, std 1). Matches the HMM pipeline's scaling."""
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True)
    sd[sd == 0] = 1.0
    return (X - mu) / sd


def fit_best_hmm(X, k, n_inits, n_iter):
    """Fit n_inits diagonal-covariance Gaussian HMMs; keep the best-LL model."""
    lengths = [X.shape[0]]
    best_model, best_ll = None, -np.inf
    first_exc = None
    for init in range(n_inits):
        model = hmm.GaussianHMM(
            n_components=k, covariance_type="diag",
            n_iter=n_iter, tol=1e-5, random_state=init * 100 + k, verbose=False,
        )
        try:
            model.fit(X, lengths)
            ll = model.score(X, lengths)
        except Exception as e:
            first_exc = first_exc or e
            continue
        if ll > best_ll:
            best_ll, best_model = ll, model
    if best_model is None:
        raise RuntimeError(f"All {n_inits} HMM inits failed. First: {first_exc}")
    return best_model, best_ll


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # k: copy from the HMM decode we are comparing against, if provided
    k = args.k
    if args.match_decode:
        dec = np.load(args.match_decode, allow_pickle=True)
        k = int(dec.get("k", dec["state_seq"].max() + 1))
        print(f"Matched k={k} from {args.match_decode}")

    print("=" * 64)
    print(f"Baseline (b): HMM on class distribution  (k={k})")
    print("=" * 64)

    # Chronological window order + weight centroids (for the saved decode)
    _, Z_scaled, window_ids, seed_ids = load_weights_pca(args.weights_npz)
    centroid_seq, sorted_window_ids = build_centroid_sequence(
        Z_scaled, window_ids, seed_ids)
    N = len(sorted_window_ids)

    # Per-window 6-way class proportions, aligned to the same window order
    props, counts = load_class_distribution(args.manifest, sorted_window_ids)
    print(f"\nLoaded {N} windows; class-distribution emission dim = {props.shape[1]}")

    # z-score the 6 class-proportion features, then fit + decode
    X = zscore(props)
    print(f"\nFitting GaussianHMM(k={k}, cov=diag) with {args.n_inits} inits "
          f"on the class-distribution sequence ...")
    model, ll = fit_best_hmm(X, k, args.n_inits, args.n_iter)
    print(f"  Best training log-likelihood: {ll:.4f}")

    raw_states = np.asarray(model.predict(X, [N]), dtype=int)
    state_seq  = relabel_by_first_appearance(raw_states)
    k_eff      = int(len(np.unique(state_seq)))
    if k_eff < k:
        print(f"  [note] HMM used only {k_eff} of {k} states on this sequence.")

    print_segmentation(state_seq, sorted_window_ids, k_eff,
                       "Class-distribution HMM segmentation:")

    # Per-state mean class distribution (interpretability)
    print("\n  Per-state mean class proportions:")
    hdr = "    " + "".join(f"{CLASS_NAMES[c][:10]:>12}" for c in range(6))
    print(hdr)
    for s in range(k_eff):
        m = props[state_seq == s].mean(axis=0)
        print("    " + "".join(f"{v:>11.1%} " for v in m))

    out = os.path.join(args.output_dir, f"final_decode_classdist_k{k_eff}.npz")
    save_decode(out, state_seq=state_seq, window_ids=sorted_window_ids, k=k_eff,
                centroid_seq=centroid_seq, method="hmm_class_dist",
                extra=dict(class_props=props,
                           transition_matrix=model.transmat_,
                           log_likelihood=np.float64(ll),
                           emission_means=model.means_))
    print(f"\nSaved decode: {out}")

    dates = load_window_dates(args.manifest, sorted_window_ids)
    plot_timeline(state_seq, sorted_window_ids, k_eff,
                  f"Class-distribution HMM segmentation (k={k_eff})",
                  os.path.join(args.output_dir, f"timeline_classdist_k{k_eff}.png"),
                  dates=dates)

    print("\nNext, evaluate it with the SAME machinery as the HMM:")
    print(f"  python src/within_across_states.py --decode_npz {out} \\")
    print(f"      --f1_npz data/hmm_perf/6way/cross_window_f1.npz \\")
    print(f"      --output_dir data/baselines/classdist/within_across")


if __name__ == "__main__":
    main()