"""
src/pelt_on_weights.py

Baseline (c): a contiguity-respecting change-point method on the weight
trajectory — PELT (Pruned Exact Linear Time, Killick et al. 2012) via `ruptures`.

PELT places change points along the SAME per-window centroid sequence the HMM
was fit on, partitioning the timeline into contiguous segments by detecting
shifts in the trajectory (default cost: l2 / mean-shift). Unlike k-means, PELT
respects time order, so it is the apples-to-apples test the reviewer asked for:
among methods that produce contiguous segments, does the HMM's probabilistic
machinery (Gaussian emissions + learned transition matrix + Viterbi) beat a
simple cost-based change-point detector on the same signal?

The segmentation is forced to exactly k segments — the same granularity as the
HMM — for a clean head-to-head. The PELT penalty is bisected so PELT itself
returns k-1 change points; if the breakpoint count steps over k (it is a step
function of the penalty), it falls back to ruptures' exact known-#-changepoints
search (Dynp) and says so.

Pass --match_decode <hmm_decode>.npz to copy k from the HMM you are comparing
against, or set --k directly.

Output: final_decode_pelt_k{k}.npz — drop-in for within_across_states.py,
check_equal_windows.py and state_pair_correlation.py.

Usage
-----
  python src/pelt_on_weights.py \
      --weights_npz  data/hmm_weights/6way/weights_pca.npz \
      --manifest     data/splits/hmm_windows/HMM_windows_manifest.csv \
      --output_dir   data/baselines/pelt \
      --match_decode data/hmm_hmm/6way/final_decode_k7.npz \
      --model l2 --seed 42

  # set k directly
  python src/pelt_on_weights.py --k 7

Requirements: ruptures, numpy, pandas, matplotlib
"""

import os
import argparse
import warnings

import numpy as np
import ruptures as rpt

from segmentation_common import (
    load_weights_pca, build_centroid_sequence, load_window_dates,
    relabel_by_first_appearance, save_decode, print_segmentation,
    plot_timeline,
)

warnings.filterwarnings("ignore", category=RuntimeWarning)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights_npz", default="data/hmm_weights/weights_pca.npz")
    p.add_argument("--manifest",    default="data/splits/hmm_windows/HMM_windows_manifest.csv")
    p.add_argument("--output_dir",  default="data/baselines/pelt")
    p.add_argument("--match_decode", default=None,
                   help="Optional HMM decode .npz to copy k from (head-to-head)")
    p.add_argument("--k",        type=int, default=7,
                   help="Number of segments (ignored if --match_decode given)")
    p.add_argument("--model",    default="l2", choices=["l2", "l1", "rbf", "normal"],
                   help="ruptures cost model (l2 = mean shift, default)")
    p.add_argument("--min_size", type=int, default=1,
                   help="Minimum segment length (windows)")
    p.add_argument("--seed",     type=int, default=42)
    return p.parse_args()


def bkps_to_labels(bkps, N):
    """ruptures returns segment END indices (last == N). Convert to per-sample
    integer segment labels 0,1,2,..."""
    labels = np.empty(N, dtype=int)
    start = 0
    for seg, end in enumerate(bkps):
        labels[start:end] = seg
        start = end
    return labels


def pelt_predict(signal, penalty, model, min_size):
    algo = rpt.Pelt(model=model, min_size=min_size, jump=1).fit(signal)
    return algo.predict(pen=penalty)        # list of end indices, last == N


def find_penalty_for_k(signal, target_segments, model, min_size,
                       lo=1e-4, hi=1e6, iters=80):
    """
    Bisect the penalty (geometrically) to make PELT return exactly
    `target_segments` segments. The number of segments is non-increasing in the
    penalty. Returns (penalty, n_segments, exact_flag).
    """
    n_at = lambda pen: len(pelt_predict(signal, pen, model, min_size))
    best = None
    for _ in range(iters):
        mid = np.sqrt(lo * hi)
        n = n_at(mid)
        best = (mid, n)
        if n == target_segments:
            return mid, n, True
        if n > target_segments:   # too many segments -> raise penalty
            lo = mid
        else:                     # too few segments  -> lower penalty
            hi = mid
    return best[0], best[1], False


def dynp_k(signal, k, model, min_size):
    """Exact known-#-changepoints search as a fallback to hit exactly k segs."""
    algo = rpt.Dynp(model=model, min_size=min_size, jump=1).fit(signal)
    return algo.predict(n_bkps=k - 1)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    k = args.k
    if args.match_decode:
        dec = np.load(args.match_decode, allow_pickle=True)
        k = int(dec.get("k", dec["state_seq"].max() + 1))
        print(f"Matched k={k} from {args.match_decode}")

    print("=" * 64)
    print(f"Baseline (c): PELT change-point detection on weights  "
          f"(k={k}, cost={args.model})")
    print("=" * 64)

    _, Z_scaled, window_ids, seed_ids = load_weights_pca(args.weights_npz)
    centroid_seq, sorted_window_ids = build_centroid_sequence(
        Z_scaled, window_ids, seed_ids)
    N, D = centroid_seq.shape
    signal = np.ascontiguousarray(centroid_seq)
    dates = load_window_dates(args.manifest, sorted_window_ids)
    print(f"\nSegmenting the {N}-window trajectory in {D}-dim PCA space "
          f"into exactly {k} contiguous segments.")

    pen, n_seg, exact = find_penalty_for_k(signal, k, args.model, args.min_size)
    if exact:
        bkps = pelt_predict(signal, pen, args.model, args.min_size)
        how  = f"PELT with bisected penalty={pen:.4g}"
    else:
        bkps = dynp_k(signal, k, args.model, args.min_size)
        how  = (f"Dynp exact search (PELT's breakpoint count is a step "
                f"function of penalty and skipped {k} segments; bisection "
                f"reached {n_seg})")
    labels_k = relabel_by_first_appearance(bkps_to_labels(bkps, N))
    k_eff    = int(len(np.unique(labels_k)))
    print(f"\n  Method: {how}")
    print_segmentation(labels_k, sorted_window_ids, k_eff,
                       "PELT segmentation:")

    out_k = os.path.join(args.output_dir, f"final_decode_pelt_k{k_eff}.npz")
    save_decode(out_k, state_seq=labels_k, window_ids=sorted_window_ids, k=k_eff,
                centroid_seq=centroid_seq, method="pelt_weights",
                extra=dict(cost_model=np.str_(args.model),
                           breakpoints=np.asarray(bkps, dtype=int),
                           penalty=np.float64(pen if exact else np.nan),
                           exact_pelt=np.bool_(exact)))
    print(f"\nSaved decode: {out_k}")
    plot_timeline(labels_k, sorted_window_ids, k_eff,
                  f"PELT on weights (k={k_eff})",
                  os.path.join(args.output_dir, f"timeline_pelt_k{k_eff}.png"),
                  dates=dates)

    print("\nNext, evaluate it with the SAME machinery as the HMM:")
    print(f"  python src/within_across_states.py --decode_npz {out_k} \\")
    print(f"      --f1_npz data/hmm_perf/6way/cross_window_f1.npz \\")
    print(f"      --output_dir data/baselines/pelt/within_across")


if __name__ == "__main__":
    main()