"""
src/kmeans_on_weights.py

Baseline (a): k-means on the per-window weight centroids.

This clusters the EXACT same per-window centroid sequence the HMM was fit on
(the across-seed mean of the z-scored PCA vectors at each window), with
n_clusters set to the same k as the HMM. k-means has no notion of time or
contiguity, so comparing it to the HMM isolates how much of the HMM result is
just "clustering the weight centroids" versus the HMM's sequential/Markov
machinery. If a memoryless clustering of the same features recovers the same
partition and the same within/across generalisation gap, the temporal model
adds little; if it does not, the contiguity the HMM imposes is doing work.

k-means cluster labels are NOT generally contiguous in time — that is the point
of the comparison, and the downstream within/across test handles non-contiguous
groups without modification.

Output: final_decode_kmeans_k{k}.npz — drop-in for within_across_states.py,
check_equal_windows.py and state_pair_correlation.py.

Usage
-----
  python src/kmeans_on_weights.py \
      --weights_npz  data/hmm_weights/6way/weights_pca.npz \
      --manifest     data/splits/hmm_windows/HMM_windows_manifest.csv \
      --output_dir   data/baselines/kmeans \
      --match_decode data/hmm_hmm/6way/final_decode_k7.npz \
      --n_init 50 --seed 42

  # or set k directly
  python src/kmeans_on_weights.py --k 7

Requirements: scikit-learn, numpy, pandas, matplotlib
"""

import os
import argparse
import warnings

import numpy as np
from sklearn.cluster import KMeans

from segmentation_common import (
    load_weights_pca, build_centroid_sequence, load_window_dates,
    relabel_by_first_appearance, save_decode, print_segmentation,
    plot_timeline, contiguous_runs,
)

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", message=".*KMeans.*")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights_npz", default="data/hmm_weights/weights_pca.npz")
    p.add_argument("--manifest",    default="data/splits/hmm_windows/HMM_windows_manifest.csv")
    p.add_argument("--output_dir",  default="data/baselines/kmeans")
    p.add_argument("--match_decode", default=None,
                   help="Optional HMM decode .npz to copy k from (head-to-head)")
    p.add_argument("--k",        type=int, default=7,
                   help="Number of clusters (ignored if --match_decode is given)")
    p.add_argument("--n_init",   type=int, default=50,
                   help="k-means restarts (best inertia kept)")
    p.add_argument("--max_iter", type=int, default=500)
    p.add_argument("--seed",     type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    k = args.k
    if args.match_decode:
        dec = np.load(args.match_decode, allow_pickle=True)
        k = int(dec.get("k", dec["state_seq"].max() + 1))
        print(f"Matched k={k} from {args.match_decode}")

    print("=" * 64)
    print(f"Baseline (a): k-means on weight centroids  (k={k})")
    print("=" * 64)

    # SAME features the HMM saw: across-seed mean of z-scored PCA vectors
    _, Z_scaled, window_ids, seed_ids = load_weights_pca(args.weights_npz)
    centroid_seq, sorted_window_ids = build_centroid_sequence(
        Z_scaled, window_ids, seed_ids)
    N, D = centroid_seq.shape
    print(f"\nClustering {N} windows in the {D}-dim z-scored PCA weight space.")

    km = KMeans(n_clusters=k, n_init=args.n_init, max_iter=args.max_iter,
                random_state=args.seed)
    raw_labels = km.fit_predict(centroid_seq)
    state_seq  = relabel_by_first_appearance(raw_labels)
    k_eff      = int(len(np.unique(state_seq)))
    print(f"  k-means inertia: {km.inertia_:.4f}")

    print_segmentation(state_seq, sorted_window_ids, k_eff,
                       "k-means segmentation:")

    # How contiguous is it? (HMM groups were fully contiguous here)
    n_runs = len(contiguous_runs(state_seq))
    contig_msg = ("fully contiguous" if n_runs == k_eff else
                  "NOT contiguous — clusters recur across time, unlike the HMM decode")
    print(f"\n  Contiguity check: {n_runs} runs for {k_eff} clusters ({contig_msg}).")

    out = os.path.join(args.output_dir, f"final_decode_kmeans_k{k_eff}.npz")
    save_decode(out, state_seq=state_seq, window_ids=sorted_window_ids, k=k_eff,
                centroid_seq=centroid_seq, method="kmeans_weights",
                extra=dict(cluster_centers=km.cluster_centers_,
                           inertia=np.float64(km.inertia_)))
    print(f"\nSaved decode: {out}")

    dates = load_window_dates(args.manifest, sorted_window_ids)
    plot_timeline(state_seq, sorted_window_ids, k_eff,
                  f"k-means on weight centroids (k={k_eff})",
                  os.path.join(args.output_dir, f"timeline_kmeans_k{k_eff}.png"),
                  dates=dates)

    print("\nNext, evaluate it with the SAME machinery as the HMM:")
    print(f"  python src/within_across_states.py --decode_npz {out} \\")
    print(f"      --f1_npz data/hmm_perf/6way/cross_window_f1.npz \\")
    print(f"      --output_dir data/baselines/kmeans/within_across")


if __name__ == "__main__":
    main()