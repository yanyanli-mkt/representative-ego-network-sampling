#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
06_run_sampling_hop2.py
-----------------------
Step 3 of the replication pipeline (Web Appendix F).

Replicates all analyses from ``02_run_sampling.py`` but under the stricter
contamination criterion: both 1st- AND 2nd-degree contamination are controlled.

  Part A — Exclusion baseline with 2nd-degree contamination:
    Starts from the 1st-degree post-exclusion samples (from script 02),
    further removes egos that share any alter with another sampled ego
    (2nd-degree contamination), and records remaining sizes and KS distances.

  Part B — Representative sampling with hop-2 constraint:
    Runs ``Balanced_Ego_Constraint_Sampling_hop2`` (h=3), which ensures all
    sampled egos are ≥ 3 hops apart. Passes ``hop2_list=None`` for large
    networks (Twitch, Pokec), triggering on-the-fly computation.

Outputs (saved under output/samples/<network_version>/)
-------------------------------------------------------
  <network_version>/summary_contamination_2nd_after_RM_500sample.csv
      Per-simulation 2nd-degree contamination stats for the exclusion approach
  <network_version>/summary_hop2_max_remaining_RM.csv
      Max feasible exclusion sample sizes under 2nd-degree constraint (Table F2)
  <network_version>_sample_match_<N>_ego_2nd_degree/
      RP node index files (hop-2 constrained)

Usage
-----
    python scripts/06_run_sampling_hop2.py --network lastfm
    python scripts/06_run_sampling_hop2.py --network all
"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from sampler import (
    Balanced_Ego_Constraint_Sampling_hop2,
    KS_dstat,
    KS_mv,
    initial_ego_sample_hop2,
    degree_list,
    get_hop1_nb_list,
    get_one_hop2_list,
)
from utils import (
    NETWORK_CONFIG,
    SIMULATED_NETWORK_CONFIG,
    load_network_data,
)

# ---------------------------------------------------------------------------
# Parameters (matching the paper)
# ---------------------------------------------------------------------------
NUM_INITIAL_SAMPLE = 500
NSIM               = 100
MAX_ITER           = 200_000
NGROUP             = 2
LOSS_RATE          = 1
TOLERANCE          = 1e-6

# Large networks: skip pre-building hop2 list; use on-the-fly computation
LARGE_NETWORKS = {"pokec", "twitch"}


def _get_2nd_degree_contaminated(ego_set, hop1_list, hop2_list, is_large):
    """
    Identify egos in ``ego_set`` that have at least one other ego as a
    2nd-degree neighbor (i.e., share at least one common alter).

    Parameters
    ----------
    ego_set : array-like
        Node indices of the ego sample (combined treat + control).
    hop1_list : dict
    hop2_list : dict or None
        None for large networks (triggers on-the-fly computation).
    is_large : bool

    Returns
    -------
    set : contaminated ego indices
    """
    ego_set_s = set(ego_set)
    contaminated = set()
    for ego in ego_set:
        if is_large or hop2_list is None:
            hop2 = get_one_hop2_list(ego, hop1_list)
        else:
            hop2 = hop2_list[ego]
        if hop2 & ego_set_s:
            contaminated.add(ego)
    return contaminated


def run_exclusion_2nd_degree(
    network_version, a_degree, a_cluster_coef, a_eigen_centrality,
    hop1_list, hop2_list, node_list, A,
    num_initial_sample, nsim, output_dir
):
    """
    Part A: measure 2nd-degree contamination in post-exclusion samples and
    compute the further-cleaned sample characteristics.
    """
    is_large = network_version in LARGE_NETWORKS
    total_node = len(node_list)

    # Read 1st-degree exclusion samples from script 02
    rm_dir = os.path.join(
        output_dir,
        f"{network_version}_sample_{num_initial_sample}_remove"
    )

    rows = []
    for sim in range(nsim):
        prefix = os.path.join(rm_dir, str(num_initial_sample))
        rp_node1 = np.genfromtxt(f"{prefix}_ego_node1_sim{sim}.csv").astype(int)
        rp_node2 = np.genfromtxt(f"{prefix}_ego_node2_sim{sim}.csv").astype(int)
        rp_node  = np.concatenate((rp_node1, rp_node2))

        # 2nd-degree contamination: ego shares an alter with another ego
        ct2 = _get_2nd_degree_contaminated(rp_node, hop1_list, hop2_list, is_large)

        # Also: egos whose 2nd-degree neighborhood contains a *treated* ego
        ct2_T = set()
        for ego in rp_node:
            hop2 = (get_one_hop2_list(ego, hop1_list) if (is_large or hop2_list is None)
                    else hop2_list[ego])
            if hop2 & set(rp_node1):
                ct2_T.add(ego)

        rp_remain_2nd = list(set(rp_node) - ct2)

        rows.append({
            "sim": sim,
            "initial_sample": num_initial_sample,
            "RM_1st_remain": len(rp_node),
            "pcent_2nd_contamination": len(ct2) / len(rp_node) if rp_node.size > 0 else np.nan,
            "RM_2nd_remain": len(rp_remain_2nd),
            "avg_degree_2nd":     a_degree[rp_remain_2nd].mean() if rp_remain_2nd else np.nan,
            "avg_cluster_2nd":    a_cluster_coef[rp_remain_2nd].mean() if rp_remain_2nd else np.nan,
            "KS_degree_2nd":      KS_dstat(a_degree, a_degree[rp_remain_2nd]) if rp_remain_2nd else np.nan,
            "KS_cluster_2nd":     KS_dstat(a_cluster_coef, a_cluster_coef[rp_remain_2nd]) if rp_remain_2nd else np.nan,
            "pcent_2nd_causal_contamination": len(ct2_T) / len(rp_node) if rp_node.size > 0 else np.nan,
        })

    result = pd.DataFrame(rows)
    out_dir = os.path.join(output_dir, network_version)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "summary_contamination_2nd_after_RM_500sample.csv")
    result.to_csv(out_path, index=False)
    print(f"  2nd-degree exclusion summary saved to {out_path}")
    return result


def run_rp_hop2_sampling(
    network_version, a_degree, a_cluster_coef,
    hop1_list, hop2_list, node_list, total_node, total_link,
    num_initial_sample, nsim, max_iter,
    rm_2nd_summary, output_dir
):
    """
    Part B: MH representative sampling with hop-2 constraint.
    """
    is_large = network_version in LARGE_NETWORKS
    p = 10 * (total_link / total_node) * np.log10(total_node)

    sim_version = os.path.join(
        output_dir,
        f"{network_version}_sample_match_{num_initial_sample}_ego_2nd_degree"
    )
    os.makedirs(sim_version, exist_ok=True)

    covariate_list = [a_degree, np.round(a_cluster_coef, 5)]

    # Matched sample size: half of 2nd-degree remaining (per condition)
    rm_sizes = (rm_2nd_summary["RM_2nd_remain"] / NGROUP).astype(int)

    rows = []
    for sim in range(nsim):
        nsample = int(rm_sizes.iloc[sim])
        if nsample < 2:
            print(f"  [hop2 RP] sim {sim+1}: nsample={nsample} too small, skipping")
            continue
        print(f"  [hop2 RP] sim {sim+1}/{nsim}, nsample={nsample}", flush=True)

        # Initialize with hop-2 constraint
        h2_list_arg = None if is_large else hop2_list
        initial_sample, initial_candidate_set = initial_ego_sample_hop2(
            node_list, hop1_list, h2_list_arg, int(nsample * NGROUP)
        )
        initial_sample_ego = list(initial_sample)
        np.random.shuffle(initial_sample_ego)

        prefix = os.path.join(sim_version, str(nsample))

        # MH sampler with hop-2 constraint
        rp_node1, rp_node2, rp_KS1, rp_KS2 = Balanced_Ego_Constraint_Sampling_hop2(
            degree_list, KS_mv,
            covariate_list, hop1_list, h2_list_arg,
            LOSS_RATE, NGROUP, nsample, p,
            max_iter, initial_sample_ego, initial_candidate_set,
            False, TOLERANCE
        )

        np.savetxt(f"{prefix}_ego_node1_sim{sim}.csv", list(rp_node1))
        np.savetxt(f"{prefix}_ego_node2_sim{sim}.csv", list(rp_node2))

        rows.append({
            "sim": sim, "nsample": nsample,
            "KS_treat": rp_KS1, "KS_control": rp_KS2,
            "avg_degree_treat":   a_degree[list(rp_node1)].mean(),
            "avg_degree_control": a_degree[list(rp_node2)].mean(),
        })

    summary = pd.DataFrame(rows)
    summary.to_csv(os.path.join(sim_version, "summary_rp_hop2_sampling.csv"), index=False)
    print(f"  Hop-2 RP summary saved to {sim_version}/")
    return summary


def run_max_feasible_size(
    network_version, a_degree, a_cluster_coef, a_eigen_centrality,
    hop1_list, hop2_list, node_list, A, total_node, total_link,
    output_dir, nsim=100
):
    """
    Compute the maximum feasible sample size under both 1st- and 2nd-degree
    contamination for the exclusion approach (Table F2 inputs).

    Sweeps sample sizes from 50 to 500, recording remaining samples after
    1st-degree exclusion then additional 2nd-degree exclusion.
    """
    is_large = network_version in LARGE_NETWORKS
    nsamples = np.arange(50, 501, 50).astype(int)

    ct1_rows = []
    ct2_rows = []

    for nsample in nsamples:
        print(f"  [Max size] nsample={nsample}", flush=True)
        ct1_sims, ct2_sims = [], []
        for _ in range(nsim):
            rp_node = np.random.choice(total_node, nsample * 2, replace=False)
            rp_node1 = rp_node[:nsample]
            rp_node2 = rp_node[nsample:]

            d_vec = np.zeros(total_node)
            d_vec[rp_node] = 1
            nd_vec = A.dot(d_vec)

            rp1_vec = np.zeros(total_node); rp1_vec[rp_node1] = 1
            rp2_vec = np.zeros(total_node); rp2_vec[rp_node2] = 1

            # 1st-degree post-exclusion
            Dmat = np.column_stack([
                rp1_vec * (nd_vec == 0),
                rp2_vec * (nd_vec == 0),
            ])
            rp1_remain = np.nonzero(Dmat[:, 0])[0]
            rp2_remain = np.nonzero(Dmat[:, 1])[0]
            rp_remain  = np.concatenate((rp1_remain, rp2_remain))

            if len(rp_remain) == 0:
                continue

            ct1_sims.append({
                "remaining": len(rp_remain),
                "avg_degree": a_degree[rp_remain].mean(),
                "avg_cluster": a_cluster_coef[rp_remain].mean(),
                "KS_degree": KS_dstat(a_degree, a_degree[rp_remain]),
                "KS_cluster": KS_dstat(a_cluster_coef, a_cluster_coef[rp_remain]),
            })

            # 2nd-degree further exclusion
            ct2 = _get_2nd_degree_contaminated(
                rp_remain, hop1_list, hop2_list, is_large
            )
            rp_remain_2nd = list(set(rp_remain) - ct2)

            if rp_remain_2nd:
                ct2_sims.append({
                    "remaining_2nd": len(rp_remain_2nd),
                    "avg_degree": a_degree[rp_remain_2nd].mean(),
                    "avg_cluster": a_cluster_coef[rp_remain_2nd].mean(),
                    "KS_degree": KS_dstat(a_degree, a_degree[rp_remain_2nd]),
                    "KS_cluster": KS_dstat(a_cluster_coef, a_cluster_coef[rp_remain_2nd]),
                })

        if ct1_sims:
            ct1_mean = pd.DataFrame(ct1_sims).mean()
            ct1_rows.append({"nsample": nsample, **ct1_mean.to_dict()})
        if ct2_sims:
            ct2_mean = pd.DataFrame(ct2_sims).mean()
            ct2_rows.append({"nsample": nsample, **ct2_mean.to_dict()})

    out_dir = os.path.join(output_dir, network_version)
    os.makedirs(out_dir, exist_ok=True)

    pd.DataFrame(ct1_rows).to_csv(
        os.path.join(out_dir, "summary_hop1_max_remaining.csv"), index=False
    )
    pd.DataFrame(ct2_rows).to_csv(
        os.path.join(out_dir, "summary_hop2_max_remaining_RM.csv"), index=False
    )
    print(f"  Max feasible size tables saved to {out_dir}/")


def process_network(network_version, args):
    print(f"\n{'='*60}\n  Network (hop-2): {network_version}\n{'='*60}")
    t0 = time.time()

    data = load_network_data(
        network_version,
        load_hop2=(network_version not in LARGE_NETWORKS),
        data_dir=args.data_dir
    )
    a_degree       = data["a_degree"]
    a_cluster_coef = data["a_cluster_coef"]
    a_eigen        = data["a_eigen_centrality"]
    hop1_list      = data["hop1_list"]
    hop2_list      = data.get("hop2_list", None)

    total_node = len(a_degree)
    node_list  = list(range(total_node))

    # Rebuild sparse adjacency
    rows_, cols_ = [], []
    for node, neighbors in hop1_list.items():
        for nb in neighbors:
            rows_.append(node); cols_.append(nb)
    A = sp.csr_matrix(
        (np.ones(len(rows_)), (rows_, cols_)),
        shape=(total_node, total_node)
    )
    total_link = A.nnz // 2

    output_dir = args.output_dir

    # Part A
    print("\n[Part A] 2nd-degree contamination in exclusion samples...")
    rm_2nd = run_exclusion_2nd_degree(
        network_version, a_degree, a_cluster_coef, a_eigen,
        hop1_list, hop2_list, node_list, A,
        args.num_initial_sample, args.nsim, output_dir
    )

    # Part B
    print("\n[Part B] MH sampling with hop-2 constraint...")
    run_rp_hop2_sampling(
        network_version, a_degree, a_cluster_coef,
        hop1_list, hop2_list, node_list, total_node, total_link,
        args.num_initial_sample, args.nsim, args.max_iter,
        rm_2nd, output_dir
    )

    # Max feasible size table (Table F2)
    print("\n[Part C] Max feasible sample size sweep...")
    run_max_feasible_size(
        network_version, a_degree, a_cluster_coef, a_eigen,
        hop1_list, hop2_list, node_list, A, total_node, total_link,
        output_dir
    )

    print(f"\nDone in {(time.time()-t0)/60:.1f} min")


def main():
    all_networks = list(NETWORK_CONFIG.keys()) + list(SIMULATED_NETWORK_CONFIG.keys())

    parser = argparse.ArgumentParser(
        description="2nd-degree contamination analysis (Web Appendix F)."
    )
    parser.add_argument("--network", nargs="+", default=["all"])
    parser.add_argument("--data_dir", default="./data")
    parser.add_argument("--output_dir", default="./output/samples")
    parser.add_argument("--num_initial_sample", type=int, default=NUM_INITIAL_SAMPLE)
    parser.add_argument("--nsim", type=int, default=NSIM)
    parser.add_argument("--max_iter", type=int, default=MAX_ITER)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    networks = all_networks if args.network == ["all"] else args.network
    for nv in networks:
        process_network(nv, args)

    print("\nAll networks complete.")


if __name__ == "__main__":
    main()
