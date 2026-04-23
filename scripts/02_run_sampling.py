#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
02_run_sampling.py
------------------
Step 2 of the replication pipeline. Pure sampling only.

For each population network, runs two procedures in sequence:

  Part A — Exclusion baseline (Section 4.2):
    Randomly samples num_initial_sample ego networks per condition,
    identifies and removes contaminated egos (1st-degree contamination,
    within or across conditions), and saves the surviving ego and alter
    node indices for each simulation.

  Part B — Representative sampling (Section 4.3):
    Runs the MH sampler (Balanced_Ego_Constraint_Sampling_hop1, loss_rate=1)
    for each simulation, matching the post-exclusion sample size so that
    comparisons are fair. Saves the resulting ego node indices.

Outputs (saved under output/samples/)
--------------------------------------
  <network_version>_sample_<N>_remove/
      <N>_ego_node1_sim<i>.csv           Exclusion treat egos, sim i
      <N>_ego_node2_sim<i>.csv           Exclusion control egos, sim i
      <N>_alter_node1_sim<i>.csv         Exclusion treat alters, sim i
      <N>_alter_node2_sim<i>.csv         Exclusion control alters, sim i
      summary_remaining_sample_size.csv  Per-sim remaining sizes and KS

  <network_version>_sample_match_<N>_ego_loss1/
      <n>_ego_node1_sim<i>.csv           RP treat egos, sim i
      <n>_ego_node2_sim<i>.csv           RP control egos, sim i

Next step
---------
Run 02b_summary_tables.py to generate summary tables
(Table 2, Figure 5 input, Figures 3 & 4 data).

Usage
-----
    python scripts/02_run_sampling.py --network lastfm
    python scripts/02_run_sampling.py --network all

Notes
-----
- Must be run after 01_prepare_networks.py.
- The main paper uses nsim=100, num_initial_sample=500, max_iter=200000.
- Runtime: ~57 minutes per network (footnote 4 of the paper).
"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from sampler import (
    Balanced_Ego_Constraint_Sampling_hop1,
    KS_dstat,
    KS_mv,
    initial_ego_sample_hop1,
    degree_list,
    get_hop1_nb_list,
)
from utils import NETWORK_CONFIG, SIMULATED_NETWORK_CONFIG, load_network_data

# ---------------------------------------------------------------------------
# Parameters (matching the paper)
# ---------------------------------------------------------------------------
NUM_INITIAL_SAMPLE = 500
NSIM               = 100
MAX_ITER           = 200_000
NGROUP             = 2
LOSS_RATE          = 1
TOLERANCE          = 1e-6


def run_exclusion_baseline(
    network_version, a_degree, a_cluster_coef, hop1_list,
    node_list, A, num_initial_sample, nsim, output_dir
):
    """
    Part A: exclusion baseline.

    For each simulation, randomly samples 2 * num_initial_sample egos,
    assigns them equally to treatment and control, then removes any egos
    that are contaminated (directly connected to any other sampled ego,
    within or across treatment conditions — nd_vec > 0).

    Returns a DataFrame with remaining sample sizes and KS distances.
    """
    total_node  = len(node_list)
    sim_version = os.path.join(
        output_dir,
        f"{network_version}_sample_{num_initial_sample}_remove"
    )
    os.makedirs(sim_version, exist_ok=True)

    summary_rows = []

    for i in range(nsim):
        print(f"  [Exclusion] sim {i+1}/{nsim}", flush=True)

        rp_node  = np.random.choice(total_node, num_initial_sample * 2, replace=False)
        rp_node1 = rp_node[:num_initial_sample]
        rp_node2 = rp_node[num_initial_sample:]

        d_vec  = np.zeros(total_node); d_vec[rp_node] = 1
        nd_vec = A.dot(d_vec)

        Dmat = np.column_stack((
            (d_vec > 0) & (nd_vec == 0),   # clean ego
            (d_vec == 0) & (nd_vec == 1),  # clean alter
        )).astype(float)
        Dmat[Dmat == 0] = np.nan

        treat_nodes = np.array(list(rp_node1) + get_hop1_nb_list(list(rp_node1), hop1_list))
        Dmat_T = Dmat.copy()
        mask_T = np.ones_like(Dmat_T, dtype=bool)
        mask_T[treat_nodes, :] = False
        Dmat_T[mask_T] = np.nan

        ctrl_nodes = np.array(list(rp_node2) + get_hop1_nb_list(list(rp_node2), hop1_list))
        Dmat_C = Dmat.copy()
        mask_C = np.ones_like(Dmat_C, dtype=bool)
        mask_C[ctrl_nodes, :] = False
        Dmat_C[mask_C] = np.nan

        rp_node1_remove    = np.nonzero(~np.isnan(Dmat_T[:, 0]))[0]
        rp_node2_remove    = np.nonzero(~np.isnan(Dmat_C[:, 0]))[0]
        rp_node1_nb_remove = np.nonzero(~np.isnan(Dmat_T[:, 1]))[0]
        rp_node2_nb_remove = np.nonzero(~np.isnan(Dmat_C[:, 1]))[0]

        prefix = os.path.join(sim_version, str(num_initial_sample))
        np.savetxt(f"{prefix}_ego_node1_sim{i}.csv",   rp_node1_remove)
        np.savetxt(f"{prefix}_ego_node2_sim{i}.csv",   rp_node2_remove)
        np.savetxt(f"{prefix}_alter_node1_sim{i}.csv", rp_node1_nb_remove)
        np.savetxt(f"{prefix}_alter_node2_sim{i}.csv", rp_node2_nb_remove)

        n1 = len(rp_node1_remove)
        n2 = len(rp_node2_remove)
        summary_rows.append({
            "initial sample size": num_initial_sample,
            "size g1":    n1,
            "size g2":    n2,
            "pcent g1":   n1 / num_initial_sample,
            "pcent g2":   n2 / num_initial_sample,
            "avg degree g1": a_degree[rp_node1_remove].mean() if n1 > 0 else np.nan,
            "avg degree g2": a_degree[rp_node2_remove].mean() if n2 > 0 else np.nan,
            "KS g1": KS_dstat(a_degree, a_degree[rp_node1_remove]) if n1 > 0 else np.nan,
            "KS g2": KS_dstat(a_degree, a_degree[rp_node2_remove]) if n2 > 0 else np.nan,
        })

    summary_df   = pd.DataFrame(summary_rows)
    summary_path = os.path.join(sim_version, "summary_remaining_sample_size.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"  Exclusion summary saved to {summary_path}")
    return summary_df


def run_representative_sampling(
    network_version, a_degree, a_cluster_coef, hop1_list,
    node_list, total_node, total_link,
    num_initial_sample, nsim, max_iter,
    removal_summary, output_dir
):
    """
    Part B: MH representative sampling.

    For each simulation, matches the post-exclusion sample size, then runs
    Balanced_Ego_Constraint_Sampling_hop1 (loss_rate=1) and saves the
    resulting ego node indices.
    """
    p = 10 * (total_link / total_node) * np.log10(total_node)

    sim_version = os.path.join(
        output_dir,
        f"{network_version}_sample_match_{num_initial_sample}_ego_loss1"
    )
    os.makedirs(sim_version, exist_ok=True)

    covariate_list = [a_degree, np.round(a_cluster_coef, 5)]
    rm_sizes = ((removal_summary["size g1"] + removal_summary["size g2"]) / 2).astype(int)

    rp_summary_rows = []

    for sim in range(nsim):
        nsample = int(rm_sizes.iloc[sim])
        print(f"  [RP] sim {sim+1}/{nsim}  nsample={nsample}", flush=True)

        initial_sample, initial_candidate_set = initial_ego_sample_hop1(
            node_list, hop1_list, int(nsample * NGROUP)
        )
        initial_sample_ego = list(initial_sample)
        np.random.shuffle(initial_sample_ego)

        prefix = os.path.join(sim_version, str(nsample))

        rp_node1, rp_node2, rp_KS1, rp_KS2 = Balanced_Ego_Constraint_Sampling_hop1(
            degree_list, KS_mv,
            covariate_list, hop1_list,
            LOSS_RATE, NGROUP, nsample, p,
            max_iter, initial_sample_ego, initial_candidate_set,
            False, TOLERANCE
        )

        np.savetxt(f"{prefix}_ego_node1_sim{sim}.csv", list(rp_node1))
        np.savetxt(f"{prefix}_ego_node2_sim{sim}.csv", list(rp_node2))

        rp_summary_rows.append({
            "sim": sim, "nsample": nsample,
            "KS_treat": rp_KS1, "KS_ctrl": rp_KS2,
        })

    rp_summary = pd.DataFrame(rp_summary_rows)
    rp_summary.to_csv(
        os.path.join(sim_version, "summary_rp_sampling.csv"), index=False
    )
    print(f"  RP summary saved.")
    return rp_summary


def process_network(network_version, args):
    """Run Part A and Part B for a single network."""
    print(f"\n{'='*60}\n  Network: {network_version}\n{'='*60}")
    t0 = time.time()

    data           = load_network_data(network_version, load_hop2=False,
                                       data_dir=args.data_dir)
    a_degree       = data["a_degree"]
    a_cluster_coef = data["a_cluster_coef"]
    hop1_list      = data["hop1_list"]
    total_node     = len(a_degree)
    node_list      = list(range(total_node))

    import scipy.sparse as sp
    rows, cols = [], []
    for node, neighbors in hop1_list.items():
        for nb in neighbors:
            rows.append(node); cols.append(nb)
    A = sp.csr_matrix(
        (np.ones(len(rows)), (rows, cols)),
        shape=(total_node, total_node)
    )
    total_link = A.nnz // 2

    output_dir = args.output_dir

    print("\n[Part A] Running exclusion baseline...")
    removal_summary = run_exclusion_baseline(
        network_version, a_degree, a_cluster_coef, hop1_list,
        node_list, A, args.num_initial_sample, args.nsim, output_dir
    )

    print("\n[Part B] Running MH representative sampling...")
    run_representative_sampling(
        network_version, a_degree, a_cluster_coef, hop1_list,
        node_list, total_node, total_link,
        args.num_initial_sample, args.nsim, args.max_iter,
        removal_summary, output_dir
    )

    print(f"\nDone in {(time.time()-t0)/60:.1f} min")
    print("Next: run scripts/02b_summary_tables.py for summary tables.")


def main():
    all_networks = list(NETWORK_CONFIG.keys()) + list(SIMULATED_NETWORK_CONFIG.keys())

    parser = argparse.ArgumentParser(
        description="Run exclusion baseline and MH representative sampling."
    )
    parser.add_argument("--network",            nargs="+", default=["all"])
    parser.add_argument("--data_dir",           default="./data")
    parser.add_argument("--output_dir",         default="./output/samples")
    parser.add_argument("--num_initial_sample", type=int, default=NUM_INITIAL_SAMPLE)
    parser.add_argument("--nsim",               type=int, default=NSIM)
    parser.add_argument("--max_iter",           type=int, default=MAX_ITER)
    parser.add_argument("--seed",               type=int, default=0)
    args = parser.parse_args()

    np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    networks = all_networks if args.network == ["all"] else args.network
    for nv in networks:
        process_network(nv, args)

    print("\nAll networks complete.")


if __name__ == "__main__":
    main()
