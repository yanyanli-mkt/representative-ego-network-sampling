#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
04_boundary_conditions.py
-------------------------
Step 4 of the replication pipeline.

Generates Table 3: maximum feasible ego network sample size under the
representative sampling method and the exclusion approach, subject to
KS distance (degree) < 0.1.

For both methods, KS is computed on the COMBINED sample (treat + ctrl),
The output CSV shows avg_remain and avg_KS
at each sweep point so you can identify the boundary (max avg_remain where
avg_KS < 0.1) by inspection.

Outputs (saved under output/tables/)
-------------------------------------
  <network>_boundary_rp.csv
      nsample       — RP sample size per condition
      avg_remain    — mean remaining sample size PER CONDITION across nsim
      avg_KS        — mean KS(combined RP sample, population degree)
      feasible      — avg_KS < 0.1

  <network>_boundary_excl.csv
      nsample       — initial draw per condition
      avg_remain    — mean remaining sample size PER CONDITION after exclusion
      avg_KS        — mean KS(combined remaining, population degree)
      feasible      — avg_KS < 0.1

  table3_boundary_conditions.csv
      Summary: max avg_remain (per condition) where avg_KS < 0.1 for each network and method

Usage
-----
    python scripts/04_boundary_conditions.py --network all
    python scripts/04_boundary_conditions.py --network simulate_base --nsim 100
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
    Balanced_Ego_Constraint_Sampling_hop1,
    KS_dstat,
    KS_mv,
    initial_ego_sample_hop1,
    degree_list,
)
from utils import NETWORK_CONFIG, SIMULATED_NETWORK_CONFIG, load_network_data

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
KS_THRESHOLD = 0.1
NGROUP       = 2
LOSS_RATE    = 1
TOLERANCE    = 1e-6

# Sweep ranges: (lo, hi, step) for initial sample size per condition.
# Chosen to bracket each network's feasibility boundary.
SWEEP_RANGES_RP = {
    "lastfm":                           (100,  2000,  100),
    "twitch":                         (500, 15000,  500),
    "pokec":                             (1000, 55000, 1000),
    "simulate_small": (100,  2000,  100),
    "simulate_base":  (500,  8000,  500),
    "simulate_sparse": (500, 20000,  500),
}

SWEEP_RANGES_EXCL = {
    # Smaller/denser networks: 50 to 500 per condition, step 50
    # (matches footnote 7a in the paper)
    "lastfm":                           (50,   500,   50),
    "twitch":                         (50,   500,   50),
    "simulate_small": (50,   500,   50),
    # Larger/sparser networks: coarser steps to bracket higher boundaries
    "pokec":                             (500, 8000,  100),
    "simulate_base":  (50,  1500,   50),
    "simulate_sparse": (500, 6000,  100),
}


def run_rp_boundary_sweep(
    network_version, a_degree, a_cluster_coef,
    hop1_list, node_list, total_node, total_link,
    sample_sizes, max_iter, nsim, output_dir
):
    """
    Sweep RP sample sizes. For each nsample, run the MH sampler nsim times.
    KS is computed on the COMBINED (treat + ctrl) RP sample, matching the


    Output CSV columns: nsample, avg_remain (per condition), avg_KS, feasible.
    Table 3 value = max avg_remain (per condition) where avg_KS < 0.1.
    """
    p = 10 * (total_link / total_node) * np.log10(total_node)
    covariate_list = [a_degree, np.round(a_cluster_coef, 5)]

    rows = []
    for nsample in sample_sizes:
        ks_list     = []
        remain_list = []

        for sim in range(nsim):
            initial_sample, initial_candidate_set = initial_ego_sample_hop1(
                node_list, hop1_list, int(nsample * NGROUP)
            )
            if not initial_sample:
                ks_list.append(np.nan)
                remain_list.append(np.nan)
                continue

            initial_ego = list(initial_sample)
            np.random.shuffle(initial_ego)

            try:
                rp1, rp2, _, _ = Balanced_Ego_Constraint_Sampling_hop1(
                    degree_list, KS_mv,
                    covariate_list, hop1_list,
                    LOSS_RATE, NGROUP, nsample, p,
                    max_iter, initial_ego, initial_candidate_set,
                    False, TOLERANCE
                )
                # KS on combined treat + ctrl sample
                rp_combined = np.array(list(rp1) + list(rp2))
                ks = KS_dstat(a_degree, a_degree[rp_combined])
                ks_list.append(ks)
                remain_list.append(len(rp_combined) / NGROUP)
            except Exception as e:
                print(f"    Warning: sim {sim} failed ({e})")
                ks_list.append(np.nan)
                remain_list.append(np.nan)

        avg_ks     = float(np.nanmean(ks_list))
        avg_remain = float(np.nanmean(remain_list))
        rows.append({
            "nsample":    nsample,
            "avg_remain": avg_remain,
            "avg_KS":     avg_ks,
            "feasible":   avg_ks < KS_THRESHOLD,
        })
        print(f"  [RP] nsample={nsample}, avg_remain/cond={avg_remain:.0f}, "
              f"avg_KS={avg_ks:.4f}, feasible={avg_ks < KS_THRESHOLD}")

        # Early stop if KS far above threshold
        if avg_ks > KS_THRESHOLD * 3 and nsample > sample_sizes[0]:
            print("  Early stop.")
            break

    result   = pd.DataFrame(rows)
    out_path = os.path.join(output_dir, f"{network_version}_boundary_rp.csv")
    result.to_csv(out_path, index=False)

    feasible    = result[result["feasible"]]
    max_remain  = float(feasible["avg_remain"].max()) if len(feasible) > 0 else 0
    print(f"  RP max avg_remain/cond (KS<0.1): {max_remain:.0f}")
    return result, max_remain


def run_exclusion_boundary_sweep(
    network_version, a_degree, a_cluster_coef,
    hop1_list, node_list, A, total_node,
    sample_sizes, nsim, output_dir
):
    """
    Sweep exclusion initial sample sizes. KS is computed on the COMBINED
    clean sample (treat + ctrl after removing contaminated egos), exactly


    Output CSV columns: nsample, avg_remain (per condition), avg_KS, feasible.
    Table 3 value = max avg_remain (per condition) where avg_KS < 0.1.
    """
    rows = []
    for nsample in sample_sizes:
        ks_list     = []
        remain_list = []

        for _ in range(nsim):
            rp_node  = np.random.choice(total_node, nsample * 2, replace=False)
            rp_node1 = rp_node[:nsample]
            rp_node2 = rp_node[nsample:]

            d_vec   = np.zeros(total_node); d_vec[rp_node]   = 1
            rp1_vec = np.zeros(total_node); rp1_vec[rp_node1] = 1
            rp2_vec = np.zeros(total_node); rp2_vec[rp_node2] = 1
            nd_vec  = A.dot(d_vec)

            rp1_remain = np.nonzero(rp1_vec * (nd_vec == 0))[0]
            rp2_remain = np.nonzero(rp2_vec * (nd_vec == 0))[0]
            rp_remain  = np.concatenate([rp1_remain, rp2_remain])

            if len(rp_remain) == 0:
                ks_list.append(np.nan)
                remain_list.append(0)
                continue

            ks = KS_dstat(a_degree, a_degree[rp_remain])
            ks_list.append(ks)
            remain_list.append(len(rp_remain) / NGROUP)

        avg_ks     = float(np.nanmean(ks_list))
        avg_remain = float(np.nanmean(remain_list))
        rows.append({
            "nsample":    nsample,
            "avg_remain": avg_remain,
            "avg_KS":     avg_ks,
            "feasible":   avg_ks < KS_THRESHOLD,
        })
        print(f"  [Excl] nsample={nsample}, avg_remain/cond={avg_remain:.0f}, "
              f"avg_KS={avg_ks:.4f}, feasible={avg_ks < KS_THRESHOLD}")

    result   = pd.DataFrame(rows)
    out_path = os.path.join(output_dir, f"{network_version}_boundary_excl.csv")
    result.to_csv(out_path, index=False)

    feasible   = result[result["feasible"]]
    max_remain = float(feasible["avg_remain"].max()) if len(feasible) > 0 else 0
    print(f"  Excl max avg_remain/cond (KS<0.1): {max_remain:.0f}")
    return result, max_remain


def process_network(network_version, args):
    print(f"\n{'='*60}\n  Boundary conditions: {network_version}\n{'='*60}")
    t0 = time.time()

    data           = load_network_data(network_version, load_hop2=False,
                                       data_dir=args.data_dir)
    a_degree       = data["a_degree"]
    a_cluster_coef = data["a_cluster_coef"]
    hop1_list      = data["hop1_list"]
    total_node     = len(a_degree)
    node_list      = list(range(total_node))

    rows_, cols_ = [], []
    for node, neighbors in hop1_list.items():
        for nb in neighbors:
            rows_.append(node); cols_.append(nb)
    A = sp.csr_matrix(
        (np.ones(len(rows_)), (rows_, cols_)),
        shape=(total_node, total_node)
    )
    total_link = A.nnz // 2

    lo_rp,   hi_rp,   step_rp   = SWEEP_RANGES_RP[network_version]
    lo_excl, hi_excl, step_excl = SWEEP_RANGES_EXCL[network_version]
    sample_sizes_rp   = np.arange(lo_rp,   hi_rp   + 1, step_rp).astype(int)
    sample_sizes_excl = np.arange(lo_excl, hi_excl + 1, step_excl).astype(int)

    max_iter   = max(200_000, lo_rp * 20)
    tables_dir = args.output_dir
    os.makedirs(tables_dir, exist_ok=True)

    print("\n[RP] Sweeping sample sizes...")
    rp_result, rp_max_remain = run_rp_boundary_sweep(
        network_version, a_degree, a_cluster_coef,
        hop1_list, node_list, total_node, total_link,
        sample_sizes_rp, max_iter, args.nsim, tables_dir
    )

    print("\n[Exclusion] Sweeping sample sizes...")
    excl_result, excl_max_remain = run_exclusion_boundary_sweep(
        network_version, a_degree, a_cluster_coef,
        hop1_list, node_list, A, total_node,
        sample_sizes_excl, args.nsim, tables_dir
    )

    print(f"\n  Summary: RP max_remain/cond={rp_max_remain:.0f}, "
          f"Excl max_remain/cond={excl_max_remain:.0f}")
    print(f"  Done in {(time.time()-t0)/60:.1f} min")

    return {
        "network":                network_version,
        "rp_max_remain":          rp_max_remain,
        "excl_max_remain":        excl_max_remain,
    }


def main():
    all_networks = list(NETWORK_CONFIG.keys()) + list(SIMULATED_NETWORK_CONFIG.keys())

    parser = argparse.ArgumentParser(
        description="Compute max feasible sample sizes for Table 3."
    )
    parser.add_argument("--network",    nargs="+", default=["all"])
    parser.add_argument("--data_dir",   default="./data")
    parser.add_argument("--output_dir", default="./output/tables")
    parser.add_argument("--nsim",       type=int, default=100,
                        help="Simulations per sample size. Default: 100")
    parser.add_argument("--seed",       type=int, default=0)
    args = parser.parse_args()

    np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    networks = all_networks if args.network == ["all"] else args.network

    table3_rows = []
    for nv in networks:
        row = process_network(nv, args)
        table3_rows.append(row)

    table3 = pd.DataFrame(table3_rows)
    table3.columns = [
        "Population Network",
        "Representative Sampling (max remaining per condition, KS<0.1)",
        "Exclusion Method (max remaining per condition, KS<0.1)",
    ]
    out_path = os.path.join(args.output_dir, "table3_boundary_conditions.csv")
    table3.to_csv(out_path, index=False)

    print("\n=== Table 3: Max remaining per condition where KS(degree) < 0.1 ===")
    print(table3.to_string(index=False))
    print(f"\nSaved to {out_path}")
    print("\nNote: per-network sweep details in <network>_boundary_rp.csv "
          "and <network>_boundary_excl.csv")


if __name__ == "__main__":
    main()
