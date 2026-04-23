#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
01_prepare_networks.py
----------------------
Step 1 of the replication pipeline.

Loads (real-world) or generates (simulated) all six population networks used
in the paper, computes node attributes, builds hop-1 and hop-2 neighbor lists,
and saves everything to disk under ``data/<network_version>/``.

All downstream scripts (02–06) load pre-computed files from these directories
rather than rebuilding the networks from scratch.

Networks processed
------------------
Real-world (download from SNAP — see data/README.md):
    - lastfm          (LastFM-Asia)
    - twitch        (Twitch-Gamer)
    - pokec            (Pokec)

Simulated (generated here via networkx.powerlaw_cluster_graph):
    - simulate_small   (Small,  N=10,000)
    - simulate_base    (Base,   N=100,000)
    - simulate_sparse   (Sparse, N=100,000)

Outputs per network (saved to data/<network_version>/)
-------------------------------------------------------
    a_degree.csv            Node degree array
    a_cluster_coef.csv      Clustering coefficient array (full precision)
    a_eigen_centrality.csv  Eigenvector centrality array
    a_nb_degree.csv         Average neighbor degree array
    hop1_list.pkl           Dict: node_idx -> set of 1st-degree neighbor indices
    hop2_list.pkl           Dict: node_idx -> set of 2nd-degree neighbor indices
                            (not built for Pokec due to memory; use on-the-fly
                             computation in sampler.get_one_hop2_list instead)

For simulated networks, also saves:
    Z1.csv, Z2.csv, Z3.csv  Correlated covariates (see paper footnote 6)

Usage
-----
    python scripts/01_prepare_networks.py [--networks all] [--data_dir ./data]

    # Process only specific networks:
    python scripts/01_prepare_networks.py --networks lastfm twitch

    # Skip hop2 list (saves time/memory for large networks):
    python scripts/01_prepare_networks.py --skip_hop2

Runtime notes
-------------
- LastFM and simulated networks: fast (< 5 minutes each)
- Twitch-Gamer: ~20 minutes (eigenvector centrality)
- Pokec: ~2 hours (1.6M nodes); hop2 list is skipped by default
"""

import argparse
import os
import sys
import time

import numpy as np

# Allow running from repo root or scripts/ directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from utils import (
    NETWORK_CONFIG,
    SIMULATED_NETWORK_CONFIG,
    build_hop1_list,
    build_hop2_list,
    compute_node_attributes,
    compute_p,
    generate_covariates,
    generate_simulated_network,
    load_real_network,
    save_network_data,
)

# ---------------------------------------------------------------------------
# Networks skipped for hop2 pre-build due to memory:
#   simulate_base (N=100K, dense)  → ~682MB pkl, computed on-the-fly
#   pokec / twitch                 → too large
# simulate_sparse (N=100K, sparse) → ~135MB pkl, pre-built normally
# ---------------------------------------------------------------------------
SKIP_HOP2_DEFAULT = {"pokec", "twitch", "simulate_base"}


def prepare_one_network(network_version, data_dir, skip_hop2=False):
    """
    Load or generate one network, compute all attributes, and save to disk.

    Parameters
    ----------
    network_version : str
    data_dir : str
    skip_hop2 : bool
        If True, skip building the hop-2 neighbor list.
    """
    print(f"\n{'='*60}")
    print(f"  Processing: {network_version}")
    print(f"{'='*60}")
    t0 = time.time()

    # ------------------------------------------------------------------
    # 1. Load or generate the network
    # ------------------------------------------------------------------
    if network_version in NETWORK_CONFIG:
        NW, features = load_real_network(network_version, data_dir=data_dir)
    else:
        NW = generate_simulated_network(network_version)
        features = None

    # Relabel nodes to contiguous 0-based integers (required by sampler)
    import networkx as nx
    mapping = {n: i for i, n in enumerate(sorted(NW.nodes()))}
    NW = nx.relabel_nodes(NW, mapping)

    # ------------------------------------------------------------------
    # 2. Compute node attributes
    # ------------------------------------------------------------------
    print("  Computing node attributes...")
    attrs = compute_node_attributes(NW)

    print(
        f"  N={attrs['node_list'].shape[0]:,}  |  "
        f"avg degree={attrs['a_degree'].mean():.3f}  |  "
        f"avg clustering={attrs['a_cluster_coef'].mean():.3f}  |  "
        f"p={compute_p(NW):.2f}"
    )

    # ------------------------------------------------------------------
    # 3. Build hop-1 neighbor list
    # ------------------------------------------------------------------
    print("  Building hop-1 neighbor list...")
    hop1_list = build_hop1_list(NW, attrs["node_list"])

    # ------------------------------------------------------------------
    # 4. Build hop-2 neighbor list (unless skipped)
    # ------------------------------------------------------------------
    hop2_list = None
    if not skip_hop2 and network_version not in SKIP_HOP2_DEFAULT:
        print("  Building hop-2 neighbor list...")
        hop2_list = build_hop2_list(hop1_list)
    else:
        reason = (
            "skipped by user flag" if skip_hop2
            else f"network {network_version!r} is too large (use on-the-fly)"
        )
        print(f"  Hop-2 list: {reason}")

    # ------------------------------------------------------------------
    # 5. Save all outputs
    # ------------------------------------------------------------------
    print("  Saving to disk...")
    save_network_data(
        network_version, attrs, hop1_list, hop2_list, data_dir=data_dir
    )

    # ------------------------------------------------------------------
    # 6. For simulated networks: generate and save covariates Z1, Z2, Z3
    # ------------------------------------------------------------------
    if network_version in SIMULATED_NETWORK_CONFIG:
        print("  Generating correlated covariates Z1, Z2, Z3...")
        covs = generate_covariates(attrs["a_degree"])
        out_dir = os.path.join(data_dir, network_version)
        for name, values in covs.items():
            np.savetxt(os.path.join(out_dir, f"{name}.csv"), values)
        print(f"  Saved Z1, Z2, Z3 to {out_dir}/")

    # ------------------------------------------------------------------
    # 7. Save summary statistics (Table 1 inputs)
    # ------------------------------------------------------------------
    import pandas as pd
    summary = {
        "network":              network_version,
        "population_size":      len(attrs["node_list"]),
        "num_edges":            NW.number_of_edges(),
        "avg_degree":           float(attrs["a_degree"].mean()),
        "avg_clustering":       float(attrs["a_cluster_coef"].mean()),
        "avg_eigen_centrality": float(attrs["a_eigen_centrality"].mean()),
        "density":              nx.density(NW),
        "avg_neighbor_degree":  float(attrs["a_nb_degree"].mean()),
    }
    summary_df = pd.DataFrame([summary])
    out_dir = os.path.join(data_dir, network_version)
    summary_df.to_csv(os.path.join(out_dir, "network_summary.csv"), index=False)

    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s")
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Prepare all six population networks for the replication pipeline."
    )
    parser.add_argument(
        "--networks",
        nargs="+",
        default=["all"],
        help=(
            "Which networks to process. Use 'all' (default) or list specific "
            "version strings, e.g. --networks lastfm twitch"
        ),
    )
    parser.add_argument(
        "--data_dir",
        default="./data",
        help="Root data directory. Default: ./data",
    )
    parser.add_argument(
        "--skip_hop2",
        action="store_true",
        help="Skip building hop-2 neighbor lists (faster; needed for Appendix F only).",
    )
    args = parser.parse_args()

    # Determine which networks to process
    all_networks = list(NETWORK_CONFIG.keys()) + list(SIMULATED_NETWORK_CONFIG.keys())
    if args.networks == ["all"]:
        networks_to_run = all_networks
    else:
        invalid = [n for n in args.networks if n not in all_networks]
        if invalid:
            print(f"ERROR: Unknown network(s): {invalid}")
            print(f"Valid options: {all_networks}")
            sys.exit(1)
        networks_to_run = args.networks

    print(f"Networks to process: {networks_to_run}")
    print(f"Data directory:      {os.path.abspath(args.data_dir)}")
    print(f"Skip hop-2 lists:    {args.skip_hop2}")

    # Check that real-world data files exist before starting
    missing = []
    for nv in networks_to_run:
        if nv in NETWORK_CONFIG:
            cfg = NETWORK_CONFIG[nv]
            edge_path = os.path.join(args.data_dir, nv, cfg["edge_file"])
            if not os.path.exists(edge_path):
                missing.append(edge_path)
    if missing:
        print("\nERROR: Missing data files (see data/README.md for download instructions):")
        for p in missing:
            print(f"  {p}")
        sys.exit(1)

    # Run
    summaries = []
    for nv in networks_to_run:
        summary = prepare_one_network(nv, args.data_dir, skip_hop2=args.skip_hop2)
        summaries.append(summary)

    # Save combined summary (Table 1 in paper)
    import pandas as pd
    combined = pd.DataFrame(summaries)
    out_path = os.path.join(args.data_dir, "table1_network_summary.csv")
    combined.to_csv(out_path, index=False)
    print(f"\nAll done. Combined network summary saved to {out_path}")
    print(combined.to_string(index=False))


if __name__ == "__main__":
    main()
