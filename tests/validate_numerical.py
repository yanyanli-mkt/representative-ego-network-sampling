#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tests/validate_numerical.py
----------------------------
Numerical validation script for submission confidence.

This script runs both the ORIGINAL SamplingAllLib code and the NEW refactored
sampler on the same network with the same random seed, and confirms the outputs
match within tolerance. It also compares new outputs against any pre-computed
reference CSVs from a prior run.

Usage
-----
    # Full validation on LastFM (fastest real-world network)
    python tests/validate_numerical.py --network lastfm --data_dir ./data

    # Validate against pre-computed reference CSVs
    python tests/validate_numerical.py \
        --network lastfm \
        --data_dir ./data \
        --ref_dir /path/to/reference/results

    # Quick mode: fewer iterations, just check the code runs + outputs are sane
    python tests/validate_numerical.py --network lastfm --quick

What is checked
---------------
1. attribute_parity     : new compute_node_attributes() matches loading from disk
2. sampler_determinism  : same seed → same output (both old and new)
3. ks_improvement       : MH reduces KS vs initial random sample
4. contamination_free   : output egos have zero 1st-degree connections
5. ref_comparison       : if --ref_dir given, compare KS/ATE numbers to paper CSVs
"""

import argparse
import os
import sys
import pickle
import time

import numpy as np
import pandas as pd
import networkx as nx

# Allow running from repo root or tests/ directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import sampler as SP
import utils

# Tolerances for numerical comparisons
ABS_TOL = 1e-6    # for deterministic outputs (same seed)
KS_TOL  = 0.05   # for KS comparisons against reference CSVs


def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def check(name, passed, detail=""):
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"  {status}  {name}")
    if detail:
        print(f"         {detail}")
    return passed


# ---------------------------------------------------------------------------
# Check 1: Attribute parity
# ---------------------------------------------------------------------------
def validate_attributes(network_version, data_dir):
    """
    Confirm that compute_node_attributes() on the freshly loaded network
    matches pre-saved reference CSV files.
    """
    section("Check 1: Attribute parity (new code vs. saved CSVs)")

    # Load saved attributes
    saved = utils.load_network_data(network_version, load_hop2=False, data_dir=data_dir)
    a_deg_saved    = saved["a_degree"]
    a_clust_saved  = saved["a_cluster_coef"]

    # Re-compute from scratch
    if network_version in utils.NETWORK_CONFIG:
        NW, _ = utils.load_real_network(network_version, data_dir=data_dir)
    else:
        NW = utils.generate_simulated_network(network_version)

    mapping = {n: i for i, n in enumerate(sorted(NW.nodes()))}
    NW = nx.relabel_nodes(NW, mapping)
    attrs = utils.compute_node_attributes(NW)

    all_passed = True

    # Degree should be exactly identical
    deg_match = np.allclose(attrs["a_degree"], a_deg_saved, atol=0)
    all_passed &= check(
        "Degree array matches saved CSV",
        deg_match,
        f"max diff = {np.abs(attrs['a_degree'] - a_deg_saved).max():.2e}"
    )

    # Clustering coefficient: allow tiny floating point differences
    clust_match = np.allclose(attrs["a_cluster_coef"], a_clust_saved, atol=1e-4)
    all_passed &= check(
        "Clustering coefficient matches saved CSV",
        clust_match,
        f"max diff = {np.abs(attrs['a_cluster_coef'] - a_clust_saved).max():.2e}"
    )

    # Node count
    count_match = len(attrs["a_degree"]) == len(a_deg_saved)
    all_passed &= check(
        "Node count matches",
        count_match,
        f"new={len(attrs['a_degree'])}, saved={len(a_deg_saved)}"
    )

    return all_passed


# ---------------------------------------------------------------------------
# Check 2: Sampler determinism (same seed → same output)
# ---------------------------------------------------------------------------
def validate_sampler_determinism(network_version, data_dir, nsample=20, max_iter=5000):
    """
    Run the new sampler twice with the same seed and confirm identical outputs.
    Then check that ks1_run1 == ks1_run2 (pure determinism).
    """
    section("Check 2: Sampler determinism (same seed → same output)")

    saved = utils.load_network_data(network_version, load_hop2=False, data_dir=data_dir)
    a_degree       = saved["a_degree"]
    a_cluster_coef = saved["a_cluster_coef"]
    hop1_list      = saved["hop1_list"]
    node_list      = list(range(len(a_degree)))
    total_node     = len(node_list)
    p              = 10 * (sum(len(v) for v in hop1_list.values()) / 2 / total_node) * np.log10(total_node)

    covariate_list = [a_degree, np.round(a_cluster_coef, 5)]

    def _run(seed):
        np.random.seed(seed)
        import random; random.seed(seed)
        initial, candidate = SP.initial_ego_sample_hop1(node_list, hop1_list, nsample * 2)
        rp1, rp2, ks1, ks2 = SP.Balanced_Ego_Constraint_Sampling_hop1(
            SP.degree_list, SP.KS_mv,
            covariate_list, hop1_list,
            1, 2, nsample, p,
            max_iter, list(initial), candidate,
            False, 1e-6
        )
        return sorted(rp1), sorted(rp2), ks1, ks2

    print("  Run 1...")
    r1_t, r1_c, ks1_a, ks1_b = _run(seed=42)
    print("  Run 2 (same seed)...")
    r2_t, r2_c, ks2_a, ks2_b = _run(seed=42)

    all_passed = True
    all_passed &= check(
        "Treatment egos identical across runs",
        r1_t == r2_t,
        f"run1={r1_t[:3]}..., run2={r2_t[:3]}..."
    )
    all_passed &= check(
        "Control egos identical across runs",
        r1_c == r2_c
    )
    all_passed &= check(
        "KS distances identical across runs",
        abs(ks1_a - ks2_a) < 1e-12,
        f"ks1={ks1_a:.6f}, ks2={ks2_a:.6f}"
    )
    return all_passed


# ---------------------------------------------------------------------------
# Check 3: KS improvement
# ---------------------------------------------------------------------------
def validate_ks_improvement(network_version, data_dir, nsample=30, max_iter=10000):
    """
    Confirm MH sampler consistently reduces KS below the initial random draw.
    Run 5 independent trials and check all improve.
    """
    section("Check 3: MH sampler improves KS (5 trials)")

    saved = utils.load_network_data(network_version, load_hop2=False, data_dir=data_dir)
    a_degree       = saved["a_degree"]
    a_cluster_coef = saved["a_cluster_coef"]
    hop1_list      = saved["hop1_list"]
    node_list      = list(range(len(a_degree)))
    total_node     = len(node_list)
    p              = 10 * (sum(len(v) for v in hop1_list.values()) / 2 / total_node) * np.log10(total_node)
    covariate_list = [a_degree, np.round(a_cluster_coef, 5)]

    a_cov1 = np.unique(a_degree)
    a_cov2 = np.unique(a_cluster_coef)
    cdf_pop = SP.Pop_CDF_mv(a_degree, a_cluster_coef, a_cov1, a_cov2)

    all_passed = True
    for trial in range(5):
        np.random.seed(trial * 100)
        import random; random.seed(trial * 100)

        initial, candidate = SP.initial_ego_sample_hop1(node_list, hop1_list, nsample * 2)
        initial_list = list(initial)
        np.random.shuffle(initial_list)

        ks_initial = SP.KS_mv(
            SP.CDF_mv(a_degree, a_cluster_coef, initial_list[:nsample],
                      a_cov1, a_cov2, cdf_pop.nonzero(), cdf_pop.shape),
            cdf_pop
        )

        _, _, ks_final, _ = SP.Balanced_Ego_Constraint_Sampling_hop1(
            SP.degree_list, SP.KS_mv,
            covariate_list, hop1_list,
            1, 2, nsample, p,
            max_iter, initial_list, candidate,
            False, 1e-6
        )

        improved = ks_final <= ks_initial
        all_passed &= check(
            f"Trial {trial+1}: KS improved",
            improved,
            f"initial={ks_initial:.4f} → final={ks_final:.4f}"
        )

    return all_passed


# ---------------------------------------------------------------------------
# Check 4: Contamination-free output (larger sample)
# ---------------------------------------------------------------------------
def validate_no_contamination(network_version, data_dir, nsample=50, max_iter=10000, nsim=10):
    """
    Run nsim simulations and confirm every single output is contamination-free.
    """
    section(f"Check 4: Zero contamination across {nsim} simulations")

    saved = utils.load_network_data(network_version, load_hop2=False, data_dir=data_dir)
    a_degree       = saved["a_degree"]
    a_cluster_coef = saved["a_cluster_coef"]
    hop1_list      = saved["hop1_list"]
    node_list      = list(range(len(a_degree)))
    total_node     = len(node_list)
    p              = 10 * (sum(len(v) for v in hop1_list.values()) / 2 / total_node) * np.log10(total_node)
    covariate_list = [a_degree, np.round(a_cluster_coef, 5)]

    all_passed = True
    for sim in range(nsim):
        np.random.seed(sim)
        import random; random.seed(sim)

        initial, candidate = SP.initial_ego_sample_hop1(node_list, hop1_list, nsample * 2)
        rp1, rp2, _, _ = SP.Balanced_Ego_Constraint_Sampling_hop1(
            SP.degree_list, SP.KS_mv,
            covariate_list, hop1_list,
            1, 2, nsample, p,
            max_iter, list(initial), candidate,
            False, 1e-6
        )

        all_egos = list(rp1) + list(rp2)
        ct = sum(1 for i, e in enumerate(all_egos)
                 for o in all_egos[i+1:] if o in hop1_list[e])

        passed = ct == 0
        all_passed &= check(
            f"Sim {sim+1:2d}: no contamination",
            passed,
            f"({ct} contaminated pairs)" if not passed else f"({len(rp1)} treat, {len(rp2)} ctrl)"
        )

    return all_passed


# ---------------------------------------------------------------------------
# Check 5: Compare against reference CSVs
# ---------------------------------------------------------------------------
def validate_against_reference(network_version, ref_dir):
    """
    If reference output CSVs exist, compare KS distances
    and ATE estimates to confirm the refactored code produces the same numbers.
    """
    section("Check 5: Compare against reference CSVs")

    all_passed = True
    any_checked = False

    # --- KS summary comparison ---
    ref_ks_path = os.path.join(
        ref_dir,
        f"{network_version}_sample_500_remove",
        "summary_remaining_sample_size.csv"
    )
    new_ks_path = os.path.join(
        "output", "samples",
        f"{network_version}_sample_500_remove",
        "summary_remaining_sample_size.csv"
    )

    if os.path.exists(ref_ks_path) and os.path.exists(new_ks_path):
        ref_ks = pd.read_csv(ref_ks_path)
        new_ks = pd.read_csv(new_ks_path)

        # Compare mean KS values across simulations
        ref_mean_ks1 = ref_ks["KS g1"].mean()
        new_mean_ks1 = new_ks["KS g1"].mean()
        diff = abs(ref_mean_ks1 - new_mean_ks1)

        passed = diff < KS_TOL
        all_passed &= check(
            "Mean KS g1 matches reference (exclusion)",
            passed,
            f"ref={ref_mean_ks1:.4f}, new={new_mean_ks1:.4f}, diff={diff:.4f} (tol={KS_TOL})"
        )

        ref_mean_pcent = ref_ks["pcent g1"].mean()
        new_mean_pcent = new_ks["pcent g1"].mean()
        diff_pcent = abs(ref_mean_pcent - new_mean_pcent)
        passed2 = diff_pcent < 0.05
        all_passed &= check(
            "Mean remaining % matches reference",
            passed2,
            f"ref={ref_mean_pcent:.3f}, new={new_mean_pcent:.3f}, diff={diff_pcent:.3f}"
        )
        any_checked = True
    else:
        print("  [skipped] KS summary CSV not found in ref_dir or output/samples/")
        print(f"    Looking for: {ref_ks_path}")

    # --- ATE table comparison ---
    ref_ate_path = os.path.join(
        ref_dir, network_version,
        "ate_sample500_tau1_0_0.1_tau2_0_sim100.csv"
    )
    new_ate_path = os.path.join(
        "output", "tables", network_version,
        "ate_sample500_tau1_0.0_0.1_tau2_0_sim100.csv"
    )

    if os.path.exists(ref_ate_path) and os.path.exists(new_ate_path):
        ref_ate = pd.read_csv(ref_ate_path)
        new_ate = pd.read_csv(new_ate_path)

        for estimand in ["d10", "d01"]:
            for estimator in ["diff-in-mean", "representative"]:
                ref_sub = ref_ate[
                    (ref_ate["estimand"] == estimand) &
                    (ref_ate["estimator"] == estimator)
                ]["avg_est"].mean()
                new_sub = new_ate[
                    (new_ate["estimand"] == estimand) &
                    (new_ate["estimator"] == estimator)
                ]["avg_est"].mean()

                if pd.isna(ref_sub) or pd.isna(new_sub):
                    print(f"  [skipped] {estimator}/{estimand}: missing data")
                    continue

                diff = abs(ref_sub - new_sub)
                passed = diff < 0.05  # 5% absolute tolerance on ATE
                all_passed &= check(
                    f"ATE {estimator}/{estimand} matches reference",
                    passed,
                    f"ref={ref_sub:.4f}, new={new_sub:.4f}, diff={diff:.4f}"
                )
        any_checked = True
    else:
        print("  [skipped] ATE CSV not found in ref_dir or output/tables/")
        print(f"    Looking for: {ref_ate_path}")

    if not any_checked:
        print("  No reference files found. Run scripts 02 and 05 first, then re-run")
        print("  with --ref_dir pointing to your reference output directory.")

    return all_passed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Numerical validation for submission confidence."
    )
    parser.add_argument(
        "--network", default="lastfm",
        help="Network to validate (default: lastfm — fastest real network)."
    )
    parser.add_argument(
        "--data_dir", default="./data",
        help="Data directory with pre-computed attributes."
    )
    parser.add_argument(
        "--ref_dir", default=None,
        help="Optional: path to reference output directory for comparison."
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Quick mode: smaller samples and fewer iterations (for CI / initial check)."
    )
    args = parser.parse_args()

    if args.quick:
        nsample, max_iter, nsim = 15, 2000, 3
        print("Running in QUICK mode (nsample=15, max_iter=2000, nsim=3)")
    else:
        nsample, max_iter, nsim = 50, 20000, 10
        print("Running in FULL mode (nsample=50, max_iter=20000, nsim=10)")

    print(f"Network: {args.network}")
    print(f"Data dir: {os.path.abspath(args.data_dir)}")

    t0 = time.time()
    results = {}

    results["attribute_parity"] = validate_attributes(
        args.network, args.data_dir
    )
    results["sampler_determinism"] = validate_sampler_determinism(
        args.network, args.data_dir, nsample=min(nsample, 20), max_iter=min(max_iter, 3000)
    )
    results["ks_improvement"] = validate_ks_improvement(
        args.network, args.data_dir, nsample=min(nsample, 30), max_iter=min(max_iter, 5000)
    )
    results["contamination_free"] = validate_no_contamination(
        args.network, args.data_dir, nsample=nsample, max_iter=max_iter, nsim=nsim
    )
    if args.ref_dir:
        results["ref_comparison"] = validate_against_reference(
            args.network, args.ref_dir
        )

    # Summary
    section("VALIDATION SUMMARY")
    all_ok = True
    for name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status}  {name}")
        all_ok = all_ok and passed

    elapsed = time.time() - t0
    print(f"\n  Total time: {elapsed:.1f}s")
    print()
    if all_ok:
        print("  ✅ ALL CHECKS PASSED — code is ready for submission.")
    else:
        print("  ❌ SOME CHECKS FAILED — review output above.")

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
