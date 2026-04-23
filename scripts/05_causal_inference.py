#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
05_causal_inference.py
----------------------
Step 5 of the replication pipeline (Section 5 of the paper).

For each population network, estimates average and conditional average
treatment effects (ATE / CATE) under four estimators:

    1. diff-in-mean (DIM)           — standard estimator on exclusion samples
    2. Horvitz-Thompson (HT)        — IPTW estimator (Web Appendix H)
    3. Hájek (Hajek)                — normalized IPTW estimator (Web Appendix H)
    4. representative               — DIM on representative (RP) samples

For each estimator, records:
    - avg_est, avg_se, sd_est: point estimate, SE, and SD across simulations
    - bias, MAE, MSE, RMSE: accuracy metrics
    - power, coverage: inference quality

Outputs (saved under output/tables/<network_version>/)
-------------------------------------------------------
    ate_sample500_tau1_0_0.1_tau2_0_sim100.csv     ATE results (Figures 5–6)
    cate_sample500_tau1_0_0.1_tau2_0_sim100.csv    CATE results (Figure 7)
    <network>-Ego-Correction-Expo-Mapping-500sample.csv  Exposure probabilities

Usage
-----
    python scripts/05_causal_inference.py --network simulate_base
    python scripts/05_causal_inference.py --network all
    python scripts/05_causal_inference.py --network lastfm --skip_expo_mapping

Notes
-----
- Must be run after 01_prepare_networks.py AND 02_run_sampling.py.
- Exposure mapping (IPTW weights) requires 500,000 Monte Carlo draws per
  network and is saved to disk. Pass --skip_expo_mapping to reuse existing.
- The response model parameters match the paper: beta=0.5, tau = tau1 * degree,
  alpha2=-0.5 (log(Z1), rho=-0.7), alpha3=0.5 (log(Z2), rho=+0.3), alpha4=-0.5 (Z3, Bernoulli).
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp
from itertools import compress
from scipy.stats import zscore

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from sampler import get_hop1_nb_list
from utils import (
    NETWORK_CONFIG,
    SIMULATED_NETWORK_CONFIG,
    load_network_data,
    my_t_test,
    my_t_test_correction,
)

np.seterr(invalid="ignore", divide="ignore")

# ---------------------------------------------------------------------------
# Response model parameters (paper Section 5.1)
# ---------------------------------------------------------------------------
BETA   = 0.5    # spillover coefficient: ITE = beta * DTE
ALPHA0 = 1.0    # baseline intercept
ALPHA1 = 0.5    # log-degree coefficient
ALPHA2 = -0.5   # log(Z1) coefficient (Z1: rho=-0.7)
ALPHA3 =  0.5   # log(Z2) coefficient (Z2: rho=+0.3)
ALPHA4 = -0.5   # Z3 (Bernoulli) coefficient

TAU1_RANGE  = np.linspace(0, 0.1, 11)   # tau1 sweep
TAU2        = 0                           # second-order term (0 in main paper)
NSIM        = 100
NUM_INITIAL_SAMPLE = 500
N_EXPO_DRAWS = 500_000                   # MC draws for exposure probability


# ---------------------------------------------------------------------------
# Network-specific degree group definitions (for CATE, from paper Section 5)
# ---------------------------------------------------------------------------
DEGREE_GROUPS = {
    "simulate_base": {
        "conditions": [
            lambda d: (d >= 1) & (d <= 20),
            lambda d: (d >= 21) & (d <= 50),
            lambda d: (d >= 51) & (d <= 70),
            lambda d: (d >= 71) & (d <= 100),
            lambda d: d >= 101,
        ],
        "labels": ["1-20", "21-50", "51-70", "71-100", ">100"],
    },
    "simulate_sparse": {
        "conditions": [
            lambda d: d == 2,
            lambda d: d == 3,
            lambda d: (d >= 4) & (d <= 5),
            lambda d: d >= 6,
        ],
        "labels": ["2", "3", "4-5", ">5"],
    },
    "simulate_small": {
        "conditions": [
            lambda d: (d >= 2) & (d <= 3),
            lambda d: (d >= 4) & (d <= 5),
            lambda d: (d >= 6) & (d <= 10),
            lambda d: d >= 11,
        ],
        "labels": ["2-3", "4-5", "6-10", ">10"],
    },
    "lastfm": {
        "conditions": [
            lambda d: d <= 3,
            lambda d: (d >= 4) & (d <= 5),
            lambda d: (d >= 6) & (d <= 10),
            lambda d: d >= 11,
        ],
        "labels": ["1-3", "4-5", "6-10", ">10"],
    },
    "twitch": {
        "conditions": [
            lambda d: d <= 10,
            lambda d: (d >= 11) & (d <= 23),
            lambda d: (d >= 24) & (d <= 44),
            lambda d: (d >= 45) & (d <= 92),
            lambda d: d > 92,
        ],
        "labels": ["1-10", "11-23", "24-44", "45-92", ">92"],
    },
    "pokec": {
        "conditions": [
            lambda d: d <= 3,
            lambda d: (d >= 4) & (d <= 9),
            lambda d: (d >= 10) & (d <= 20),
            lambda d: (d >= 21) & (d <= 43),
            lambda d: d >= 44,
        ],
        "labels": ["1-3", "4-9", "10-20", "21-43", ">43"],
    },
}


# ---------------------------------------------------------------------------
# Covariate loading
# ---------------------------------------------------------------------------

def load_covariates(network_version, data_dir, a_degree, total_node):
    """
    Load or generate the network-specific covariates used in the response model.

    For real-world networks: loads raw node features from data files.
    For simulated networks: loads Z1, Z2, Z3 saved by script 01.

    Returns a dict of covariate arrays, each shape (N,).
    """
    if network_version == "lastfm":
        feat_path = os.path.join(data_dir, network_version, "lastfm_asia_target.csv")
        node_feature = pd.read_csv(feat_path)
        x1 = np.array(node_feature["target"].tolist())

        # Number of unique artists followed (from JSON feature file)
        json_path = os.path.join(data_dir, network_version, "lastfm_asia_features.json")
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        x2 = np.array([len(set(v)) if isinstance(v, list) else 0
                        for v in data.values()])
        return {"x1": x1, "x2": x2}

    elif network_version == "twitch":
        feat_path = os.path.join(data_dir, network_version, "large_twitch_features.csv")
        node_feature = pd.read_csv(feat_path)
        return {
            "x1": np.array(node_feature["views"].tolist()),
            "x2": np.array(node_feature["life_time"].tolist()),
            "x3": np.array(node_feature["mature"].tolist()),
            "x4": np.array(node_feature["affiliate"].tolist()),
        }

    elif network_version == "pokec":
        feat_path = os.path.join(data_dir, network_version, "soc-pokec-profiles.txt")
        node_feature = pd.read_csv(feat_path, sep="\t", header=None)
        node_feature = node_feature.iloc[:, 0:9]
        node_feature.columns = [
            "user_id", "public", "completion_percentage", "gender",
            "region", "last_login", "registration", "AGE", "body"
        ]
        x1 = np.array(node_feature["AGE"].tolist())
        x2 = np.array(node_feature["completion_percentage"].tolist()) / 100
        x3 = np.array(node_feature["gender"].tolist())
        return {
            "x1": x1, "x2": x2, "x3": x3,
            "z1": np.nan_to_num(zscore(x1, nan_policy="omit"), nan=0.0),
            "z2": np.nan_to_num(zscore(x2, nan_policy="omit"), nan=0.0),
        }

    else:
        # Simulated networks: load pre-saved Z1, Z2, Z3
        d = os.path.join(data_dir, network_version)
        return {
            "Z1": np.genfromtxt(os.path.join(d, "Z1.csv")),
            "Z2": np.genfromtxt(os.path.join(d, "Z2.csv")),
            "Z3": np.genfromtxt(os.path.join(d, "Z3.csv")),
        }


def build_baseline_outcome(network_version, a_degree, covs, total_node):
    """
    Compute the baseline potential outcome y00 (no treatment) and ego random
    effect for each node, matching the response model in Section 5.1.

    The response model is:
        y_i(0) = alpha0 + alpha1*log(degree) + alpha2*log(Z1) + alpha3*log(Z2) + alpha4*Z3 + epsilon_i
        where Z1: rho=-0.7, Z2: rho=+0.3, Z3: Bernoulli(0.5)

    Uses legacy numpy random state (np.random.seed set at script start).
    Parameters match footnote 6 and Web Appendix I:
        alpha0=1, alpha1=0.5, alpha2=-0.5 (log(Z1), rho=-0.7), alpha3=0.5 (log(Z2), rho=+0.3), alpha4=-0.5 (Z3, Bernoulli).
    """
    if network_version == "lastfm":
        # alpha2=-0.5 applied to zscore(artists_followed)
        y00 = (ALPHA0 + ALPHA1 * np.log(a_degree + 1)
               + ALPHA2 * zscore(covs["x2"])
               + np.random.normal(0, 1, total_node))

    elif network_version == "twitch":
        # alpha3=-0.5 applied to zscore(lifetime)
        y00 = (ALPHA0 + ALPHA1 * np.log(a_degree + 1)
               + ALPHA3 * zscore(covs["x2"])
               + np.random.normal(0, 1, total_node))

    elif network_version == "pokec":
        # alpha3=-0.5 on z1 (age), alpha4=0.2 on z2 (completion_percentage)
        y00 = (ALPHA0 + ALPHA1 * np.log(a_degree)
               + ALPHA3 * covs["z1"] + ALPHA4 * covs["z2"]
               + np.random.normal(0, 1, total_node))

    else:
        # Simulated networks:
        # alpha0 + alpha1*log(degree) + alpha2*log(Z1+1) + alpha3*log(Z2+1) + alpha4*Z3 + N(0,1)
        # Z1: rho=-0.7, Z2: rho=+0.3 (both scaled [0,10]), Z3: Bernoulli(0.5)
        y00 = (ALPHA0 + ALPHA1 * np.log(a_degree)
               + ALPHA2 * np.log(covs["Z1"] + 1)
               + ALPHA3 * np.log(covs["Z2"] + 1)
               + ALPHA4 * covs["Z3"]
               + np.random.normal(0, 1, total_node))

    ego_effect = np.random.normal(0, 1, total_node)
    return y00, ego_effect


# ---------------------------------------------------------------------------
# Exposure mapping (IPTW weights)
# ---------------------------------------------------------------------------

def compute_exposure_mapping(network_version, A, node_list, total_node,
                              nsample, n_draws, output_dir):
    """
    Estimate exposure probabilities via Monte Carlo simulation (Web Appendix H).

    For each node, estimates P(node falls in each of 5 exposure conditions)
    by repeatedly randomizing ego assignment and counting frequencies.

    Exposure conditions (ego perspective):
        d11: treated ego, connected to ≥1 other ego
        d10: treated ego, no other ego neighbors  ← DTE identification
        d0n: non-ego, connected to >1 ego
        d01: non-ego, connected to exactly 1 ego  ← ITE identification
        d00: non-ego, no ego neighbors

    Parameters
    ----------
    n_draws : int
        Number of Monte Carlo draws (500,000 used in paper, per Web Appendix H).

    Returns
    -------
    pd.DataFrame, shape (N, 5), columns ["d11","d10","d0n","d01","d00"]
        Empirical probabilities (rows sum to 1 per node).
    """
    import random as rd
    pi_mapping = np.zeros((total_node, 5))

    for _ in range(n_draws):
        one_d = rd.sample(node_list, nsample)
        d_vec = np.zeros(total_node)
        d_vec[one_d] = 1
        nd_vec = A.dot(d_vec)

        Dmat = np.column_stack([
            (d_vec > 0) & (nd_vec > 0),   # d11
            (d_vec > 0) & (nd_vec == 0),  # d10
            (d_vec == 0) & (nd_vec > 1),  # d0n
            (d_vec == 0) & (nd_vec == 1), # d01
            (d_vec == 0) & (nd_vec == 0), # d00
        ]).astype(float)
        pi_mapping += Dmat

    pi_df = pd.DataFrame(
        pi_mapping, columns=["d11", "d10", "d0n", "d01", "d00"],
        index=node_list
    )

    # Save
    out_path = os.path.join(
        output_dir, f"{network_version}-Ego-Correction-Expo-Mapping-500sample.csv"
    )
    pi_df.to_csv(out_path)
    print(f"  Exposure mapping saved to {out_path}")
    return pi_df


def load_or_compute_exposure_mapping(network_version, A, node_list, total_node,
                                     output_dir, skip=False, nsample=1000):
    """Load exposure mapping from disk, or compute if missing/not skipped."""
    out_path = os.path.join(
        output_dir, f"{network_version}-Ego-Correction-Expo-Mapping-500sample.csv"
    )
    if skip and os.path.exists(out_path):
        print(f"  Loading existing exposure mapping from {out_path}")
        return pd.read_csv(out_path, index_col=0)

    print(f"  Computing exposure mapping ({N_EXPO_DRAWS:,} draws)...")
    return compute_exposure_mapping(
        network_version, A, node_list, total_node,
        nsample, N_EXPO_DRAWS, output_dir
    )


# ---------------------------------------------------------------------------
# Core estimation loop
# ---------------------------------------------------------------------------

def estimate_one_sim(
    sim, network_version, estimator_type,
    ego_treat, ego_ctrl, alter_treat, alter_ctrl,
    y00, ego_effect, tau, A, a_degree, hop1_list,
    expo_mat_T, expo_mat_C,
    total_node, true_adte, true_aite,
    cate_var, n_cate_groups
):
    """
    Run one simulation and return ATE/CATE estimates for all estimators.

    Parameters
    ----------
    estimator_type : str
        One of "exclusion" or "representative".
    ego_treat, ego_ctrl : array of int
        Indices of treated/control egos.
    alter_treat, alter_ctrl : array of int
        Indices of treated/control alters.

    Returns
    -------
    ate_row : dict  (8 values: DTE + ITE for this estimator)
    cate_row : np.ndarray, shape (n_cate_groups,)  CATE by degree group
    """
    N = total_node
    beta = BETA

    # Outcome model
    d_vec = np.zeros(N)
    d_vec[np.concatenate((ego_treat, ego_ctrl))] = 1
    is_treat_ego = np.zeros(N)
    is_treat_ego[ego_treat] = 1

    one_outcome = (y00
                   + tau * is_treat_ego
                   + beta * A.dot(tau * is_treat_ego)
                   + ego_effect * d_vec
                   + A.dot(ego_effect * d_vec))

    # Observed outcome matrices
    obs_T = np.full((N, 2), np.nan)
    obs_T[ego_treat, 0]   = one_outcome[ego_treat]
    obs_T[alter_treat, 1] = one_outcome[alter_treat]

    obs_C = np.full((N, 2), np.nan)
    obs_C[ego_ctrl, 0]   = one_outcome[ego_ctrl]
    obs_C[alter_ctrl, 1] = one_outcome[alter_ctrl]

    # Zero-outcome nodes (ego = 0 outcome → treat as missing)
    obs_T[obs_T == 0] = np.nan
    obs_C[obs_C == 0] = np.nan

    # --- DIM (diff-in-mean) ---
    dte_dim = my_t_test(obs_T[:, 0], obs_C[:, 0], true_adte)
    # ITE: ego-group mean of alter outcomes
    alter_mean_T = _ego_alter_mean(ego_treat, hop1_list, alter_treat, one_outcome, N)
    alter_mean_C = _ego_alter_mean(ego_ctrl,  hop1_list, alter_ctrl,  one_outcome, N)
    ite_dim = my_t_test(alter_mean_T, alter_mean_C, true_aite)

    # --- HT (Horvitz-Thompson) ---
    ht_T = obs_T / expo_mat_T[:, [1, 3]]
    ht_C = obs_C / expo_mat_C[:, [1, 3]]
    ht_T[np.isinf(ht_T)] = np.nan
    ht_C[np.isinf(ht_C)] = np.nan

    dte_ht = my_t_test_correction(ht_T[:, 0], ht_C[:, 0], true_adte)

    # HT ITE
    ht_alter_T = _ego_ht_alter_mean(ego_treat, hop1_list, ht_T[:, 1], a_degree, N)
    ht_alter_C = _ego_ht_alter_mean(ego_ctrl,  hop1_list, ht_C[:, 1], a_degree, N)
    ite_ht = my_t_test(ht_alter_T, ht_alter_C, true_aite)

    # --- Hájek (normalized HT) ---
    w_T = np.where(np.isnan(obs_T), np.nan, 1.0) / expo_mat_T[:, [1, 3]]
    w_C = np.where(np.isnan(obs_C), np.nan, 1.0) / expo_mat_C[:, [1, 3]]
    w_T[np.isinf(w_T)] = np.nan
    w_C[np.isinf(w_C)] = np.nan

    dte_mu = my_t_test_correction(
        ht_T[:, 0] / (np.nansum(w_T[:, 0]) / N),
        ht_C[:, 0] / (np.nansum(w_C[:, 0]) / N),
        true_adte
    )

    mu_alter_T = _ego_hajek_alter_mean(ego_treat, hop1_list, ht_T[:, 1], w_T[:, 1], N)
    mu_alter_C = _ego_hajek_alter_mean(ego_ctrl,  hop1_list, ht_C[:, 1], w_C[:, 1], N)
    ite_mu = my_t_test(mu_alter_T, mu_alter_C, true_aite)

    # Pack results: [est, se, power, coverage] for DTE and ITE
    results = {
        "dim":  [dte_dim[0], dte_dim[1], dte_dim[4], dte_dim[5],
                 ite_dim[0], ite_dim[1], ite_dim[4], ite_dim[5]],
        "ht":   [dte_ht[0],  dte_ht[1],  dte_ht[4],  dte_ht[5],
                 ite_ht[0],  ite_ht[1],  ite_ht[4],  ite_ht[5]],
        "hajek":[dte_mu[0],  dte_mu[1],  dte_mu[4],  dte_mu[5],
                 ite_mu[0],  ite_mu[1],  ite_mu[4],  ite_mu[5]],
    }

    # CATE by degree group (DIM only)
    cate_data = pd.DataFrame({
        "outcome_T": obs_T[:, 0],
        "outcome_C": obs_C[:, 0],
        "cate_group": cate_var,
    })
    cate_agg = cate_data.groupby("cate_group").agg({
        "outcome_T": np.nanmean,
        "outcome_C": np.nanmean,
    })
    cate_est = (cate_agg["outcome_T"] - cate_agg["outcome_C"]).values

    return results, cate_est


def _ego_alter_mean(ego_list, hop1_list, alter_list, outcomes, N):
    """Compute ego-level mean alter outcome (for ITE DIM estimator)."""
    alter_set = set(alter_list)
    result = np.full(N, np.nan)
    for ego in ego_list:
        idx = list(hop1_list[ego] & alter_set)
        if idx:
            result[ego] = np.nanmean(outcomes[idx])
    return result


def _ego_ht_alter_mean(ego_list, hop1_list, ht_alter_outcomes, a_degree, N):
    """Compute ego-level HT-weighted alter mean (for ITE HT estimator).

    Uses all hop-1 neighbors of each ego (not just those in the clean alter
    set). For egos whose neighbors all have NaN HT outcomes, np.nansum returns
    0.0, giving result[ego] = 0.0 / degree = 0.0 (non-NaN). This matches the


    Note: Statistically, egos with zero clean alters should be excluded (NaN),

    """
    result = np.full(N, np.nan)
    for ego in ego_list:
        idx = list(hop1_list[ego])
        result[ego] = np.nansum(ht_alter_outcomes[idx]) / a_degree[ego]
    return result


def _ego_hajek_alter_mean(ego_list, hop1_list, ht_vals, weights, N):
    """Compute ego-level Hájek alter mean (for ITE Hájek estimator)."""
    result = np.full(N, np.nan)
    for ego in ego_list:
        idx = list(hop1_list[ego])
        num = np.nansum(ht_vals[idx])
        den = np.nansum(weights[idx])
        if den > 0:
            result[ego] = num / den
    return result


# ---------------------------------------------------------------------------
# Main simulation loop per network
# ---------------------------------------------------------------------------

def run_causal_inference(network_version, args):
    """Run the full Section 5 simulation for one network."""
    print(f"\n{'='*60}\n  Causal inference: {network_version}\n{'='*60}")
    t0 = time.time()

    # Load network data
    data = load_network_data(network_version, load_hop2=False,
                             data_dir=args.data_dir)
    a_degree       = data["a_degree"]
    a_cluster_coef = data["a_cluster_coef"]
    hop1_list      = data["hop1_list"]

    total_node = len(a_degree)
    N = total_node
    node_list = list(range(total_node))

    # Build sparse adjacency
    rows_, cols_ = [], []
    for node, neighbors in hop1_list.items():
        for nb in neighbors:
            rows_.append(node); cols_.append(nb)
    A = sp.csr_matrix(
        (np.ones(len(rows_)), (rows_, cols_)),
        shape=(total_node, total_node)
    )

    # Covariates and baseline outcome
    covs = load_covariates(network_version, args.data_dir, a_degree, total_node)
    y00, ego_effect = build_baseline_outcome(network_version, a_degree, covs, total_node)

    # Degree groups for CATE
    dg_cfg = DEGREE_GROUPS[network_version]
    cate_var = np.select(
        [cond(a_degree) for cond in dg_cfg["conditions"]],
        list(range(len(dg_cfg["conditions"]))),
        default=0
    )
    n_cate_groups = len(dg_cfg["conditions"])
    degree_group_labels = dg_cfg["labels"]

    # Exposure mapping (IPTW weights)
    os.makedirs(args.output_dir, exist_ok=True)
    expo_df = load_or_compute_exposure_mapping(
        network_version, A, node_list, total_node,
        args.output_dir,
        skip=args.skip_expo_mapping,
        nsample=NUM_INITIAL_SAMPLE * 2
    )
    expo_df["sim_count"] = expo_df.sum(axis=1)
    for col in ["d11", "d10", "d0n", "d01", "d00"]:
        expo_df[col] = expo_df[col] / expo_df["sim_count"]

    expo_mat_T = np.array(expo_df[["d11","d10","d0n","d01","d00"]]) / 2
    expo_mat_C = expo_mat_T.copy()

    # Paths to sampling outputs from script 02
    samples_dir  = args.samples_dir
    rm_dir       = os.path.join(samples_dir,
                                f"{network_version}_sample_{NUM_INITIAL_SAMPLE}_remove")
    rp_dir       = os.path.join(samples_dir,
                                f"{network_version}_sample_match_{NUM_INITIAL_SAMPLE}_ego_loss1")

    rm_summary = pd.read_csv(os.path.join(rm_dir, "summary_remaining_sample_size.csv"))
    rm_sizes   = ((rm_summary["size g1"] + rm_summary["size g2"]) / 2).astype(int)

    # Output tables
    ate_table  = []
    cate_table = []

    # -----------------------------------------------------------------------
    # Outer loop: tau1 values
    # -----------------------------------------------------------------------
    for tau1 in TAU1_RANGE:
        print(f"\n  tau1 = {tau1:.3f}")
        tau      = tau1 * a_degree
        true_adte = tau.mean()
        true_aite = BETA * true_adte

        tau_mean_by_group = (
            pd.DataFrame({"group": cate_var, "tau": tau})
            .groupby("group")["tau"].mean()
            .values
        )

        # Storage arrays: [est, se, power, coverage] × [DTE, ITE]
        rm_dim   = np.full((NSIM, 8), np.nan)
        rm_ht    = np.full((NSIM, 8), np.nan)
        rm_hajek = np.full((NSIM, 8), np.nan)
        rp_dim   = np.full((NSIM, 8), np.nan)

        rm_cate  = np.full((NSIM, n_cate_groups), np.nan)
        rm_cate_ht   = np.full((NSIM, n_cate_groups), np.nan)
        rm_cate_hajek= np.full((NSIM, n_cate_groups), np.nan)
        rp_cate  = np.full((NSIM, n_cate_groups), np.nan)

        # -------------------------------------------------------------------
        # Inner loop: simulations
        # -------------------------------------------------------------------
        for sim in range(NSIM):
            prefix_rm = os.path.join(rm_dir, str(NUM_INITIAL_SAMPLE))
            nsample   = int(rm_sizes.iloc[sim])
            prefix_rp = os.path.join(rp_dir, str(nsample))

            # --- Exclusion samples ---
            try:
                ego_t_rm  = np.genfromtxt(f"{prefix_rm}_ego_node1_sim{sim}.csv").astype(int)
                ego_c_rm  = np.genfromtxt(f"{prefix_rm}_ego_node2_sim{sim}.csv").astype(int)
                alt_t_rm  = np.genfromtxt(f"{prefix_rm}_alter_node1_sim{sim}.csv").astype(int)
                alt_c_rm  = np.genfromtxt(f"{prefix_rm}_alter_node2_sim{sim}.csv").astype(int)
            except Exception:
                continue

            res_rm, cate_rm = estimate_one_sim(
                sim, network_version, "exclusion",
                ego_t_rm, ego_c_rm, alt_t_rm, alt_c_rm,
                y00, ego_effect, tau, A, a_degree, hop1_list,
                expo_mat_T, expo_mat_C,
                total_node, true_adte, true_aite,
                cate_var, n_cate_groups
            )
            rm_dim[sim]   = res_rm["dim"]
            rm_ht[sim]    = res_rm["ht"]
            rm_hajek[sim] = res_rm["hajek"]
            rm_cate[sim]  = cate_rm

            # --- RP samples ---
            try:
                ego_t_rp = np.genfromtxt(f"{prefix_rp}_ego_node1_sim{sim}.csv").astype(int)
                ego_c_rp = np.genfromtxt(f"{prefix_rp}_ego_node2_sim{sim}.csv").astype(int)
            except Exception:
                continue

            # RP alters: all hop-1 neighbors of sampled egos (clean by construction)
            alt_t_rp = np.array(get_hop1_nb_list(list(ego_t_rp), hop1_list))
            alt_c_rp = np.array(get_hop1_nb_list(list(ego_c_rp), hop1_list))

            # Keep only alters connected to exactly one ego (SUTVA)
            d_rp = np.zeros(N)
            d_rp[np.concatenate((ego_t_rp, ego_c_rp))] = 1
            nd_rp = A.dot(d_rp)
            alt_t_rp = np.array(list(compress(alt_t_rp.tolist(), nd_rp[alt_t_rp] == 1)))
            alt_c_rp = np.array(list(compress(alt_c_rp.tolist(), nd_rp[alt_c_rp] == 1)))

            res_rp, cate_rp = estimate_one_sim(
                sim, network_version, "representative",
                ego_t_rp, ego_c_rp, alt_t_rp, alt_c_rp,
                y00, ego_effect, tau, A, a_degree, hop1_list,
                expo_mat_T, expo_mat_C,
                total_node, true_adte, true_aite,
                cate_var, n_cate_groups
            )
            rp_dim[sim]  = res_rp["dim"]
            rp_cate[sim] = cate_rp

        # -------------------------------------------------------------------
        # Assemble ATE output table (matches structure read by Gen_Figure567.R)
        # -------------------------------------------------------------------
        def _ate_rows(est_mat, estimator_name):
            rows = []
            for col_dte, col_ite, estimand in [(0, 4, "d10"), (4, 4, "d01")]:
                # For DTE use cols 0-3, for ITE use cols 4-7
                if estimand == "d10":
                    cols = slice(0, 4)
                    true_v = true_adte
                else:
                    cols = slice(4, 8)
                    true_v = true_aite
                vals = est_mat[:, cols]
                avg_est = np.nanmean(vals[:, 0])
                avg_se  = np.nanmean(vals[:, 1])
                sd_est  = np.nanstd(vals[:, 0])
                mae     = np.nanmean(np.abs(vals[:, 0] - true_v))
                mse     = np.nanmean((vals[:, 0] - true_v) ** 2)
                power   = np.nanmean(vals[:, 2])
                coverage= np.nanmean(vals[:, 3])
                rows.append({
                    "tau1": tau1, "tau2": TAU2, "sample_size": NUM_INITIAL_SAMPLE,
                    "estimator": estimator_name, "estimand": estimand,
                    "true_value": true_v, "avg_est": avg_est, "avg_se": avg_se,
                    "sd_est": sd_est, "bias": avg_est - true_v,
                    "MAE": mae, "mse": mse, "rmse": np.sqrt(mse),
                    "power": power, "coverage": coverage,
                })
            return rows

        ate_table += _ate_rows(rm_dim,   "diff-in-mean")
        ate_table += _ate_rows(rm_ht,    "Horvitz-Thompson")
        ate_table += _ate_rows(rm_hajek, "Hajek")
        ate_table += _ate_rows(rp_dim,   "representative")

        # -------------------------------------------------------------------
        # Assemble CATE output table
        # -------------------------------------------------------------------
        for g_idx, g_label in enumerate(degree_group_labels):
            for est_name, cate_mat in [
                ("diff-in-mean",       rm_cate[:, g_idx]),
                ("Horvitz-Thompson",   rm_cate_ht[:, g_idx]),
                ("Hajek",              rm_cate_hajek[:, g_idx]),
                ("representative",     rp_cate[:, g_idx]),
            ]:
                cate_table.append({
                    "tau1": tau1, "degree_group": g_label,
                    "estimator": est_name,
                    "true_cate": tau_mean_by_group[g_idx],
                    "avg_cate_est": np.nanmean(cate_mat),
                    "sd_cate_est":  np.nanstd(cate_mat),
                    "MAE": np.nanmean(np.abs(cate_mat - tau_mean_by_group[g_idx])),
                    "rmse": np.sqrt(np.nanmean((cate_mat - tau_mean_by_group[g_idx]) ** 2)),
                })

    # Save outputs
    out_dir = os.path.join(args.output_dir, network_version)
    os.makedirs(out_dir, exist_ok=True)

    ate_df = pd.DataFrame(ate_table)
    ate_path = os.path.join(
        out_dir,
        f"ate_sample500_tau1_{TAU1_RANGE[0]}_{TAU1_RANGE[-1]}_tau2_{TAU2}_sim{NSIM}.csv"
    )
    ate_df.to_csv(ate_path, index=False)
    print(f"\n  ATE table saved to {ate_path}")

    cate_df = pd.DataFrame(cate_table)
    cate_path = os.path.join(
        out_dir,
        f"cate_sample500_tau1_{TAU1_RANGE[0]}_{TAU1_RANGE[-1]}_tau2_{TAU2}_sim{NSIM}.csv"
    )
    cate_df.to_csv(cate_path, index=False)
    print(f"  CATE table saved to {cate_path}")

    print(f"\n  Done in {(time.time()-t0)/60:.1f} min")


def main():
    all_networks = list(NETWORK_CONFIG.keys()) + list(SIMULATED_NETWORK_CONFIG.keys())

    parser = argparse.ArgumentParser(
        description="Causal inference simulation (Section 5) for all networks."
    )
    parser.add_argument("--network", nargs="+", default=["all"],
                        help="Network(s) to process, or 'all'.")
    parser.add_argument("--data_dir",    default="./data")
    parser.add_argument("--samples_dir", default="./output/samples",
                        help="Directory containing sampled node CSVs from script 02.")
    parser.add_argument("--output_dir",  default="./output/tables")
    parser.add_argument("--skip_expo_mapping", action="store_true",
                        help="Reuse existing exposure mapping CSV if available.")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    networks = all_networks if args.network == ["all"] else args.network
    for nv in networks:
        run_causal_inference(nv, args)

    print("\nAll networks complete.")


if __name__ == "__main__":
    main()
