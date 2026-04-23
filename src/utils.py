#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
utils.py
--------
Shared utilities for network loading, generation, attribute computation,
and statistical helpers used across all simulation scripts.

Covers
------
- Loading the three real-world networks (LastFM, Twitch, Pokec) from SNAP
- Generating the three simulated power-law cluster networks (Small, Base, Sparse)
- Computing and caching node attributes (degree, clustering coefficient,
  eigenvector centrality, neighbor degree)
- Building and saving hop-1 / hop-2 neighbor dictionaries
- Generating correlated covariates Z1, Z2, Z3 (used in simulated networks)
- Statistical helpers: t-test, confidence interval, power, coverage
"""

import os
import pickle
from collections import defaultdict

import networkx as nx
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.stats import t as t_dist


# ---------------------------------------------------------------------------
# Network configurations
# ---------------------------------------------------------------------------

#: Maps internal network version strings to their data file names.
#: Edge files and (optionally) feature files must be placed under
#: ``data/<network_version>/``.
NETWORK_CONFIG = {
    "lastfm": {
        "edge_file": "lastfm_asia_edges.csv",
        "feature_file": "lastfm_asia_target.csv",
        "sep": ",",
        "header": 0,
    },
    "twitch": {
        "edge_file": "large_twitch_edges.csv",
        "feature_file": "large_twitch_features.csv",
        "sep": ",",
        "header": 0,
    },
    "pokec": {
        "edge_file": "soc-pokec-relationships.txt",
        "feature_file": None,
        "sep": "\t",
        "header": None,
    },
}

#: Simulated network parameters: (N, m, p_triangle, seed).
#: Uses ``nx.powerlaw_cluster_graph(N, m, p, seed)``.
SIMULATED_NETWORK_CONFIG = {
    "simulate_small": dict(n=10_000,  m=3,  p=0.30, seed=1982),
    "simulate_base":  dict(n=100_000, m=10, p=0.85, seed=1982),
    "simulate_sparse": dict(n=100_000, m=2,  p=0.10, seed=1982),
}

#: User attributes included in the response model per network (Web Appendix I).
NETWORK_FEATURES = {
    "lastfm":                           ["artists_followed"],
    "twitch":                         ["views", "life_time", "mature", "affiliation"],
    "pokec":                             ["age", "profile_completion", "gender"],
    "simulate_small": ["Z1", "Z2", "Z3"],
    "simulate_base":  ["Z1", "Z2", "Z3"],
    "simulate_sparse": ["Z1", "Z2", "Z3"],
}


# ---------------------------------------------------------------------------
# Network loading and generation
# ---------------------------------------------------------------------------

def load_real_network(network_version, data_dir="./data"):
    """
    Load a real-world network from edge list CSV/TSV files.

    Parameters
    ----------
    network_version : str
        One of ``"lastfm"``, ``"twitch"``, ``"pokec"``.
    data_dir : str
        Path to the data directory containing ``<network_version>/`` subfolder.

    Returns
    -------
    NW : networkx.Graph
        The population network (undirected, largest connected component).
    features : pd.DataFrame or None
        Node feature table if a feature file exists; else ``None``.
    """
    cfg = NETWORK_CONFIG[network_version]
    edge_path = os.path.join(data_dir, network_version, cfg["edge_file"])

    nw_edge = pd.read_csv(edge_path, sep=cfg["sep"], header=cfg["header"])
    cols = nw_edge.columns
    NW = nx.from_pandas_edgelist(nw_edge, cols[0], cols[1])
    NW = NW.to_undirected()

    features = None
    if cfg["feature_file"] is not None:
        feat_path = os.path.join(data_dir, network_version, cfg["feature_file"])
        features = pd.read_csv(feat_path)

    print(
        f"Loaded {network_version}: "
        f"{NW.number_of_nodes()} nodes, {NW.number_of_edges()} edges"
    )
    return NW, features


def generate_simulated_network(network_version):
    """
    Generate one of the three simulated power-law cluster networks used in
    the paper.

    Parameters
    ----------
    network_version : str
        One of the keys in ``SIMULATED_NETWORK_CONFIG``.

    Returns
    -------
    networkx.Graph
        The generated network.
    """
    cfg = SIMULATED_NETWORK_CONFIG[network_version]
    NW = nx.powerlaw_cluster_graph(**cfg)
    print(
        f"Generated {network_version}: "
        f"{NW.number_of_nodes()} nodes, {NW.number_of_edges()} edges"
    )
    return NW


# ---------------------------------------------------------------------------
# Node attribute computation
# ---------------------------------------------------------------------------

def compute_node_attributes(NW):
    """
    Compute degree, clustering coefficient, eigenvector centrality, and
    average neighbor degree for all nodes in the network.

    Parameters
    ----------
    NW : networkx.Graph

    Returns
    -------
    dict with keys:
        ``a_degree``          : np.ndarray, shape (N,)
        ``a_cluster_coef``    : np.ndarray, shape (N,)  (full precision)
        ``a_eigen_centrality``: np.ndarray, shape (N,)
        ``a_nb_degree``       : np.ndarray, shape (N,)
        ``node_list``         : np.ndarray, sorted node indices
    """
    node_list = np.array(sorted(NW.nodes()))
    N = len(node_list)

    # Degree
    deg_dict = dict(NW.degree())
    a_degree = np.array([deg_dict[n] for n in node_list])

    # Clustering coefficient — saved at full precision.
    # Note: the MH sampler rounds to 5 d.p. at call time via
    # covariate_list = [a_degree, np.round(a_cluster_coef, 5)]
    # to limit unique CDF values, but the saved file keeps full precision.
    clust_dict = nx.clustering(NW)
    a_cluster_coef = np.array([clust_dict[n] for n in node_list])

    # Eigenvector centrality
    try:
        eigen_dict = nx.eigenvector_centrality_numpy(NW)
    except Exception:
        eigen_dict = nx.eigenvector_centrality(NW, max_iter=1000, tol=1e-6)
    a_eigen_centrality = np.array([eigen_dict[n] for n in node_list])

    # Average neighbor degree
    avg_nb_deg = nx.average_neighbor_degree(NW)
    a_nb_degree = np.array([avg_nb_deg[n] for n in node_list])

    return {
        "a_degree": a_degree,
        "a_cluster_coef": a_cluster_coef,
        "a_eigen_centrality": a_eigen_centrality,
        "a_nb_degree": a_nb_degree,
        "node_list": node_list,
    }


def build_hop1_list(NW, node_list):
    """
    Build a dict mapping each node to its set of 1st-degree neighbors
    (using internal 0-based indices matching ``node_list``).

    Parameters
    ----------
    NW : networkx.Graph
    node_list : np.ndarray
        Sorted node indices as returned by ``compute_node_attributes``.

    Returns
    -------
    dict : {node_index: set of neighbor indices}
    """
    node_to_idx = {n: i for i, n in enumerate(node_list)}
    hop1_list = {}
    for n in node_list:
        idx = node_to_idx[n]
        hop1_list[idx] = set(node_to_idx[nb] for nb in NW.neighbors(n))
    return hop1_list


def build_hop2_list(hop1_list):
    """
    Build a dict mapping each node to its set of 2nd-degree neighbors
    (friends-of-friends, excluding 1st-degree neighbors and self).

    Parameters
    ----------
    hop1_list : dict
        Output of ``build_hop1_list``.

    Returns
    -------
    dict : {node_index: set of 2nd-degree neighbor indices}

    Notes
    -----
    For very large networks (e.g. Pokec with 1.6M nodes) this can require
    substantial memory. Consider computing on-the-fly with
    ``sampler.get_one_hop2_list`` if memory is a concern.
    """
    hop2_list = {}
    for node, neighbors in hop1_list.items():
        hop2 = set()
        for nb in neighbors:
            hop2.update(hop1_list[nb])
        hop2 -= neighbors
        hop2.discard(node)
        hop2_list[node] = hop2
    return hop2_list


def save_network_data(network_version, attrs, hop1_list, hop2_list=None,
                      data_dir="./data"):
    """
    Save pre-computed network attributes and neighbor lists to disk so they
    can be loaded quickly in simulation scripts without re-computing.

    Files saved under ``data/<network_version>/``:
        ``a_degree.csv``, ``a_cluster_coef.csv``, ``a_eigen_centrality.csv``,
        ``a_nb_degree.csv``, ``hop1_list.pkl``, (optionally) ``hop2_list.pkl``.

    Parameters
    ----------
    network_version : str
    attrs : dict
        Output of ``compute_node_attributes``.
    hop1_list : dict
    hop2_list : dict or None
    data_dir : str
    """
    out_dir = os.path.join(data_dir, network_version)
    os.makedirs(out_dir, exist_ok=True)

    np.savetxt(os.path.join(out_dir, "a_degree.csv"),          attrs["a_degree"])
    np.savetxt(os.path.join(out_dir, "a_cluster_coef.csv"),    attrs["a_cluster_coef"])
    np.savetxt(os.path.join(out_dir, "a_eigen_centrality.csv"),attrs["a_eigen_centrality"])
    np.savetxt(os.path.join(out_dir, "a_nb_degree.csv"),       attrs["a_nb_degree"])

    with open(os.path.join(out_dir, "hop1_list.pkl"), "wb") as f:
        pickle.dump(hop1_list, f)

    if hop2_list is not None:
        with open(os.path.join(out_dir, "hop2_list.pkl"), "wb") as f:
            pickle.dump(hop2_list, f)

    print(f"Saved network data for {network_version} to {out_dir}/")


def load_network_data(network_version, load_hop2=False, data_dir="./data"):
    """
    Load pre-computed network attributes and neighbor lists from disk.

    Parameters
    ----------
    network_version : str
    load_hop2 : bool
        Whether to also load the hop-2 neighbor list.
    data_dir : str

    Returns
    -------
    dict with keys:
        ``a_degree``, ``a_cluster_coef``, ``a_eigen_centrality``,
        ``a_nb_degree``, ``hop1_list``, and (if ``load_hop2``) ``hop2_list``.
    """
    d = os.path.join(data_dir, network_version)

    result = {
        "a_degree":           np.genfromtxt(os.path.join(d, "a_degree.csv")),
        "a_cluster_coef":     np.genfromtxt(os.path.join(d, "a_cluster_coef.csv")),
        "a_eigen_centrality": np.genfromtxt(os.path.join(d, "a_eigen_centrality.csv")),
        "a_nb_degree":        np.genfromtxt(os.path.join(d, "a_nb_degree.csv")),
    }

    with open(os.path.join(d, "hop1_list.pkl"), "rb") as f:
        result["hop1_list"] = pickle.load(f)

    if load_hop2:
        with open(os.path.join(d, "hop2_list.pkl"), "rb") as f:
            result["hop2_list"] = pickle.load(f)

    return result


# ---------------------------------------------------------------------------
# Covariate generation for simulated networks
# ---------------------------------------------------------------------------

def generate_covariates(a_degree, seed_bernoulli=0, seed_z1=2012, seed_z2=1982):
    """
    Generate covariates Z1, Z2, Z3 for simulated networks (paper footnote 6).

    - Z1: correlated with degree (rho = -0.7), seed=2012, scaled to [0, 10]
    - Z2: correlated with degree (rho = +0.3), seed=1982, scaled to [0, 10]
    - Z3: independent Bernoulli(0.5),          seed=0

    Parameters
    ----------
    a_degree      : np.ndarray, shape (N,)
    seed_bernoulli: int  (default 0)     — seed for Z3 Bernoulli
    seed_z1       : int  (default 2012)  — seed for Z1 (rho=-0.7)
    seed_z2       : int  (default 1982)  — seed for Z2 (rho=+0.3)

    Returns
    -------
    dict with keys "Z1", "Z2", "Z3", each np.ndarray of shape (N,).
    """
    from scipy.stats import zscore as _zscore
    N       = len(a_degree)
    deg_std = _zscore(a_degree)

    def _correlated(rho, seed):
        np.random.seed(seed)
        rand_std = np.random.randn(N)
        L        = np.linalg.cholesky([[1, rho], [rho, 1]])
        corr     = np.dot(L, np.vstack([deg_std, rand_std]))
        z        = corr[1]
        return (z - z.min()) / (z.max() - z.min()) * 10

    Z1 = _correlated(rho=-0.7, seed=seed_z1)
    Z2 = _correlated(rho=0.3,  seed=seed_z2)
    np.random.seed(seed_bernoulli)
    Z3 = np.random.binomial(1, 0.5, N).astype(float)

    return {"Z1": Z1, "Z2": Z2, "Z3": Z3}


# ---------------------------------------------------------------------------
# Scaling exponent p
# ---------------------------------------------------------------------------

def compute_p(NW):
    """
    Compute the scaling exponent p for the MH target distribution:
        p = 10 * (edges / nodes) * log10(nodes)

    This rewards representative samples more aggressively in denser or
    larger networks. See Hübler et al. (2008) and Section 3.2 of the paper.

    Parameters
    ----------
    NW : networkx.Graph

    Returns
    -------
    float
    """
    N = NW.number_of_nodes()
    k = NW.number_of_edges()
    return 10 * (k / N) * np.log10(N)


# ---------------------------------------------------------------------------
# Statistical inference helpers
# ---------------------------------------------------------------------------

def my_t_test(y1, y0, true_effect):
    """
    Two-sample t-test for treatment effect estimation, excluding NaN values.

    Used to evaluate DTE/ITE estimates across simulation repetitions.

    Parameters
    ----------
    y1 : np.ndarray
        Outcomes in the treatment group (NaN = missing/excluded).
    y0 : np.ndarray
        Outcomes in the control group.
    true_effect : float
        True treatment effect (for coverage computation).

    Returns
    -------
    beta_hat : float
        Estimated average treatment effect (mean difference).
    se : float
        Standard error of the estimate.
    tstat : float
        t-statistic.
    conf : list of [float, float]
        95% confidence interval [lower, upper].
    power : bool
        Whether the CI lower bound exceeds 0 (one-sided power at α = 0.05).
    coverage : bool
        Whether the CI contains ``true_effect``.
    """
    y1 = y1[~np.isnan(y1)]
    y0 = y0[~np.isnan(y0)]
    return _t_test_core(y1, y0, true_effect)


def my_t_test_correction(y1, y0, true_effect):
    """
    Two-sample t-test replacing NaN values with 0 before estimation.

    Used for the exclusion-approach estimator where excluded units are
    imputed as zero rather than dropped.

    Parameters
    ----------
    y1, y0 : np.ndarray
    true_effect : float

    Returns
    -------
    Same six-tuple as ``my_t_test``.
    """
    y1 = y1.copy()
    y0 = y0.copy()
    y1[np.isnan(y1)] = 0.0
    y0[np.isnan(y0)] = 0.0
    return _t_test_core(y1, y0, true_effect)


def _t_test_core(y1, y0, true_effect):
    """Internal: pooled-variance two-sample t-test."""
    n1, n0 = len(y1), len(y0)
    df = n1 + n0 - 2
    beta_hat = y1.mean() - y0.mean()
    sp2 = (
        np.var(y1, ddof=1) * (n1 - 1) + np.var(y0, ddof=1) * (n0 - 1)
    ) / df
    se = np.sqrt(sp2) * np.sqrt(1 / n1 + 1 / n0)
    tstat = beta_hat / se
    cv = t_dist.ppf(1 - 0.05 / 2, df)
    conf = [beta_hat - cv * se, beta_hat + cv * se]
    power = conf[0] > 0
    coverage = conf[0] < true_effect < conf[1]
    return beta_hat, se, tstat, conf, power, coverage
