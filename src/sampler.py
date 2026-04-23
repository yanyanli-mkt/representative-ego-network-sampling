#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sampler.py
----------
Core Metropolis-Hastings ego network sampling algorithm for peer encouragement designs.

This module implements the representative ego network sampling method described in:

    "A Representative Sampling Method for Peer Encouragement Designs
     in Network Experiments"

The main contribution is a constrained Metropolis-Hastings (MH) algorithm that
simultaneously draws treatment and control ego network samples from a population
network such that:
  (1) No two sampled egos are contaminated (first- or second-degree contamination
      is controlled via the `hop` parameter).
  (2) The joint distribution of degree and clustering coefficient in each sample
      converges to that of the population (minimizing KS distance).

Primary entry points
--------------------
- ``initial_ego_sample_hop1``  : greedy initialization under 1st-degree constraint
- ``initial_ego_sample_hop2``  : greedy initialization under 2nd-degree constraint
- ``Balanced_Ego_Constraint_Sampling_hop1`` : MH sampler (1st-degree, KS distance) [**main method**]
- ``Balanced_Ego_Constraint_Sampling_hop2`` : MH sampler (2nd-degree, KS distance)
- ``Balanced_Ego_Constraint_Sampling_hop1_KL`` : MH sampler (1st-degree, KL divergence) [Appendix A benchmark]
- ``Balanced_Ego_Constraint_Sampling_hop1_KS_linear`` : MH sampler (1st-degree, linear KS) [Appendix A benchmark]

Dependencies
------------
numpy, scipy, networkx (for ``initial_ego_sample`` only)
"""

from collections import Counter
from itertools import chain
import random as rd

import numpy as np
from scipy import stats
from scipy.sparse import coo_matrix, csr_matrix


# ---------------------------------------------------------------------------
# Distribution utilities
# ---------------------------------------------------------------------------

def EPDF_table(table):
    """
    Normalize a Counter or frequency dict into an empirical probability
    distribution (EPDF).

    Parameters
    ----------
    table : dict
        Raw frequency counts, e.g. ``Counter(degree_array)``.

    Returns
    -------
    dict
        Keys are the same as ``table``; values sum to 1.0.
    """
    total = sum(table.values(), 0.0)
    return {k: v / total for k, v in table.items()}


def EPDF_mv(a_degree, a_cluster_coef, node_list):
    """
    Compute the joint empirical probability distribution of (degree,
    clustering coefficient) for a given node list.

    Parameters
    ----------
    a_degree : np.ndarray, shape (N,)
        Degree array for all population nodes.
    a_cluster_coef : np.ndarray, shape (N,)
        Clustering coefficient array for all population nodes.
    node_list : list or array-like
        Indices of the nodes to summarize.

    Returns
    -------
    dict
        Maps (degree, cluster_coef) tuples to their joint probability.
    """
    pop_table = Counter(zip(a_degree[node_list], a_cluster_coef[node_list]))
    total = len(node_list)
    return {key: value / total for key, value in pop_table.items()}


def Pop_CDF_mv(a_degree, a_cluster_coef, unique_degrees, unique_cluster_coefs):
    """
    Compute the joint cumulative distribution function (CDF) of degree and
    clustering coefficient for the *entire* population. Called once per
    simulation run; the result is reused across all MH iterations.

    Parameters
    ----------
    a_degree : np.ndarray, shape (N,)
        Degree array for all N population nodes.
    a_cluster_coef : np.ndarray, shape (N,)
        Clustering coefficient array (pre-rounded to reduce unique values).
    unique_degrees : np.ndarray
        Sorted unique degree values; typically ``np.unique(a_degree)``.
    unique_cluster_coefs : np.ndarray
        Sorted unique clustering coefficient values.

    Returns
    -------
    scipy.sparse.csr_matrix
        Sparse joint CDF matrix of shape
        ``(len(unique_degrees), len(unique_cluster_coefs))``.
        Entry [i, j] = P(degree ≤ unique_degrees[i]  AND
                         cluster_coef ≤ unique_cluster_coefs[j]).
    """
    deg_idx = np.searchsorted(unique_degrees, a_degree)
    coef_idx = np.searchsorted(unique_cluster_coefs, a_cluster_coef)

    values = np.ones(len(a_degree))
    hist2d = coo_matrix(
        (values, (deg_idx, coef_idx)),
        shape=(len(unique_degrees), len(unique_cluster_coefs))
    )
    hist2d = (hist2d / len(a_degree)).tocsr()

    cdf_dense = hist2d.toarray()
    cdf_dense = np.cumsum(cdf_dense, axis=1)
    cdf_dense = np.cumsum(cdf_dense, axis=0)

    return csr_matrix(
        (cdf_dense[hist2d.nonzero()], hist2d.nonzero()),
        shape=hist2d.shape
    )


def CDF_mv(a_degree, a_cluster_coef, node_list,
           unique_degrees, unique_cluster_coefs,
           pop_nonzero, pop_shape):
    """
    Compute the joint CDF of (degree, clustering coefficient) for a *sample*
    node list, evaluated only at the support points of the population CDF.
    This is the fast per-iteration version; ``Pop_CDF_mv`` computes the
    population reference once up front.

    Parameters
    ----------
    a_degree : np.ndarray, shape (N,)
    a_cluster_coef : np.ndarray, shape (N,)
    node_list : list
        Indices of the sampled egos.
    unique_degrees : np.ndarray
        Same sorted unique degrees used in ``Pop_CDF_mv``.
    unique_cluster_coefs : np.ndarray
        Same sorted unique cluster coefs used in ``Pop_CDF_mv``.
    pop_nonzero : tuple of arrays
        ``cdf_mat_pop.nonzero()`` — row/col indices of nonzero pop CDF entries.
    pop_shape : tuple
        Shape of the population CDF matrix.

    Returns
    -------
    scipy.sparse.csr_matrix
        Sample joint CDF evaluated at population support points.
    """
    deg_idx = np.searchsorted(unique_degrees, a_degree[node_list])
    coef_idx = np.searchsorted(unique_cluster_coefs, a_cluster_coef[node_list])

    values = np.ones(len(node_list))
    hist2d = coo_matrix(
        (values, (deg_idx, coef_idx)),
        shape=(len(unique_degrees), len(unique_cluster_coefs))
    )
    hist2d = (hist2d / len(node_list)).tocsr()

    cdf_dense = hist2d.toarray()
    cdf_dense = np.cumsum(cdf_dense, axis=1)
    cdf_dense = np.cumsum(cdf_dense, axis=0)

    return csr_matrix(
        (cdf_dense[pop_nonzero], pop_nonzero),
        shape=pop_shape
    )


# ---------------------------------------------------------------------------
# Distance metrics
# ---------------------------------------------------------------------------

def KS_mv(cdf_mat_sample, cdf_mat_pop):
    """
    Multivariate KS distance between a sample and population joint CDF.
    This is the primary objective used in the MH sampler.

    Parameters
    ----------
    cdf_mat_sample : scipy.sparse matrix
        Sample joint CDF from ``CDF_mv``.
    cdf_mat_pop : scipy.sparse matrix
        Population joint CDF from ``Pop_CDF_mv``.

    Returns
    -------
    float
        Maximum absolute difference between the two CDFs: sup|F_S - F_G|.
    """
    return abs(cdf_mat_pop - cdf_mat_sample).max()


def KS_dstat(s_metric, a_metric):
    """
    Standard two-sample KS statistic between a sample and population
    distribution.

    Parameters
    ----------
    s_metric : array-like
        Metric values for sampled nodes (e.g. degree array subset).
    a_metric : array-like
        Metric values for all population nodes.

    Returns
    -------
    float
        KS test statistic (max absolute CDF difference).
    """
    d, _ = stats.ks_2samp(s_metric, a_metric)
    return d


def KL_mv(epdf_dict_sample, epdf_dict_pop_key, epdf_dict_pop_value):
    """
    KL divergence between the sample and population joint distributions
    (multivariate). Used in the KL-distance benchmark sampler (Appendix A).

    Parameters
    ----------
    epdf_dict_sample : dict
        Sample joint EPDF from ``EPDF_mv``.
    epdf_dict_pop_key : dict_keys
        Keys of the population EPDF (computed once up front).
    epdf_dict_pop_value : np.ndarray
        Values of the population EPDF corresponding to ``epdf_dict_pop_key``.

    Returns
    -------
    float
        KL(population || sample).

    Notes
    -----
    Missing sample probabilities are smoothed to 1e-10 to avoid log(0).
    """
    sample_values = np.array(
        [epdf_dict_sample.get(key, 1e-10) for key in epdf_dict_pop_key]
    )
    return stats.entropy(epdf_dict_pop_value, sample_values)


def KL_stat(a_table, s_metric):
    """
    Univariate KL divergence using EPDF tables.


    Parameters
    ----------
    a_table : dict
        Population EPDF from ``EPDF_table(Counter(population_metric))``.
    s_metric : array-like
        Sample metric values.

    Returns
    -------
    float
        Absolute KL divergence.
    """
    s_table = EPDF_table(Counter(s_metric))
    all_vals = set(a_table.keys()).union(s_table.keys())
    score = 0.0
    for v in all_vals:
        s_i = s_table.get(v, 0)
        if s_i == 0:
            continue
        g_i = a_table.get(v, 0)
        if g_i == 0:
            continue
        score += s_i * np.log(s_i / g_i)
    return abs(score)


def AD_dstat(s_metric, a_metric):
    """
    Anderson-Darling two-sample test statistic.
    Used in the AD-distance benchmark sampler (Appendix A).

    Parameters
    ----------
    s_metric : array-like
        Sample metric values.
    a_metric : array-like
        Population metric values.

    Returns
    -------
    float
        Anderson-Darling statistic.
    """
    return stats.anderson_ksamp([s_metric, a_metric])[0]


# ---------------------------------------------------------------------------
# Node-attribute extractors  (passed as ``objective_func`` to samplers)
# ---------------------------------------------------------------------------

def degree_list(a_degree, list_sample):
    """
    Return the degree values for a list of sampled nodes.

    Parameters
    ----------
    a_degree : np.ndarray, shape (N,)
        Degree array for all population nodes.
    list_sample : list
        Indices of sampled nodes.

    Returns
    -------
    np.ndarray
        Degree values for the sampled nodes.

    Notes
    -----
    This function is passed as the ``objective_func`` argument to the MH
    samplers but is not called internally — it exists for API consistency.
    """
    return a_degree[np.array(list_sample)]


# ---------------------------------------------------------------------------
# Neighbor-list utilities
# ---------------------------------------------------------------------------

def get_hop1_nb_list(one_list, hop1_list):
    """
    Return all 1st-degree neighbors of a node set, excluding the nodes
    themselves.

    Parameters
    ----------
    one_list : list
        Source ego node indices.
    hop1_list : dict
        Maps each node index to its set of 1st-degree neighbors.

    Returns
    -------
    list
        All unique 1st-degree neighbors not in ``one_list``.
    """
    hop1_nb = set()
    for node in one_list:
        hop1_nb.update(hop1_list[node])
    hop1_nb -= set(one_list)
    return list(hop1_nb)


def get_one_hop2_list(one_node, hop1_list):
    """
    Return all 2nd-degree neighbors of a single node (i.e. friends-of-friends),
    excluding 1st-degree neighbors and the node itself.

    Parameters
    ----------
    one_node : int
        Source node index.
    hop1_list : dict
        Maps each node index to its set of 1st-degree neighbors.

    Returns
    -------
    set
        2nd-degree neighbor indices.
    """
    hop2 = set()
    for nb in hop1_list[one_node]:
        hop2.update(hop1_list[nb])
    hop2.difference_update(hop1_list[one_node])
    hop2.discard(one_node)
    return hop2


# ---------------------------------------------------------------------------
# Initialization: greedy contamination-free starting samples
# ---------------------------------------------------------------------------

def initial_ego_sample(NW, node_list, hop1_list, hop2_list, size):
    """
    Greedy initialization: sample egos that are at least 3 hops apart
    (no shared alters; controls both 1st- and 2nd-degree contamination).
    Used by older benchmark scripts.

    Parameters
    ----------
    NW : networkx.Graph
        The population network (unused internally; kept for API consistency).
    node_list : iterable
        Full set of candidate ego nodes.
    hop1_list : dict
        Node -> set of 1st-degree neighbors.
    hop2_list : dict
        Node -> set of 2nd-degree neighbors.
    size : int
        Desired number of sampled egos.

    Returns
    -------
    sample_node_list : set
        Sampled ego indices (may be smaller than ``size`` if pool exhausted).
    candidate_set : set
        Remaining valid candidates after sampling.
    """
    node_list = set(node_list)
    sample_node_list = set()
    candidate_set = node_list

    for i in range(size):
        if len(candidate_set) > 0:
            one_sample = rd.choice(tuple(candidate_set))
            sample_node_list.add(one_sample)
            candidate_set = (
                candidate_set - sample_node_list
                - hop1_list[one_sample]
                - hop2_list[one_sample]
            )
        else:
            print(f"WARNING: candidate pool exhausted; stopped at sample {i}")
            sample_node_list = dict()
            candidate_set = set()
            break

    return sample_node_list, candidate_set


def initial_ego_sample_hop1(node_list, hop1_list, size):
    """
    Greedy initialization: sample egos that are at least 2 hops apart
    (no direct connections; controls 1st-degree contamination only).
    This is the standard initialization used before running
    ``Balanced_Ego_Constraint_Sampling_hop1``.

    Parameters
    ----------
    node_list : iterable
        Full set of candidate ego nodes.
    hop1_list : dict
        Node -> set of 1st-degree neighbors.
    size : int
        Total number of egos to initialize (typically ``ngroup * sample_size``).

    Returns
    -------
    sample_node_list : set or dict
        Sampled ego indices.  Returns an empty dict if pool is exhausted.
    candidate_set : set
        Remaining valid candidates.
    """
    node_list = set(node_list)
    sample_node_list = set()
    candidate_set = node_list

    for i in range(size):
        if len(candidate_set) > 0:
            one_sample = rd.choice(tuple(candidate_set))
            sample_node_list.add(one_sample)
            candidate_set = candidate_set - sample_node_list - hop1_list[one_sample]
        else:
            print(f"WARNING: candidate pool exhausted; stopped at sample {i}")
            sample_node_list = dict()
            candidate_set = set()
            break

    return sample_node_list, candidate_set


def initial_ego_sample_hop2(node_list, hop1_list, hop2_list, size):
    """
    Greedy initialization: sample egos that are at least 3 hops apart
    (controls both 1st- and 2nd-degree contamination).
    Used before ``Balanced_Ego_Constraint_Sampling_hop2``.

    Parameters
    ----------
    node_list : iterable
        Full set of candidate ego nodes.
    hop1_list : dict
        Node -> set of 1st-degree neighbors.
    hop2_list : dict or None
        Node -> set of 2nd-degree neighbors.  If ``None``, 2nd-degree
        neighbors are computed on-the-fly via ``get_one_hop2_list``
        (slower but avoids pre-computing hop2 for very large networks).
    size : int
        Total number of egos to initialize.

    Returns
    -------
    sample_node_list : set or dict
        Sampled ego indices.
    candidate_set : set
        Remaining valid candidates.
    """
    node_list = set(node_list)
    sample_node_list = set()
    candidate_set = node_list

    for i in range(size):
        if len(candidate_set) > 0:
            one_sample = rd.choice(tuple(candidate_set))
            sample_node_list.add(one_sample)
            if hop2_list is None:
                candidate_set = (
                    candidate_set - sample_node_list
                    - hop1_list[one_sample]
                    - get_one_hop2_list(one_sample, hop1_list)
                )
            else:
                candidate_set = (
                    candidate_set - sample_node_list
                    - hop1_list[one_sample]
                    - hop2_list[one_sample]
                )
        else:
            print(f"WARNING: candidate pool exhausted; stopped at sample {i}")
            sample_node_list = dict()
            candidate_set = set()
            break

    return sample_node_list, candidate_set


def initial_ego_constraint_sample(NW, node_list, hop1_list, loss_rate, size):
    """
    Greedy initialization with a partial-overlap constraint: an ego is
    added only if the fraction of its alters that overlap with already-sampled
    egos' alters is below ``loss_rate``.

    When ``loss_rate = 1.0`` this reduces to ``initial_ego_sample_hop1``.
    Used in older benchmark comparisons.

    Parameters
    ----------
    NW : networkx.Graph
        Population network (unused; kept for API consistency).
    node_list : iterable
        Full set of candidate ego nodes.
    hop1_list : dict
        Node -> set of 1st-degree neighbors.
    loss_rate : float
        Maximum allowed fraction of a candidate's alters that may overlap
        with existing sampled alters (0 = no overlap, 1 = any overlap).
    size : int
        Desired sample size.

    Returns
    -------
    sample_node_list : set or dict
    candidate_set : set
    """
    node_list = set(node_list)
    sample_node_list = set()
    candidate_set = node_list

    while len(sample_node_list) < int(size):
        if len(candidate_set) > 0:
            one_sample = rd.choice(tuple(candidate_set))
            overlap_nb_count = 0
            for nb in hop1_list[one_sample]:
                if hop1_list[nb] & sample_node_list:
                    overlap_nb_count += 1
            if overlap_nb_count / len(hop1_list[one_sample]) < loss_rate:
                sample_node_list.add(one_sample)
                candidate_set = candidate_set - sample_node_list - hop1_list[one_sample]
        else:
            print(
                f"WARNING: candidate pool exhausted at "
                f"{len(sample_node_list)} samples"
            )
            sample_node_list = dict()
            candidate_set = set()
            break

    return sample_node_list, candidate_set


# ---------------------------------------------------------------------------
# MH Samplers
# ---------------------------------------------------------------------------

def Balanced_Ego_Constraint_Sampling_hop1(
    objective_func, distance_metric,
    covariate_list, hop1_list,
    loss_rate, ngroup, size, p,
    max_iter, initial_sample, initial_candidate_set,
    plot, tolerance
):
    """
    **Primary method.** Metropolis-Hastings sampler that simultaneously draws
    ``ngroup`` ego network samples (typically treatment and control), each of
    size ``size``, controlling 1st-degree contamination only.

    The target distribution is π(S) ∝ 1 / KS(S, G)^p, where KS(S, G) is the
    multivariate KS distance between the joint (degree, clustering coefficient)
    CDF of the sample S and the population G.  The group with the higher KS
    distance is preferentially updated at each iteration (active learning).

    Parameters
    ----------
    objective_func : callable
        Node attribute extractor, e.g. ``degree_list``.  Passed for API
        consistency; not called inside this function.
    distance_metric : callable
        Distance function; not called inside this function (KS_mv is
        hard-coded as the objective).
    covariate_list : list of two np.ndarray, each shape (N,)
        ``[a_degree, a_cluster_coef]`` for all N population nodes.
        The clustering coefficient should be pre-rounded to limit unique values
        and reduce CDF matrix size.
    hop1_list : dict
        Node -> set of 1st-degree neighbors.
    loss_rate : float
        Kept for API compatibility (set to 1 in all paper results; the
        partial-overlap constraint is disabled in this sampler).
    ngroup : int
        Number of treatment conditions (2 for treatment/control).
    size : int
        Number of ego networks per treatment condition.
    p : float
        Scaling exponent for the target distribution:
        p = 10 * (edges/nodes) * log10(nodes).
        Higher p rewards representative samples more aggressively.
    max_iter : int
        Number of MH iterations (200,000 used in the paper).
    initial_sample : set
        Output of ``initial_ego_sample_hop1`` — ``ngroup * size`` initial egos.
    initial_candidate_set : set
        Output of ``initial_ego_sample_hop1`` — remaining valid candidates.
    plot : bool
        Unused (plotting removed for clean replication code).
    tolerance : float
        Unused convergence tolerance (stopping is fixed at ``max_iter``).

    Returns
    -------
    best_sample_treat : set
        Best treatment ego set found (minimum joint KS across both groups).
    best_sample_control : set
        Best control ego set found.
    ks_treat : float
        KS distance of the best treatment sample.
    ks_control : float
        KS distance of the best control sample.

    Notes
    -----
    Algorithm (Web Appendix C):
    1. Split ``initial_sample`` randomly into ``ngroup`` groups of ``size``.
    2. For each iteration:
       a. Select the group with the higher KS distance (Bernoulli draw).
       b. Remove a random ego ``v`` from the selected group.
       c. Expand the candidate set with ``v`` and its 1-hop neighbors that are
          ≥ 2 hops from all remaining egos.
       d. Draw a new ego ``w`` from the updated candidate set.
       e. Accept with probability min(1, (KS_old / KS_new)^p).
       f. If both groups improve, update the best sample.
    """
    a_cov1_set = np.unique(covariate_list[0])
    a_cov2_set = np.unique(covariate_list[1])

    # Pre-compute population CDF once
    cdf_mat_pop = Pop_CDF_mv(
        covariate_list[0], covariate_list[1], a_cov1_set, a_cov2_set
    )
    cdf_mat_pop_nonzero = cdf_mat_pop.nonzero()
    cdf_mat_pop_shape = cdf_mat_pop.shape

    candidate_set = initial_candidate_set.copy()
    group_list = list(range(ngroup))

    # Randomly assign initial egos to groups
    initial_sample_ego = list(initial_sample)
    np.random.shuffle(initial_sample_ego)
    current_sample = [
        set(initial_sample_ego[i: i + size])
        for i in range(0, len(initial_sample_ego), size)
    ]

    # Initial KS distances
    kl_current = [
        KS_mv(
            CDF_mv(
                covariate_list[0], covariate_list[1], list(current_sample[g]),
                a_cov1_set, a_cov2_set,
                cdf_mat_pop_nonzero, cdf_mat_pop_shape
            ),
            cdf_mat_pop
        )
        for g in range(ngroup)
    ]
    print("Initial KS scores:", kl_current)

    kl_best = kl_current.copy()
    best_sample = current_sample.copy()
    ct = 0

    for i in range(max_iter):
        # Step 2a: select group proportional to KS distance (higher = more likely)
        ind_group = int(rd.choices(group_list, kl_current, k=1)[0])

        # Step 2b: remove a random ego
        new_sample = current_sample[ind_group].copy()
        v_pos = np.random.choice(tuple(new_sample))
        new_sample.remove(v_pos)

        leftover_sample = set(chain.from_iterable(current_sample))
        leftover_sample.discard(v_pos)

        # Step 2c: return v and its hop-1 neighborhood to candidate set
        #   (only nodes with no remaining hop-1 contact to leftover egos)
        v_neighborhood = set([v_pos]).union(hop1_list[v_pos])
        v_neighborhood_eligible = [
            node for node in v_neighborhood
            if len(hop1_list[node] & leftover_sample) == 0
        ]
        tmp_candidate_set = candidate_set.union(v_neighborhood_eligible)

        # Step 2d: draw a new ego
        w_pos = np.random.choice(tuple(tmp_candidate_set))
        new_sample.add(w_pos)

        # Step 2e: compute new KS and accept/reject
        kl_new = KS_mv(
            CDF_mv(
                covariate_list[0], covariate_list[1], list(new_sample),
                a_cov1_set, a_cov2_set,
                cdf_mat_pop_nonzero, cdf_mat_pop_shape
            ),
            cdf_mat_pop
        )

        alpha = rd.uniform(0, 1)
        if alpha < (kl_current[ind_group] / kl_new) ** p:
            # Accept: update candidate set and current sample
            candidate_set = tmp_candidate_set - hop1_list[w_pos]
            candidate_set.discard(w_pos)

            current_sample[ind_group] = new_sample.copy()
            kl_current[ind_group] = kl_new

            # Step 2f: update best sample if both groups improve
            if (kl_current[ind_group] < kl_best[ind_group] and
                    kl_current[1 - ind_group] < kl_best[1 - ind_group]):
                best_sample = current_sample.copy()
                kl_best = kl_current.copy()
                ct += 1

    print(f"Done. Best updated {ct} times. Final KS scores: {kl_best}")
    return best_sample[0], best_sample[1], kl_best[0], kl_best[1]


def Balanced_Ego_Constraint_Sampling_hop2(
    objective_func, distance_metric,
    covariate_list, hop1_list, hop2_list,
    loss_rate, ngroup, size, p,
    max_iter, initial_sample, initial_candidate_set,
    plot, tolerance
):
    """
    MH sampler controlling both 1st- and 2nd-degree contamination (Web Appendix F).

    Identical to ``Balanced_Ego_Constraint_Sampling_hop1`` except:
    - The distance constraint requires sampled egos to be ≥ 3 hops apart
      (no shared alters).
    - Candidate set restoration and pruning use both ``hop1_list`` and
      ``hop2_list``.

    Parameters
    ----------
    hop2_list : dict or None
        Node -> set of 2nd-degree neighbors.  If ``None``, 2nd-degree
        sets are computed on-the-fly via ``get_one_hop2_list``.
    (all other parameters identical to ``Balanced_Ego_Constraint_Sampling_hop1``)

    Returns
    -------
    best_sample_treat, best_sample_control, ks_treat, ks_control
        Same as ``Balanced_Ego_Constraint_Sampling_hop1``.
    """
    a_cov1_set = np.unique(covariate_list[0])
    a_cov2_set = np.unique(covariate_list[1])

    cdf_mat_pop = Pop_CDF_mv(
        covariate_list[0], covariate_list[1], a_cov1_set, a_cov2_set
    )
    cdf_mat_pop_nonzero = cdf_mat_pop.nonzero()
    cdf_mat_pop_shape = cdf_mat_pop.shape

    candidate_set = initial_candidate_set.copy()
    group_list = list(range(ngroup))

    initial_sample_ego = list(initial_sample)
    np.random.shuffle(initial_sample_ego)
    current_sample = [
        set(initial_sample_ego[i: i + size])
        for i in range(0, len(initial_sample_ego), size)
    ]

    kl_current = [
        KS_mv(
            CDF_mv(
                covariate_list[0], covariate_list[1], list(current_sample[g]),
                a_cov1_set, a_cov2_set,
                cdf_mat_pop_nonzero, cdf_mat_pop_shape
            ),
            cdf_mat_pop
        )
        for g in range(ngroup)
    ]
    print("Initial KS scores:", kl_current)

    kl_best = kl_current.copy()
    best_sample = current_sample.copy()
    ct = 0

    for i in range(max_iter):
        ind_group = int(rd.choices(group_list, kl_current, k=1)[0])

        new_sample = current_sample[ind_group].copy()
        v_pos = np.random.choice(tuple(new_sample))
        new_sample.remove(v_pos)

        leftover_sample = set(chain.from_iterable(current_sample))
        leftover_sample.discard(v_pos)

        # Restore v, its hop-1 and hop-2 neighborhood that are ≥ 3 hops
        # from remaining egos
        if hop2_list is None:
            v_hop2 = get_one_hop2_list(v_pos, hop1_list)
            v_neighborhood = set([v_pos]).union(hop1_list[v_pos]).union(v_hop2)
            v_neighborhood_eligible = [
                node for node in v_neighborhood
                if (node not in leftover_sample
                    and len(hop1_list[node] & leftover_sample) == 0
                    and len(get_one_hop2_list(node, hop1_list) & leftover_sample) == 0)
            ]
        else:
            v_neighborhood = (
                set([v_pos]).union(hop1_list[v_pos]).union(hop2_list[v_pos])
            )
            v_neighborhood_eligible = [
                node for node in v_neighborhood
                if (node not in leftover_sample
                    and len(hop1_list[node] & leftover_sample) == 0
                    and len(hop2_list[node] & leftover_sample) == 0)
            ]

        tmp_candidate_set = candidate_set.union(v_neighborhood_eligible)
        w_pos = np.random.choice(tuple(tmp_candidate_set))
        new_sample.add(w_pos)

        kl_new = KS_mv(
            CDF_mv(
                covariate_list[0], covariate_list[1], list(new_sample),
                a_cov1_set, a_cov2_set,
                cdf_mat_pop_nonzero, cdf_mat_pop_shape
            ),
            cdf_mat_pop
        )

        alpha = rd.uniform(0, 1)
        if alpha < (kl_current[ind_group] / kl_new) ** p:
            candidate_set = tmp_candidate_set.copy()
            if hop2_list is None:
                candidate_set = (
                    candidate_set - hop1_list[w_pos]
                    - get_one_hop2_list(w_pos, hop1_list)
                )
            else:
                candidate_set = (
                    candidate_set - hop1_list[w_pos] - hop2_list[w_pos]
                )
            candidate_set.discard(w_pos)

            current_sample[ind_group] = new_sample.copy()
            kl_current[ind_group] = kl_new

            if (kl_current[ind_group] < kl_best[ind_group] and
                    kl_current[1 - ind_group] < kl_best[1 - ind_group]):
                best_sample = current_sample.copy()
                kl_best = kl_current.copy()
                ct += 1
                print(f"  iter {i}: updated best KS = {kl_best}")

    print(f"Done. Best updated {ct} times. Final KS scores: {kl_best}")
    return best_sample[0], best_sample[1], kl_best[0], kl_best[1]


def Balanced_Ego_Constraint_Sampling_hop1_KL(
    objective_func, distance_metric, node_list, A,
    covariate_list, hop1_list,
    loss_rate, ngroup, size, p,
    max_iter, initial_sample, initial_candidate_set,
    plot, tolerance
):
    """
    MH sampler using KL divergence as the distance metric (Appendix A benchmark).

    Identical structure to ``Balanced_Ego_Constraint_Sampling_hop1`` but
    replaces the multivariate KS objective with KL divergence computed via
    ``KL_mv`` and ``EPDF_mv``.

    Parameters
    ----------
    node_list : list
        Full population node list (needed to compute population EPDF).
    A : ignored
        Kept for API consistency.
    (all other parameters same as ``Balanced_Ego_Constraint_Sampling_hop1``)
    """
    a_cov1 = covariate_list[0]
    a_cov2 = covariate_list[1]

    epdf_dict_pop = EPDF_mv(a_cov1, a_cov2, node_list)
    epdf_dict_pop_key = epdf_dict_pop.keys()
    epdf_dict_pop_value = np.array(list(epdf_dict_pop.values()))

    candidate_set = initial_candidate_set.copy()
    group_list = list(range(ngroup))

    initial_sample_ego = list(initial_sample)
    np.random.shuffle(initial_sample_ego)
    current_sample = [
        set(initial_sample_ego[i: i + size])
        for i in range(0, len(initial_sample_ego), size)
    ]

    kl_current = [
        KL_mv(
            EPDF_mv(a_cov1, a_cov2, list(current_sample[g])),
            epdf_dict_pop_key, epdf_dict_pop_value
        )
        for g in range(ngroup)
    ]

    kl_best = kl_current.copy()
    best_sample = current_sample.copy()
    ct = 0

    for i in range(max_iter):
        ind_group = int(
            rd.choices(
                group_list,
                [kl / sum(kl_current) for kl in kl_current],
                k=1
            )[0]
        )

        new_sample = current_sample[ind_group].copy()
        v_pos = rd.choice(tuple(new_sample))
        new_sample.remove(v_pos)

        leftover_sample = set().union(*current_sample)
        leftover_sample.discard(v_pos)

        v_neighborhood = set([v_pos]).union(hop1_list[v_pos])
        v_neighborhood_eligible = [
            node for node in v_neighborhood
            if len(hop1_list[node] & leftover_sample) == 0
        ]
        tmp_candidate_set = candidate_set.union(v_neighborhood_eligible)

        w_pos = rd.choice(tuple(tmp_candidate_set))
        new_sample.add(w_pos)

        kl_new = KL_mv(
            EPDF_mv(a_cov1, a_cov2, list(new_sample)),
            epdf_dict_pop_key, epdf_dict_pop_value
        )

        alpha = rd.uniform(0, 1)
        if alpha < (kl_current[ind_group] / kl_new) ** p:
            candidate_set = tmp_candidate_set - hop1_list[w_pos]
            candidate_set.discard(w_pos)

            current_sample[ind_group] = new_sample.copy()
            kl_current[ind_group] = kl_new

            if (kl_current[ind_group] < kl_best[ind_group] and
                    kl_current[1 - ind_group] < kl_best[1 - ind_group]):
                best_sample = current_sample.copy()
                kl_best = kl_current.copy()
                ct += 1

    print(f"Done. Best updated {ct} times. Final KL scores: {kl_best}")
    return best_sample[0], best_sample[1], kl_best[0], kl_best[1]


def Balanced_Ego_Constraint_Sampling_hop1_KS_linear(
    objective_func, distance_metric, node_list, A,
    covariate_list, hop1_list,
    loss_rate, ngroup, size, p,
    max_iter, initial_sample, initial_candidate_set,
    plot, tolerance
):
    """
    MH sampler using a linear (univariate) KS distance on degree only
    (Appendix A benchmark).

    Uses a precomputed ECDF for speed instead of the full joint CDF.

    Parameters
    ----------
    node_list : list
        Full population node list (needed to build population ECDF).
    A : ignored
        Kept for API consistency.
    (all other parameters same as ``Balanced_Ego_Constraint_Sampling_hop1``)
    """
    a_degree = covariate_list[0]

    a_degree_key = np.sort(np.unique(a_degree))

    def _ecdf(node_ids):
        counter = Counter(a_degree[node_ids])
        counts = [counter.get(k, 0) for k in a_degree_key]
        return np.cumsum(counts) / len(node_ids)

    pop_ecdf = _ecdf(node_list)

    candidate_set = initial_candidate_set.copy()
    group_list = list(range(ngroup))

    initial_sample_ego = list(initial_sample)
    np.random.shuffle(initial_sample_ego)
    current_sample = [
        set(initial_sample_ego[i: i + size])
        for i in range(0, len(initial_sample_ego), size)
    ]

    kl_current = [
        np.abs(pop_ecdf - _ecdf(list(current_sample[g]))).max()
        for g in range(ngroup)
    ]

    kl_best = kl_current.copy()
    best_sample = current_sample.copy()
    ct = 0

    for i in range(max_iter):
        ind_group = int(
            rd.choices(
                group_list,
                [kl / sum(kl_current) for kl in kl_current],
                k=1
            )[0]
        )

        new_sample = current_sample[ind_group].copy()
        v_pos = rd.choice(tuple(new_sample))
        new_sample.remove(v_pos)

        leftover_sample = set().union(*current_sample)
        leftover_sample.discard(v_pos)

        v_neighborhood = set([v_pos]).union(hop1_list[v_pos])
        v_neighborhood_eligible = [
            node for node in v_neighborhood
            if len(hop1_list[node] & leftover_sample) == 0
        ]
        tmp_candidate_set = candidate_set.union(v_neighborhood_eligible)

        w_pos = rd.choice(tuple(tmp_candidate_set))
        new_sample.add(w_pos)

        kl_new = np.abs(pop_ecdf - _ecdf(list(new_sample))).max()

        alpha = rd.uniform(0, 1)
        if alpha < (kl_current[ind_group] / kl_new) ** p:
            candidate_set = tmp_candidate_set - hop1_list[w_pos]
            candidate_set.discard(w_pos)

            current_sample[ind_group] = new_sample.copy()
            kl_current[ind_group] = kl_new

            if (kl_current[ind_group] < kl_best[ind_group] and
                    kl_current[1 - ind_group] < kl_best[1 - ind_group]):
                best_sample = current_sample.copy()
                kl_best = kl_current.copy()
                ct += 1

    print(f"Done. Best updated {ct} times. Final KS-linear scores: {kl_best}")
    return best_sample[0], best_sample[1], kl_best[0], kl_best[1]


# ---------------------------------------------------------------------------
# Alter-level samplers (used in older exploratory scripts)
# ---------------------------------------------------------------------------

def Balanced_Alter_MV_Sampling(
    degree_list, KS_mv, node_list, A,
    covariate_list, hop1_list, rp_ego,
    percent, p, max_iter, tolerance
):
    """
    MH sampler for alter-level sampling within a fixed ego set.

    Draws a representative subset of alters for one treatment condition,
    subject to a per-ego minimum: each ego retains at least
    ``max(1, floor(degree * percent))`` alters.

    Parameters
    ----------
    degree_list, KS_mv : callable
        Passed for API consistency (not called inside this function).
    node_list : list
        Full population node list (for population CDF reference).
    A : ignored
    covariate_list : list of two np.ndarray
        [a_degree, a_cluster_coef].
    hop1_list : dict
    rp_ego : list
        Ego indices whose alters form the sampling universe.
    percent : float
        Minimum fraction of alters to retain per ego.
    p, max_iter, tolerance : float, int, float
        MH hyperparameters.

    Returns
    -------
    best_sample : set
        Best alter subset found.
    ks_best : float
        KS distance of the best sample.
    """
    all_sample = list(node_list)
    a_degree = covariate_list[0]
    a_cov1_set = np.unique(covariate_list[0])
    a_cov2_set = np.unique(covariate_list[1])

    cdf_mat_pop = Pop_CDF_mv(
        covariate_list[0], covariate_list[1], a_cov1_set, a_cov2_set
    )
    cdf_mat_pop_nonzero = cdf_mat_pop.nonzero()
    cdf_mat_pop_shape = cdf_mat_pop.shape

    alter_set = set()
    alter_ego_map = {}
    initial_sample = set()

    for ego in rp_ego:
        alters = hop1_list[ego]
        alter_set.update(alters)
        initial_sample.update(
            set(rd.sample(alters, max(1, int(len(alters) * percent))))
        )
        for nb in alters:
            alter_ego_map[nb] = ego

    alter_set -= set(rp_ego)
    extra = int(len(alter_set) * percent) - len(initial_sample)
    if extra > 0:
        initial_sample.update(
            set(rd.sample(list(alter_set - initial_sample), extra))
        )

    candidate_set = alter_set - initial_sample
    current_ego_degree = Counter(
        [alter_ego_map[a] for a in initial_sample]
    )

    kl_current = KS_mv(
        CDF_mv(
            covariate_list[0], covariate_list[1], list(initial_sample),
            a_cov1_set, a_cov2_set,
            cdf_mat_pop_nonzero, cdf_mat_pop_shape
        ),
        cdf_mat_pop
    )
    current_sample = initial_sample.copy()
    kl_best = kl_current
    best_sample = current_sample.copy()
    ct = 0

    for _ in range(int(max_iter)):
        new_sample = current_sample.copy()
        v_pos = rd.choice(tuple(new_sample))
        w_pos = rd.choice(tuple(candidate_set))

        v_ego = alter_ego_map[v_pos]
        w_ego = alter_ego_map[w_pos]

        tmp_degree = current_ego_degree.copy()
        tmp_degree[v_ego] -= 1
        tmp_degree[w_ego] += 1

        if tmp_degree[v_ego] >= max(1, a_degree[v_ego] * 0.5):
            new_sample.discard(v_pos)
            new_sample.add(w_pos)

            kl_new = KS_mv(
                CDF_mv(
                    covariate_list[0], covariate_list[1], list(new_sample),
                    a_cov1_set, a_cov2_set,
                    cdf_mat_pop_nonzero, cdf_mat_pop_shape
                ),
                cdf_mat_pop
            )

            alpha = rd.uniform(0, 1)
            if alpha < (kl_current / kl_new) ** p:
                candidate_set.add(v_pos)
                candidate_set.discard(w_pos)
                current_sample = new_sample.copy()
                current_ego_degree = tmp_degree.copy()
                kl_current = kl_new
                if kl_current < kl_best:
                    best_sample = current_sample.copy()
                    kl_best = kl_current
                    ct += 1

    return best_sample, kl_best


def Balanced_All_Alter_MV_Sampling(
    degree_list, KS_mv, node_list, A,
    covariate_list, hop1_list, alter_set,
    p, max_iter, tolerance
):
    """
    MH sampler for alter-level sampling treating all alters as a single pool
    (no per-ego minimum constraint).

    Degree-2 alters are always retained; the remaining alters are sampled
    proportionally to match the population degree-2 fraction.

    Parameters
    ----------
    alter_set : set
        Pre-cleaned alter indices (common alters already removed).
    (all other parameters same as ``Balanced_Alter_MV_Sampling``)

    Returns
    -------
    best_sample : set
        Best alter subset (including mandatory degree-2 alters).
    ks_best : float
    """
    all_sample = list(node_list)
    a_cov1_set = np.unique(covariate_list[0])
    a_cov2_set = np.unique(covariate_list[1])

    cdf_mat_pop = Pop_CDF_mv(
        covariate_list[0], covariate_list[1], a_cov1_set, a_cov2_set
    )
    cdf_mat_pop_nonzero = cdf_mat_pop.nonzero()
    cdf_mat_pop_shape = cdf_mat_pop.shape

    # Always keep degree-2 alters
    alter_keep = set(
        a for a in alter_set if covariate_list[0][a] == 2
    )
    n_deg2_pop = int((covariate_list[0] == 2).sum())
    num_to_sample = int(
        len(alter_keep) / n_deg2_pop * len(node_list) - len(alter_keep)
    )
    initial_sample = set(
        rd.sample(list(alter_set - alter_keep), num_to_sample)
    )

    candidate_set = alter_set - initial_sample - alter_keep

    kl_current = KS_mv(
        CDF_mv(
            covariate_list[0], covariate_list[1],
            list(initial_sample | alter_keep),
            a_cov1_set, a_cov2_set,
            cdf_mat_pop_nonzero, cdf_mat_pop_shape
        ),
        cdf_mat_pop
    )
    current_sample = initial_sample.copy()
    kl_best = kl_current
    best_sample = current_sample.copy()
    ct = 0

    for _ in range(int(max_iter)):
        new_sample = current_sample.copy()
        v_pos = rd.choice(tuple(new_sample))
        w_pos = rd.choice(tuple(candidate_set))

        new_sample.discard(v_pos)
        new_sample.add(w_pos)

        kl_new = KS_mv(
            CDF_mv(
                covariate_list[0], covariate_list[1],
                list(new_sample | alter_keep),
                a_cov1_set, a_cov2_set,
                cdf_mat_pop_nonzero, cdf_mat_pop_shape
            ),
            cdf_mat_pop
        )

        alpha = rd.uniform(0, 1)
        if alpha < (kl_current / kl_new) ** p:
            candidate_set.add(v_pos)
            candidate_set.discard(w_pos)
            current_sample = new_sample.copy()
            kl_current = kl_new
            if kl_current < kl_best:
                best_sample = current_sample.copy()
                kl_best = kl_current
                ct += 1

    print(f"Best KS: {kl_best}, updates: {ct}")
    return best_sample | alter_keep, kl_best
