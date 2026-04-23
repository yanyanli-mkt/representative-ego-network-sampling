#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tests/test_pipeline.py
----------------------
Test suite for the representative ego network sampling replication package.

Three test levels, run in order:
  - TestImports       : all modules and functions are importable (seconds)
  - TestSampler       : core MH algorithm correctness on a tiny network (seconds)
  - TestPipelineSmoke : full pipeline end-to-end on synthetic data (minutes)

Usage
-----
    # Run all tests
    python tests/test_pipeline.py

    # Run only fast tests
    python tests/test_pipeline.py TestImports TestSampler

    # Verbose output
    python tests/test_pipeline.py -v
"""

import sys
import os
import unittest
import numpy as np
import networkx as nx

# Allow running from repo root or tests/ directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import sampler
import utils


# ---------------------------------------------------------------------------
# Shared fixture: tiny synthetic network (built once)
# ---------------------------------------------------------------------------
def _make_tiny_network(n=300, seed=42):
    """Build a small power-law cluster graph for testing."""
    NW = nx.powerlaw_cluster_graph(n, 3, 0.3, seed=seed)
    mapping = {node: i for i, node in enumerate(sorted(NW.nodes()))}
    NW = nx.relabel_nodes(NW, mapping)
    attrs = utils.compute_node_attributes(NW)
    hop1_list = utils.build_hop1_list(NW, attrs["node_list"])
    return NW, attrs, hop1_list


# ---------------------------------------------------------------------------
# Level 1: import checks
# ---------------------------------------------------------------------------
class TestImports(unittest.TestCase):
    """All expected functions are importable and callable."""

    SAMPLER_FUNCTIONS = [
        "Balanced_Ego_Constraint_Sampling_hop1",
        "Balanced_Ego_Constraint_Sampling_hop2",
        "Balanced_Ego_Constraint_Sampling_hop1_KL",
        "Balanced_Ego_Constraint_Sampling_hop1_KS_linear",
        "Balanced_Alter_MV_Sampling",
        "Balanced_All_Alter_MV_Sampling",
        "initial_ego_sample_hop1",
        "initial_ego_sample_hop2",
        "initial_ego_sample",
        "initial_ego_constraint_sample",
        "KS_mv", "KS_dstat", "KL_mv", "KL_stat", "AD_dstat",
        "Pop_CDF_mv", "CDF_mv",
        "EPDF_table", "EPDF_mv",
        "degree_list",
        "get_hop1_nb_list", "get_one_hop2_list",
    ]

    UTILS_FUNCTIONS = [
        "load_real_network", "generate_simulated_network",
        "compute_node_attributes", "build_hop1_list", "build_hop2_list",
        "save_network_data", "load_network_data",
        "generate_covariates", "compute_p",
        "my_t_test", "my_t_test_correction",
        "NETWORK_CONFIG", "SIMULATED_NETWORK_CONFIG", "NETWORK_FEATURES",
    ]

    def test_sampler_exports(self):
        for fn in self.SAMPLER_FUNCTIONS:
            with self.subTest(function=fn):
                self.assertTrue(
                    hasattr(sampler, fn),
                    f"sampler.{fn} not found"
                )

    def test_utils_exports(self):
        for fn in self.UTILS_FUNCTIONS:
            with self.subTest(function=fn):
                self.assertTrue(
                    hasattr(utils, fn),
                    f"utils.{fn} not found"
                )

    def test_network_configs_not_empty(self):
        self.assertGreater(len(utils.NETWORK_CONFIG), 0)
        self.assertGreater(len(utils.SIMULATED_NETWORK_CONFIG), 0)

    def test_simulated_network_params(self):
        for nv, cfg in utils.SIMULATED_NETWORK_CONFIG.items():
            with self.subTest(network=nv):
                self.assertIn("N", cfg)
                self.assertIn("m", cfg)
                self.assertIn("p", cfg)
                self.assertIn("seed", cfg)


# ---------------------------------------------------------------------------
# Level 2: sampler correctness
# ---------------------------------------------------------------------------
class TestSampler(unittest.TestCase):
    """Core MH algorithm produces valid, contamination-free samples."""

    @classmethod
    def setUpClass(cls):
        np.random.seed(42)
        cls.NW, cls.attrs, cls.hop1_list = _make_tiny_network(n=300)
        cls.a_degree = cls.attrs["a_degree"]
        cls.a_cluster_coef = cls.attrs["a_cluster_coef"]
        cls.node_list = list(cls.attrs["node_list"])
        cls.total_node = len(cls.node_list)
        cls.p = utils.compute_p(cls.NW)
        cls.nsample = 8
        cls.ngroup = 2

    def _no_first_degree_contamination(self, ego_list):
        """Return number of directly-connected ego pairs."""
        count = 0
        for i, ego in enumerate(ego_list):
            for other in ego_list[i + 1:]:
                if other in self.hop1_list[ego]:
                    count += 1
        return count

    def test_initial_sample_size(self):
        initial, candidate = sampler.initial_ego_sample_hop1(
            self.node_list, self.hop1_list,
            self.nsample * self.ngroup
        )
        self.assertEqual(len(initial), self.nsample * self.ngroup)

    def test_initial_sample_no_contamination(self):
        initial, _ = sampler.initial_ego_sample_hop1(
            self.node_list, self.hop1_list,
            self.nsample * self.ngroup
        )
        ct = self._no_first_degree_contamination(list(initial))
        self.assertEqual(ct, 0, "Initial sample has 1st-degree contamination")

    def test_mh_output_size(self):
        initial, candidate = sampler.initial_ego_sample_hop1(
            self.node_list, self.hop1_list,
            self.nsample * self.ngroup
        )
        rp1, rp2, ks1, ks2 = sampler.Balanced_Ego_Constraint_Sampling_hop1(
            sampler.degree_list, sampler.KS_mv,
            [self.a_degree, np.round(self.a_cluster_coef, 5)],
            self.hop1_list,
            1, self.ngroup, self.nsample, self.p,
            200, list(initial), candidate,
            False, 1e-6
        )
        self.assertEqual(len(rp1), self.nsample)
        self.assertEqual(len(rp2), self.nsample)

    def test_mh_output_no_contamination(self):
        initial, candidate = sampler.initial_ego_sample_hop1(
            self.node_list, self.hop1_list,
            self.nsample * self.ngroup
        )
        rp1, rp2, _, _ = sampler.Balanced_Ego_Constraint_Sampling_hop1(
            sampler.degree_list, sampler.KS_mv,
            [self.a_degree, np.round(self.a_cluster_coef, 5)],
            self.hop1_list,
            1, self.ngroup, self.nsample, self.p,
            200, list(initial), candidate,
            False, 1e-6
        )
        all_egos = list(rp1) + list(rp2)
        ct = self._no_first_degree_contamination(all_egos)
        self.assertEqual(ct, 0, "MH output has 1st-degree contamination")

    def test_mh_ks_nonnegative(self):
        initial, candidate = sampler.initial_ego_sample_hop1(
            self.node_list, self.hop1_list,
            self.nsample * self.ngroup
        )
        _, _, ks1, ks2 = sampler.Balanced_Ego_Constraint_Sampling_hop1(
            sampler.degree_list, sampler.KS_mv,
            [self.a_degree, np.round(self.a_cluster_coef, 5)],
            self.hop1_list,
            1, self.ngroup, self.nsample, self.p,
            200, list(initial), candidate,
            False, 1e-6
        )
        self.assertGreaterEqual(ks1, 0)
        self.assertGreaterEqual(ks2, 0)
        self.assertLessEqual(ks1, 1)
        self.assertLessEqual(ks2, 1)

    def test_mh_improves_ks(self):
        """MH should reduce KS from the initial random sample."""
        initial, candidate = sampler.initial_ego_sample_hop1(
            self.node_list, self.hop1_list,
            self.nsample * self.ngroup
        )
        initial_list = list(initial)
        np.random.shuffle(initial_list)

        covariate_list = [self.a_degree, np.round(self.a_cluster_coef, 5)]
        a_cov1 = np.unique(covariate_list[0])
        a_cov2 = np.unique(covariate_list[1])
        cdf_pop = sampler.Pop_CDF_mv(covariate_list[0], covariate_list[1], a_cov1, a_cov2)

        ks_initial = sampler.KS_mv(
            sampler.CDF_mv(
                covariate_list[0], covariate_list[1],
                initial_list[:self.nsample],
                a_cov1, a_cov2,
                cdf_pop.nonzero(), cdf_pop.shape
            ),
            cdf_pop
        )

        _, _, ks1, _ = sampler.Balanced_Ego_Constraint_Sampling_hop1(
            sampler.degree_list, sampler.KS_mv,
            covariate_list, self.hop1_list,
            1, self.ngroup, self.nsample, self.p,
            500, initial_list, candidate,
            False, 1e-6
        )
        self.assertLessEqual(
            ks1, ks_initial,
            f"MH did not improve KS: initial={ks_initial:.4f}, final={ks1:.4f}"
        )

    def test_ks_dstat_range(self):
        ks = sampler.KS_dstat(self.a_degree, self.a_degree[:50])
        self.assertGreaterEqual(ks, 0)
        self.assertLessEqual(ks, 1)

    def test_get_hop1_nb_list(self):
        ego_list = self.node_list[:3]
        nbs = sampler.get_hop1_nb_list(ego_list, self.hop1_list)
        # All neighbors should not be in ego_list
        self.assertTrue(all(nb not in ego_list for nb in nbs))

    def test_get_one_hop2_list(self):
        hop2 = sampler.get_one_hop2_list(0, self.hop1_list)
        # Should not contain the node itself
        self.assertNotIn(0, hop2)
        # Should not contain direct neighbors
        for nb in self.hop1_list[0]:
            self.assertNotIn(nb, hop2)


# ---------------------------------------------------------------------------
# Level 3: full pipeline smoke test
# ---------------------------------------------------------------------------
class TestPipelineSmoke(unittest.TestCase):
    """
    Full pipeline: generate network → sample → estimate.
    Uses nsim=2 and max_iter=300 for speed.
    Confirms outputs are the right shape and values are in valid ranges.
    """

    @classmethod
    def setUpClass(cls):
        np.random.seed(0)
        cls.NW, cls.attrs, cls.hop1_list = _make_tiny_network(n=400)
        cls.a_degree = cls.attrs["a_degree"]
        cls.a_cluster_coef = cls.attrs["a_cluster_coef"]
        cls.node_list = list(cls.attrs["node_list"])
        cls.total_node = len(cls.node_list)
        cls.p = utils.compute_p(cls.NW)
        cls.nsample = 10
        cls.ngroup = 2
        cls.nsim = 2
        cls.covariate_list = [
            cls.a_degree,
            np.round(cls.a_cluster_coef, 5)
        ]

    def test_compute_p(self):
        p = utils.compute_p(self.NW)
        self.assertGreater(p, 0)

    def test_generate_covariates(self):
        covs = utils.generate_covariates(self.a_degree, seed=42)
        self.assertEqual(set(covs.keys()), {"Z1", "Z2", "Z3"})
        for name, val in covs.items():
            with self.subTest(covariate=name):
                self.assertEqual(val.shape, (self.total_node,))
                self.assertFalse(np.any(np.isnan(val)))
        # Z1 should be negatively correlated with degree
        self.assertLess(np.corrcoef(self.a_degree, covs["Z1"])[0, 1], 0)
        # Z2 should be positively correlated with degree
        self.assertGreater(np.corrcoef(self.a_degree, covs["Z2"])[0, 1], 0)

    def test_t_test_helpers(self):
        rng = np.random.default_rng(42)
        y1 = rng.normal(1.5, 1, 100)
        y0 = rng.normal(1.0, 1, 100)
        true_effect = 0.5
        beta_hat, se, tstat, conf, power, coverage = utils.my_t_test(y1, y0, true_effect)
        self.assertGreater(se, 0)
        self.assertEqual(len(conf), 2)
        self.assertLess(conf[0], conf[1])
        self.assertIsInstance(power, (bool, np.bool_))
        self.assertIsInstance(coverage, (bool, np.bool_))

    def test_t_test_with_nans(self):
        y1 = np.array([1.0, np.nan, 2.0, 1.5, np.nan])
        y0 = np.array([0.5, 1.0, np.nan, 0.8, 0.9])
        beta_hat, se, _, conf, power, coverage = utils.my_t_test(y1, y0, 0.5)
        self.assertFalse(np.isnan(beta_hat))
        self.assertFalse(np.isnan(se))

    def test_full_sampling_loop(self):
        """Run nsim=2 complete sampling iterations and verify outputs."""
        results = []
        for sim in range(self.nsim):
            initial, candidate = sampler.initial_ego_sample_hop1(
                self.node_list, self.hop1_list,
                self.nsample * self.ngroup
            )
            rp1, rp2, ks1, ks2 = sampler.Balanced_Ego_Constraint_Sampling_hop1(
                sampler.degree_list, sampler.KS_mv,
                self.covariate_list, self.hop1_list,
                1, self.ngroup, self.nsample, self.p,
                300, list(initial), candidate,
                False, 1e-6
            )
            results.append((rp1, rp2, ks1, ks2))

        self.assertEqual(len(results), self.nsim)
        for rp1, rp2, ks1, ks2 in results:
            self.assertEqual(len(rp1), self.nsample)
            self.assertEqual(len(rp2), self.nsample)
            self.assertGreaterEqual(ks1, 0)
            self.assertGreaterEqual(ks2, 0)

    def test_pop_cdf_mv_shape(self):
        a_cov1 = np.unique(self.a_degree)
        a_cov2 = np.unique(self.a_cluster_coef)
        cdf_pop = sampler.Pop_CDF_mv(
            self.a_degree, self.a_cluster_coef, a_cov1, a_cov2
        )
        self.assertEqual(cdf_pop.shape, (len(a_cov1), len(a_cov2)))

    def test_cdf_mv_consistency(self):
        """CDF of the full population should be close to the population CDF."""
        a_cov1 = np.unique(self.a_degree)
        a_cov2 = np.unique(self.a_cluster_coef)
        cdf_pop = sampler.Pop_CDF_mv(
            self.a_degree, self.a_cluster_coef, a_cov1, a_cov2
        )
        cdf_sample = sampler.CDF_mv(
            self.a_degree, self.a_cluster_coef, self.node_list,
            a_cov1, a_cov2,
            cdf_pop.nonzero(), cdf_pop.shape
        )
        ks = sampler.KS_mv(cdf_sample, cdf_pop)
        # CDF of full population vs itself should be ~0
        self.assertAlmostEqual(ks, 0.0, places=5)

    def test_hop2_sampler(self):
        """hop2 sampler should produce contamination-free samples."""
        hop2_list = utils.build_hop2_list(self.hop1_list)
        initial, candidate = sampler.initial_ego_sample_hop2(
            self.node_list, self.hop1_list, hop2_list, self.nsample * self.ngroup
        )
        if not initial:
            self.skipTest("Pool exhausted for hop2 on this network")

        rp1, rp2, ks1, ks2 = sampler.Balanced_Ego_Constraint_Sampling_hop2(
            sampler.degree_list, sampler.KS_mv,
            self.covariate_list, self.hop1_list, hop2_list,
            1, self.ngroup, self.nsample, self.p,
            200, list(initial), candidate,
            False, 1e-6
        )
        self.assertEqual(len(rp1), self.nsample)
        self.assertEqual(len(rp2), self.nsample)

        # No ego should share an alter with another ego
        all_egos = list(rp1) + list(rp2)
        for ego in all_egos:
            others = set(all_egos) - {ego}
            shared = hop2_list[ego] & others
            self.assertEqual(
                len(shared), 0,
                f"Ego {ego} shares an alter with another ego"
            )


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # If specific test classes are named as args, run only those
    if len(sys.argv) > 1 and not sys.argv[1].startswith("-"):
        suite = unittest.TestSuite()
        loader = unittest.TestLoader()
        for cls_name in sys.argv[1:]:
            cls = globals()[cls_name]
            suite.addTests(loader.loadTestsFromTestCase(cls))
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
    else:
        unittest.main(verbosity=2)
