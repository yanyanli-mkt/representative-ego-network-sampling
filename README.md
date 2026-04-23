# Replication Package
## "A Representative Sampling Method for Peer Encouragement Designs in Network Experiments"
### Marketing Science

---

## Overview

This package replicates all tables, figures, and numerical results in the paper.
It is organized into numbered pipeline scripts that run in order, plus source
libraries in `src/` and validation tests in `tests/`.

**Language requirements:** Python 3.10+ and R 4.2+

---

## Directory Structure

```
.
├── src/
│   ├── sampler.py          Core MH sampler and KS/CDF utilities
│   └── utils.py            Network loading, attribute computation, t-test helpers
│
├── scripts/
│   ├── 01_prepare_networks.py      Load networks, compute attributes (Table 1)
│   ├── 02_run_sampling.py          Exclusion baseline + MH sampler — saves node files
│   ├── 02b_summary_tables.py       Post-sampling summaries (Tables 2, Figs 3-5 data)
│   ├── 03_figures_34.R            Figures 3 and 4
│   ├── 04_boundary_conditions.py   Max feasible sample sizes (Table 3)
│   ├── 05_causal_inference.py      ATE/CATE estimation (Section 5, Figures 6 & 7)
│   ├── 06_figures.R                Figures 5, 6, and 7
│   ├── 06_run_sampling_hop2.py     2nd-degree contamination (Web Appendix F)
│   └── 07_figures_cate.R           Supplementary CATE figures
│
├── data/
│   └── README.md           Instructions for downloading network data
│
├── tests/
│   ├── test_pipeline.py    Unit and smoke tests (21 tests)
│   └── validate_numerical.py  Numerical validation vs. reference outputs
│
├── requirements.txt        Python dependencies
└── README.md               This file
```

---

## Setup

### Python

```bash
pip install -r requirements.txt
```

### R

```r
install.packages(c("tidyverse", "patchwork", "cowplot", "ggbreak"))
```

---

## Data

All network data must be downloaded separately and placed under `data/`.
See `data/README.md` for download links and folder structure.

| Paper name | Folder |
|-----------|--------|
| LastFM    | `data/lastfm/` |
| Twitch    | `data/twitch/` |
| Pokec     | `data/pokec/` |

Simulated networks (Small, Base, Sparse) are generated automatically by `01_prepare_networks.py`.

---

## Running the Pipeline

Run all scripts from the repository root in the order below.

```bash
# Step 0: Download real-world data (see data/README.md for links and filenames)
#   Place files in data/lastfm/, data/twitch/, data/pokec/

# Step 1: Prepare networks → Table 1
#   Generates attributes and hop lists for all 6 networks.
#   Use --skip_hop2 for large networks (pokec, simulate_base, simulate_sparse)
#   to avoid memory issues — hop-2 is computed on-the-fly during sampling.
python scripts/01_prepare_networks.py
# or: python scripts/01_prepare_networks.py --skip_hop2

# Step 2: Exclusion baseline + MH sampler → saves node index files
python scripts/02_run_sampling.py

# Step 2b: Post-sampling summaries → Table 2, Figure 5 input, Figures 3 & 4 data
python scripts/02b_summary_tables.py

# Step 3: Boundary conditions → Table 3
python scripts/04_boundary_conditions.py

# Step 4: Causal inference → Section 5, Figures 6 and 7
#   Use --skip_expo_mapping to reuse a previously computed exposure mapping
#   (recommended after the first run — each network takes ~10 min to compute)
python scripts/05_causal_inference.py
# or: python scripts/05_causal_inference.py --skip_expo_mapping

# Step 5: 2nd-degree contamination → Web Appendix F (optional, run after step 4)
python scripts/06_run_sampling_hop2.py

# Figures 3 and 4 (R)
Rscript scripts/03_figures_34.R

# Figures 5, 6, and 7 (R)
Rscript scripts/06_figures.R
```

All figures are saved as PDF and PNG to `output/figures/`.

---

## Output Files

| File | Used for |
|------|----------|
| `output/tables/table1_network_summary.csv` | Table 1 |
| `output/tables/table2_population_vs_excluded.csv` | Table 2 |
| `output/tables/<network>/summary_contamination_hop1_pcent.csv` | Figures 3A, 4A |
| `output/tables/<network>/summary_contamination_hop2_pcent.csv` | Figures 3B, 4B |
| `output/tables/<network>/summary_network_property_matchRP_combine_sample.csv` | Figure 5 |
| `output/tables/<network>/ate_sample500_tau1_0_0.1_tau2_0_sim100.csv` | Figure 6 |

---

## Validation

```bash
# Unit tests (~30 sec)
python tests/test_pipeline.py -v

# Numerical validation against reference CSVs (requires LastFM data)
python tests/validate_numerical.py --network lastfm --data_dir ./data --quick
```

---

## Estimator Notes (Section 5)

Four estimators are implemented following Aronow & Samii (2017):

- **Diff-in-Means (DIM):** Standard difference in group means.
- **Horvitz-Thompson (HT):** IPW estimator per A&S Eq. 1 & 3. Egos with no
  clean alters contribute 0 to the ITE sum (not excluded), consistent with
  the A&S formula where non-exposed units contribute zero to the population sum.
- **Hájek:** Normalized HT ratio estimator per A&S Section 4.
- **Representative:** DIM applied to the MH representative sample.

---

## Network Labels

| Folder name | Paper label |
|-------------|-------------|
| `lastfm` | LastFM |
| `twitch` | Twitch |
| `pokec` | Pokec |
| `simulate_small` | Small |
| `simulate_base` | Base |
| `simulate_sparse` | Sparse |

---

## Runtime Estimates

| Script | Approx. time |
|--------|-------------|
| 01_prepare_networks.py | 5–15 min |
| 02_run_sampling.py | 2–6 hours |
| 02b_summary_tables.py | 1–4 hours |
| 05_causal_inference.py | 4–12 hours |
| R figure scripts | < 5 min each |
