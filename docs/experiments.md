# Module 10: Experimental Harness and CLI Guide

## Overview
Module 10 provides the complete evaluation infrastructure for executing benchmarks, baseline comparisons, component ablations, sensitivity analyses, and publication figure generation.

---

## Command-Line Scripts

### 1. Running an Experiment
Execute a full matrix from a JSON configuration or CLI arguments:
```bash
# Run canonical 5-baseline evaluation on combined degradation scenario
python scripts/run_experiment.py --scenario combined_degradation --seed 42 --tokens 32

# Run using a full ExperimentConfig JSON
python scripts/run_experiment.py --config configs/canonical_experiment.json
```

### 2. Five-Baseline Comparison
Compare policies B1 (`static`), B2 (`network_reactive`), B3 (`memory_reactive`), B4 (`joint_reactive`), and B5 (`predictive`) under identical environmental traces:
```bash
python scripts/run_comparison.py --scenario combined_degradation --seed 42
```

### 3. Component Ablation Suite
Evaluate the isolated contribution of each component (A1–A4 mandatory, A5–A8 optional):
```bash
# Run mandatory ablations (A1-A4)
python scripts/run_ablation.py --scenario combined_degradation --seed 42

# Run all ablations including stability sweeps (A1-A8)
python scripts/run_ablation.py --all-ablations --seed 42
```

### 4. Analyzing Results
Validate experimental invariants and compute statistical aggregations:
```bash
python scripts/analyze_results.py --raw-dir results/raw --output-dir results/aggregated
```

### 5. Building the Research Dataset
Export raw trials to CSV and JSON formats:
```bash
python scripts/build_research_dataset.py --raw-dir results/raw --output-dir results/export
```

### 6. Generating Figures and Tables
Generate scientific figures and formatted tables:
```bash
python scripts/generate_figures.py --raw-dir results/raw --output-dir results/figures
python scripts/generate_tables.py --raw-dir results/raw --output-dir results/tables
```

---

## Result Directory Layout
```
results/
├── raw/                  # Immutable trial JSON results (<trial_id>.json)
├── traces/               # Cached shared environment traces (<scenario>_<seed>.json)
├── aggregated/           # Aggregated statistics and paired comparison JSON/CSV
├── export/               # Clean tabular research datasets (trials.csv, trials.json)
├── figures/              # High-resolution scientific figures (PNG)
├── tables/               # Formatted LaTeX (.tex) and Markdown (.md) tables
└── manifest.json         # Execution tracking and environment provenance
```
