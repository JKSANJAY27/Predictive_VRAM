# Module 12: Reproducible Research, Statistical Validation & Publication-Ready Evaluation

## Overview

Module 12 establishes a research-grade reproducibility, statistical-validation, and publication-ready reporting layer for the **Predictive VRAM & Network-Aware Dynamic Split Inference for Edge LLMs** project.

It provides rigorous, auditable guarantees turning raw experimental execution data into peer-review-quality scientific evidence without introducing any new partition-selection algorithms or compromising runtime invariants.

---

## 1. Architecture & Component Structure

The Module 12 layer resides in [`src/research/`](file:///c:/Users/reape/OneDrive/Desktop/Predictive_VRAM/src/research/):

```text
src/research/
├── __init__.py           # Public exports
├── reproducibility.py   # SeedManager, EnvironmentFingerprint, ResearchConfigHasher
├── manifest.py          # ResearchManifest, ManifestTrialRecord, CANONICAL_METRIC_DEFINITIONS
├── statistics.py        # StatisticalAnalyzer, Student-t & Bootstrap CIs, paired tests, Holm/FDR
├── comparisons.py       # PolicyComparisonEngine (paired baseline tests vs reference)
├── ablations.py         # ResearchAblationRunner (pre-stated hypotheses, effect sizes)
├── sensitivity.py       # ResearchSensitivityRunner (parameter sweeps, degradation curves)
├── validation.py        # ResearchValidator (provenance, CPU/VRAM separation, checksums)
├── artifacts.py         # PublicationArtifactGenerator (publication figures and tables)
├── report.py            # ResearchReportGenerator (15-section scientific Markdown report)
└── runner.py            # ResearchRunner (campaign executor, resumption, raw logging)
```

---

## 2. Research Lifecycle

```
[configs/research/*.yaml]
         │
         ▼
[ResearchRunner] ─── SeedManager (Hierarchical SHA-256 derivation)
         │
         ├──► ClosedLoopRuntime (Simulation / Execution Mode)
         │
         ▼
[results/research/<experiment_id>/]
         ├── manifest.json (Environment, Config Hash, Checksums)
         ├── resolved_config.yaml
         ├── raw_observations.csv
         ├── policy_comparisons.csv
         ├── tables/ (Table 1: Summary, Table 2: Paired, Table 3: Ablations, Table 4: Provenance)
         ├── figures/ (Fig 1: Latency, Fig 2: Throughput, Fig 3: Overhead, Fig 6: Ablations, Fig 7: Sweeps)
         └── report.md (15-section comprehensive scientific report)
```

1. **Planning & Hashing:** Configuration is validated and cryptographically hashed (`ResearchConfigHasher.hash_dict()`) after stripping volatile metadata.
2. **Deterministic Seed Derivation:** Per-trial seeds are derived via `sha256(root_seed:component:repetition:subtask)`, avoiding stream correlation.
3. **Execution with Immutability:** Trials run with identical environment traces; results persist in `raw/<trial_id>.json` and `raw_observations.csv`.
4. **Resumption:** If interrupted, `--resume` verifies that the candidate configuration hash exactly matches `manifest.json`. Completed runs are skipped without re-execution.
5. **Decoupled Statistical Analysis:** Analysis and artifact compilation run independently on saved raw observations without rerunning expensive model inference.
6. **Integrity Auditing:** `ResearchValidator` verifies artifact checksums, physical plausibility, and provenance rules before producing final publications.

---

## 3. Statistical Analysis Methodology

### Uncertainty Estimation & Estimands
- **Normal Estimands:** Student's t distribution with exact critical values ($t_{\text{crit}} = \text{scipy.stats.t.ppf}(1 - \alpha/2, \text{df})$) for sample means across independent repetitions.
- **Skewed Distributions:** Non-parametric percentile bootstrap confidence intervals ($B = 2000$ resamples) for asymmetric latency observations.
- **Correlated Timestep Guardrail:** Token-level time steps within a single generation sequence are **never** treated as independent observations. Repetition-level means serve as the primary statistical unit.

### Paired Hypothesis Testing & Effect Sizes
- **Paired Student's t-test:** Evaluates matched pairs $(A_i, B_i)$ under identical scenarios and matched random seeds.
- **Wilcoxon Signed-Rank Test:** Non-parametric rank test for non-Gaussian paired differences.
- **Hedges' $g$ Effect Size:** Standardized mean difference corrected for small-sample upward bias:
  $$g = d \cdot \left(1 - \frac{3}{4\,\text{df} - 1}\right)$$
- **Cliff's Delta:** Non-parametric ordinal effect size measuring distribution non-overlap.

### Multiple Testing Corrections
When evaluating multiple pairwise policy comparisons across $m$ metrics and scenarios:
- **Holm-Bonferroni:** Step-down procedure controlling Family-Wise Error Rate (FWER) without requiring test independence.
- **Benjamini-Hochberg FDR:** Step-up procedure controlling False Discovery Rate.

### Small-Sample Safeguards
- $N < 2$: Standard deviation and confidence intervals are undefined (`is_valid = False`).
- $N < 3$: Hypothesis test p-values are suppressed (`None`) with explicit `insufficient_sample` warning.
- $N < 5$: Inferences are flagged with a `small_sample_warning` and labeled as exploratory.
- **Zero-Fabrication Guard:** Missing metrics remain `None` and are **never** replaced with zeros.

---

## 4. Measurement Provenance & Evidence Classification

Every observation retains a strict data provenance label ([`DataSource`](file:///c:/Users/reape/OneDrive/Desktop/Predictive_VRAM/src/telemetry/events.py)):

| Category | Platform Classification | Methodology | Boundary / Limitation |
| :--- | :---: | :--- | :--- |
| **Inference Latency (ITL, TTFT)** | `MEASURED` | High-resolution monotonic timers | Reflects host runtime execution speed. |
| **Host RAM Utilization** | `MEASURED` | `psutil.virtual_memory()` | Physical CPU host memory; **never** substituted for VRAM. |
| **GPU VRAM Allocation** | `UNAVAILABLE` (CPU) / `MEASURED` (CUDA) | `torch.cuda.mem_get_info()` | On CPU-only environments, strictly reported as `None`. |
| **6G Profile Link Dynamics** | `EMULATED` | Deterministic profile replay | Prospective operating conditions; **not** physical 6G radio measurements. |
| **Loopback TCP Transport** | `MEASURED (Loopback)` | OS sockets on `127.0.0.1` | Verifies framing & HMAC; does not model cellular WAN fading. |
| **Energy Consumption** | `ESTIMATED` | Tier proxy power coefficients | Modeled heuristic estimate; not physical power meter telemetry. |

---

## 5. Artifact Directory Structure

Each experiment writes a complete, auditable research package to `results/research/<experiment_id>/`:

```text
results/research/<experiment_id>/
├── manifest.json              # Machine-readable experiment manifest with checksums
├── resolved_config.yaml       # Complete, unredacted experiment configuration
├── raw_observations.csv       # Flattened tabular dataset of all completed trials
├── policy_comparisons.csv     # Paired test statistics, effect sizes, Holm/FDR p-values
├── report.md                  # Comprehensive 15-section scientific research report
├── raw/                       # Immutable per-trial JSON records
│   ├── <trial_id_1>.json
│   └── <trial_id_2>.json
├── tables/
│   ├── table1_policy_summary.md / .csv
│   ├── table2_paired_comparisons.md / .csv
│   ├── table3_ablation_summary.md / .csv
│   └── table4_provenance_matrix.md / .csv
└── figures/
    ├── fig1_latency_distribution.png
    ├── fig2_throughput_comparison.png
    ├── fig3_switching_and_overhead.png
    ├── fig6_ablation_waterfall.png
    └── fig7_sensitivity_sweeps.png
```

---

## 6. Command-Line Interface (CLI)

```powershell
# 1. Run a research experiment campaign
uv run python scripts/run_research_experiment.py --config configs/research/default.yaml --experiment-id campaign_01 --repetitions 3

# 2. Resume an interrupted experiment (skips completed trials)
uv run python scripts/run_research_experiment.py --config configs/research/default.yaml --experiment-id campaign_01 --resume

# 3. Analyze saved observations independently (re-compute statistics, Holm/FDR corrections, figures)
uv run python scripts/analyze_research_results.py --input-dir results/research/campaign_01 --reference-policy predictive

# 4. Audit artifacts for provenance integrity and checksum validation
uv run python scripts/validate_research_artifacts.py --input-dir results/research/campaign_01 --output-json results/research/campaign_01/audit.json

# 5. Compile 15-section publication report
uv run python scripts/generate_research_report.py --input-dir results/research/campaign_01
```

---

## 7. Research Boundaries and Limitations

1. **CPU-Only Environments:** GPU VRAM allocation, reservation, and GPU utilization cannot be measured without physical CUDA hardware. Host RAM must never be substituted.
2. **6G Network Emulation:** Emulated 6G profiles evaluate partition adaptation under prospective sub-THz or high-mobility channel dynamics. They **do not** validate performance over an actual physical 6G radio testbed.
3. **Transport Protocol:** Framing with HMAC-SHA256 and SHA-256 payload integrity guarantees safe serialization over TCP sockets, but TCP kernel dynamics do not replicate lower-layer radio physical channel fading (e.g. beamforming, Doppler shift).
