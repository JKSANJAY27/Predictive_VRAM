# Table 1: Comprehensive Inference Policy Summary

| Policy | Metric | Unit | N | Mean | 95% CI | Median | IQR | Std Dev |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| joint_reactive | `mean_itl_ms` | milliseconds | 4 | 10.50 | [10.50, 10.50] | 10.50 | [10.50, 10.50] | 0.00 |
| joint_reactive | `median_itl_ms` | milliseconds | 4 | 10.50 | [10.50, 10.50] | 10.50 | [10.50, 10.50] | 0.00 |
| joint_reactive | `p95_itl_ms` | milliseconds | 4 | 10.50 | [10.50, 10.50] | 10.50 | [10.50, 10.50] | 0.00 |
| joint_reactive | `tokens_per_second` | tokens/sec | 4 | 502010.04 | [427247.99, 576772.09] | 484899.40 | [480382.96, 506526.48] | 46984.01 |
| joint_reactive | `total_switches` | count | 4 | 0.00 | [0.00, 0.00] | 0.00 | [0.00, 0.00] | 0.00 |
| joint_reactive | `total_migration_time_ms` | milliseconds | 4 | 0.00 | [0.00, 0.00] | 0.00 | [0.00, 0.00] | 0.00 |
| joint_reactive | `total_transferred_mb` | megabytes | 0 | nan | N/A | nan | N/A | nan |
| joint_reactive | `total_control_overhead_ms` | milliseconds | 0 | nan | N/A | nan | N/A | nan |
| joint_reactive | `peak_memory_pressure` | ratio [0.0-1.0] | 0 | nan | N/A | nan | N/A | nan |
| predictive | `mean_itl_ms` | milliseconds | 4 | 10.50 | [10.50, 10.50] | 10.50 | [10.50, 10.50] | 0.00 |
| predictive | `median_itl_ms` | milliseconds | 4 | 10.50 | [10.50, 10.50] | 10.50 | [10.50, 10.50] | 0.00 |
| predictive | `p95_itl_ms` | milliseconds | 4 | 10.50 | [10.50, 10.50] | 10.50 | [10.50, 10.50] | 0.00 |
| predictive | `tokens_per_second` | tokens/sec | 4 | 512289.86 | [451040.76, 573538.95] | 523813.26 | [501505.20, 534597.92] | 38491.83 |
| predictive | `total_switches` | count | 4 | 0.00 | [0.00, 0.00] | 0.00 | [0.00, 0.00] | 0.00 |
| predictive | `total_migration_time_ms` | milliseconds | 4 | 0.00 | [0.00, 0.00] | 0.00 | [0.00, 0.00] | 0.00 |
| predictive | `total_transferred_mb` | megabytes | 0 | nan | N/A | nan | N/A | nan |
| predictive | `total_control_overhead_ms` | milliseconds | 0 | nan | N/A | nan | N/A | nan |
| predictive | `peak_memory_pressure` | ratio [0.0-1.0] | 0 | nan | N/A | nan | N/A | nan |
| static | `mean_itl_ms` | milliseconds | 4 | 10.50 | [10.50, 10.50] | 10.50 | [10.50, 10.50] | 0.00 |
| static | `median_itl_ms` | milliseconds | 4 | 10.50 | [10.50, 10.50] | 10.50 | [10.50, 10.50] | 0.00 |
| static | `p95_itl_ms` | milliseconds | 4 | 10.50 | [10.50, 10.50] | 10.50 | [10.50, 10.50] | 0.00 |
| static | `tokens_per_second` | tokens/sec | 4 | 430418.58 | [-10282.13, 871119.29] | 535162.02 | [380451.21, 585129.38] | 276957.20 |
| static | `total_switches` | count | 4 | 0.00 | [0.00, 0.00] | 0.00 | [0.00, 0.00] | 0.00 |
| static | `total_migration_time_ms` | milliseconds | 4 | 0.00 | [0.00, 0.00] | 0.00 | [0.00, 0.00] | 0.00 |
| static | `total_transferred_mb` | megabytes | 0 | nan | N/A | nan | N/A | nan |
| static | `total_control_overhead_ms` | milliseconds | 0 | nan | N/A | nan | N/A | nan |
| static | `peak_memory_pressure` | ratio [0.0-1.0] | 0 | nan | N/A | nan | N/A | nan |

*Note: Physical GPU VRAM is explicitly UNAVAILABLE (value=None) in CPU environments.*
