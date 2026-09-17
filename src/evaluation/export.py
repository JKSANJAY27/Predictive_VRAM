"""
Data export and research dataset builder for Module 10.

Exports raw trials and aggregated statistics to CSV and JSON formats:
  - JSONExporter: structured preservation of full traces and metrics
  - CSVExporter: flattened columnar formats for tabular and statistical packages
  - ResearchDatasetBuilder: consolidates entire experimental campaigns into clean datasets
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union


class JSONExporter:
    """Exports structured JSON data with indentation and schema metadata."""

    @staticmethod
    def export_trial(trial_data: Dict[str, Any], path: Path | str) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "w", encoding="utf-8") as f:
            json.dump(trial_data, f, indent=2, default=str)
        return target

    @staticmethod
    def export_aggregated(aggregated_data: Dict[str, Any], path: Path | str) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "w", encoding="utf-8") as f:
            json.dump(aggregated_data, f, indent=2, default=str)
        return target


class CSVExporter:
    """Exports flat, columnar CSV tables suitable for statistical analysis."""

    CORE_COLUMNS = [
        "trial_id",
        "experiment_id",
        "scenario_id",
        "baseline_id",
        "seed",
        "repetition",
        "ablation_id",
        "execution_mode",
        "switch_threshold",
        "control_interval_tokens",
        "horizon_steps",
        "predictor_type",
        "max_new_tokens",
        # Latency
        "ttft_ms",
        "mean_itl_ms",
        "median_itl_ms",
        "p90_itl_ms",
        "p95_itl_ms",
        "p99_itl_ms",
        "itl_std_ms",
        # Throughput
        "throughput_tokens_per_sec",
        # SLO
        "slo_violation_count",
        "slo_violation_rate",
        # Adaptation & Stability
        "total_switches",
        "proactive_switches",
        "reactive_switches",
        "emergency_switches",
        "switch_frequency_per_token",
        "oscillation_count",
        "flip_flop_rate",
        # Migration Cost
        "total_migration_time_ms",
        "mean_migration_time_ms",
        "total_transferred_bytes",
        "total_transferred_mb",
        # Memory
        "peak_memory_pressure",
        "mean_memory_pressure",
        # Overhead
        "total_control_overhead_ms",
        "control_overhead_fraction",
    ]

    @classmethod
    def flatten_trial(cls, trial_data: Dict[str, Any]) -> Dict[str, Any]:
        """Flatten a single trial result dictionary into a 1D tabular row."""
        row: Dict[str, Any] = {}

        # Identifiers
        row["trial_id"] = trial_data.get("trial_id", "")
        row["experiment_id"] = trial_data.get("experiment_id", "")
        row["scenario_id"] = trial_data.get("scenario_id", "")
        row["baseline_id"] = trial_data.get("baseline_id", "")
        row["seed"] = trial_data.get("seed", 0)
        row["repetition"] = trial_data.get("repetition", 0)
        row["ablation_id"] = trial_data.get("ablation_id", "")

        # Config knobs
        cfg = trial_data.get("configuration", {})
        row["execution_mode"] = cfg.get("execution_mode", "")
        row["switch_threshold"] = cfg.get("switch_threshold", 0.25)
        row["control_interval_tokens"] = cfg.get("control_interval_tokens", 4)
        row["horizon_steps"] = cfg.get("horizon_steps", 4)
        row["predictor_type"] = cfg.get("predictor_type", "")
        row["max_new_tokens"] = cfg.get("max_new_tokens", 32)

        # Metrics
        m = trial_data.get("metrics", {})
        for col in cls.CORE_COLUMNS:
            if col not in row:
                row[col] = m.get(col, "")

        # Any extra metrics not in CORE_COLUMNS
        for k, v in m.items():
            if k not in row and not isinstance(v, (dict, list)):
                row[k] = v

        return row

    @classmethod
    def export_trials_to_csv(
        cls,
        trials: Sequence[Dict[str, Any]],
        path: Path | str,
    ) -> Path:
        """Export a collection of trial results to a CSV file."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)

        rows = [cls.flatten_trial(t) for t in trials]
        if not rows:
            with open(target, "w", newline="", encoding="utf-8") as f:
                f.write(",".join(cls.CORE_COLUMNS) + "\n")
            return target

        fieldnames = list(rows[0].keys())
        # Ensure all columns in any row are included
        for r in rows:
            for k in r.keys():
                if k not in fieldnames:
                    fieldnames.append(k)

        with open(target, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in rows:
                writer.writerow(r)

        return target

    @classmethod
    def export_aggregated_to_csv(
        cls,
        aggregated_results: Sequence[Dict[str, Any]],
        path: Path | str,
    ) -> Path:
        """Export aggregated baseline metrics table to CSV."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = [
            "baseline_id",
            "scenario_id",
            "n_trials",
            "p95_itl_mean",
            "p95_itl_std",
            "mean_itl_mean",
            "mean_itl_std",
            "slo_violation_count_mean",
            "total_switches_mean",
            "proactive_switches_mean",
            "migration_time_ms_mean",
            "control_overhead_pct_mean",
        ]

        rows = []
        for agg in aggregated_results:
            row = {
                "baseline_id": agg.get("baseline_id", ""),
                "scenario_id": agg.get("scenario_id", ""),
                "n_trials": agg.get("n_trials", 0),
            }
            mstats = agg.get("metrics", {})
            for m_key, field_prefix in [
                ("p95_itl_ms", "p95_itl"),
                ("mean_itl_ms", "mean_itl"),
                ("slo_violation_count", "slo_violation_count"),
                ("total_switches", "total_switches"),
                ("proactive_switches", "proactive_switches"),
                ("total_migration_time_ms", "migration_time_ms"),
            ]:
                stat = mstats.get(m_key, {})
                row[f"{field_prefix}_mean"] = stat.get("mean", "")
                row[f"{field_prefix}_std"] = stat.get("std", "")

            oh_stat = mstats.get("control_overhead_fraction", {})
            oh_mean = oh_stat.get("mean", 0.0)
            row["control_overhead_pct_mean"] = round(oh_mean * 100.0, 3) if isinstance(oh_mean, (int, float)) else ""
            rows.append(row)

        with open(target, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in rows:
                writer.writerow(r)

        return target


class ResearchDatasetBuilder:
    """
    Builds a clean, publication-ready research dataset from raw trial files.
    """

    def __init__(self, raw_results_dir: Path | str = "results/raw") -> None:
        self.raw_dir = Path(raw_results_dir)

    def load_all_trials(self) -> List[Dict[str, Any]]:
        """Load all valid trial JSON files from raw results directory."""
        trials = []
        if not self.raw_dir.is_dir():
            return trials

        for p in sorted(self.raw_dir.glob("*.json")):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, dict) and data.get("trial_id"):
                        trials.append(data)
            except Exception:
                continue
        return trials

    def build_dataset(
        self,
        output_dir: Path | str = "results/export",
        experiment_id: Optional[str] = None,
    ) -> Dict[str, Path]:
        """
        Consolidate raw trials into trials.csv, trials.json, and summary metadata.
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        trials = self.load_all_trials()
        if experiment_id:
            trials = [t for t in trials if t.get("experiment_id") == experiment_id]

        csv_path = CSVExporter.export_trials_to_csv(trials, out / "trials.csv")
        json_path = out / "trials.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(trials, f, indent=2, default=str)

        meta_path = out / "dataset_metadata.json"
        metadata = {
            "n_trials": len(trials),
            "experiment_id": experiment_id or "all",
            "unique_baselines": sorted(list({t.get("baseline_id", "") for t in trials})),
            "unique_scenarios": sorted(list({t.get("scenario_id", "") for t in trials})),
            "csv_path": str(csv_path),
            "json_path": str(json_path),
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        return {
            "csv": csv_path,
            "json": json_path,
            "metadata": meta_path,
        }

    def to_pandas(self, experiment_id: Optional[str] = None) -> Any:
        """
        Load trials into a pandas DataFrame if pandas is installed.
        Returns a list of dicts if pandas is not available.
        """
        trials = self.load_all_trials()
        if experiment_id:
            trials = [t for t in trials if t.get("experiment_id") == experiment_id]
        rows = [CSVExporter.flatten_trial(t) for t in trials]
        try:
            import pandas as pd
            return pd.DataFrame(rows)
        except ImportError:
            return rows
