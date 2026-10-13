"""
Research Experiment Runner (Module 12).

Orchestrates reproducible experimental campaigns, manages trial execution
across policies and scenarios, enables resumption without overwriting, and
persists raw observations and publication artifacts.
"""

from __future__ import annotations

import copy
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd
import yaml

from src.evaluation.baselines import BaselinePolicy, BaselinePolicyFactory
from src.evaluation.experiment_config import ExperimentConfig, TrialConfig
from src.evaluation.export import CSVExporter, JSONExporter
from src.evaluation.runner import ExperimentRunner, TrialResult
from src.evaluation.scenario_catalog import ScenarioCatalog
from src.research.ablations import ResearchAblationRunner
from src.research.artifacts import PublicationArtifactGenerator
from src.research.comparisons import PolicyComparisonEngine
from src.research.manifest import ManifestTrialRecord, ResearchManifest
from src.research.report import ResearchReportGenerator
from src.research.reproducibility import ResearchConfigHasher, SeedManager
from src.research.sensitivity import ResearchSensitivityRunner
from src.research.validation import ResearchValidator

logger = logging.getLogger(__name__)


class ResearchRunner:
    """
    Autonomous research experiment campaign executor.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        experiment_id: Optional[str] = None,
        base_output_dir: Path | str = "results/research",
        resume: bool = False,
    ) -> None:
        self.config = copy.deepcopy(config)
        self.experiment_id = experiment_id or self.config.get("experiment_id", f"exp_{int(time.time())}")
        self.config["experiment_id"] = self.experiment_id

        self.root_seed = int(self.config.get("seed", 42))
        self.seed_manager = SeedManager(root_seed=self.root_seed)

        self.output_dir = Path(base_output_dir) / self.experiment_id
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.raw_dir = self.output_dir / "raw"
        self.raw_dir.mkdir(parents=True, exist_ok=True)

        self.manifest_path = self.output_dir / "manifest.json"
        self.resolved_config_path = self.output_dir / "resolved_config.yaml"

        # Initialize or load manifest
        if resume and self.manifest_path.is_file():
            self.manifest = ResearchManifest.load(self.manifest_path)
            valid, reason = self.manifest.validate_for_resume(self.config)
            if not valid:
                raise ValueError(f"Cannot resume experiment: {reason}")
            logger.info(f"Resuming experiment {self.experiment_id} with {len(self.manifest.trials)} recorded trials.")
        else:
            self.manifest = ResearchManifest(
                experiment_id=self.experiment_id,
                title=self.config.get("title", "Dynamic Split Inference Research Campaign"),
                resolved_config=self.config,
                root_seed=self.root_seed,
                scenarios=self.config.get("scenarios", ["stable"]),
                policies=self.config.get("policies", ["static", "predictive"]),
                requested_repetitions=int(self.config.get("repetitions", 1)),
            )
            # Save resolved config
            with open(self.resolved_config_path, "w", encoding="utf-8") as f:
                yaml.dump(self.config, f, default_flow_style=False)

    def execute_campaign(self) -> Dict[str, Any]:
        """
        Execute all planned trials in the campaign, record raw outputs,
        and generate statistical artifacts.
        """
        completed_trial_ids = self.manifest.get_completed_trial_ids()
        scenarios = self.manifest.scenarios
        policies = self.manifest.policies
        repetitions = self.manifest.requested_repetitions
        execution_mode = self.config.get("execution_mode", "simulation")

        # Plan trials
        trial_plans: List[TrialConfig] = []
        for scen_id in scenarios:
            for pol_id in policies:
                for rep in range(repetitions):
                    derived_seed = self.seed_manager.derive_seed("trial", rep, f"{scen_id}:{pol_id}")
                    trial_id = f"{self.experiment_id}_{scen_id}_{pol_id}_rep{rep}_s{derived_seed}"

                    tc = TrialConfig(
                        trial_id=trial_id,
                        experiment_id=self.experiment_id,
                        scenario_id=scen_id,
                        baseline_id=pol_id,
                        seed=derived_seed,
                        repetition=rep,
                        switch_threshold=float(self.config.get("switch_threshold", 0.25)),
                        control_interval_tokens=int(self.config.get("control_interval_tokens", 2)),
                        horizon_steps=int(self.config.get("horizon_steps", 4)),
                        predictor_type=str(self.config.get("predictor_type", "linear_trend")),
                        max_new_tokens=int(self.config.get("max_new_tokens", 16)),
                        n_trace_steps=int(self.config.get("n_trace_steps", 64)),
                        sampling_interval_s=float(self.config.get("sampling_interval_s", 0.25)),
                        execution_mode=execution_mode,
                        cost_weights=self.config.get("cost_weights", {
                            "alpha": 0.35, "beta": 0.20, "gamma": 0.25, "delta": 0.10, "epsilon": 0.10,
                        }),
                        ablation_id="",
                    )
                    trial_plans.append(tc)

        logger.info(f"Planned {len(trial_plans)} trials for experiment {self.experiment_id}.")

        executed_results: List[TrialResult] = []
        raw_rows: List[Dict[str, Any]] = []

        # Execute trials using ExperimentRunner
        for tc in trial_plans:
            if tc.trial_id in completed_trial_ids:
                logger.info(f"Skipping already completed trial {tc.trial_id}")
                # Load existing trial result
                raw_file = self.raw_dir / f"{tc.trial_id}.json"
                if raw_file.is_file():
                    try:
                        import json
                        with open(raw_file, "r", encoding="utf-8") as f:
                            t_data = json.load(f)
                        raw_rows.append(CSVExporter.flatten_trial(t_data))
                    except Exception:
                        pass
                continue

            record = ManifestTrialRecord(
                trial_id=tc.trial_id,
                scenario_id=tc.scenario_id,
                baseline_id=tc.baseline_id,
                seed=tc.seed,
                repetition=tc.repetition,
                started_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            )

            t_start = time.perf_counter()
            try:
                # Set global seeds deterministically for this trial
                self.seed_manager.set_global_seeds(tc.seed)

                # Execute trial via ExperimentRunner
                runner = ExperimentRunner(raw_results_dir=self.raw_dir)
                trial_res = runner.run_trial(tc)

                duration = time.perf_counter() - t_start
                record.completed_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                record.execution_duration_sec = round(duration, 4)
                record.status = "completed"
                record.output_artifact_path = f"raw/{tc.trial_id}.json"

                executed_results.append(trial_res)
                raw_rows.append(CSVExporter.flatten_trial(trial_res.to_dict()))

            except Exception as e:
                duration = time.perf_counter() - t_start
                record.status = "failed"
                record.error_message = str(e)
                import traceback
                record.error_trace = traceback.format_exc()
                logger.error(f"Trial {tc.trial_id} failed: {e}")

            self.manifest.add_trial_record(record)
            self.manifest.save(self.manifest_path)

        # Build raw observations DataFrame
        trials_df = pd.DataFrame(raw_rows) if raw_rows else pd.DataFrame()
        raw_csv_path = self.output_dir / "raw_observations.csv"
        if not trials_df.empty:
            trials_df.to_csv(raw_csv_path, index=False)
            self.manifest.record_artifact_checksum("raw_observations.csv", raw_csv_path)

        # Statistical analysis & comparisons
        comparisons_df = pd.DataFrame()
        if not trials_df.empty and len(policies) > 1:
            all_cmp_rows = []
            for scen_id in scenarios:
                rep = PolicyComparisonEngine.compare_scenario_trials(
                    scenario_id=scen_id,
                    trials_df=trials_df,
                    reference_policy="predictive" if "predictive" in policies else policies[0],
                )
                cmp_df = PolicyComparisonEngine.comparisons_to_dataframe(rep.comparisons)
                if not cmp_df.empty:
                    cmp_df["scenario_id"] = scen_id
                    all_cmp_rows.append(cmp_df)

            if all_cmp_rows:
                comparisons_df = pd.concat(all_cmp_rows, ignore_index=True)
                cmp_csv_path = self.output_dir / "policy_comparisons.csv"
                comparisons_df.to_csv(cmp_csv_path, index=False)
                self.manifest.record_artifact_checksum("policy_comparisons.csv", cmp_csv_path)

        # Generate publication figures, tables, and provenance matrix
        is_cpu = not self.manifest.environment.cuda_available
        artifacts = PublicationArtifactGenerator.generate_all_artifacts(
            trials_df=trials_df,
            output_dir=self.output_dir,
            comparisons_df=comparisons_df if not comparisons_df.empty else None,
            is_cpu_only=is_cpu,
        )

        for rel_k, abs_v in artifacts.items():
            self.manifest.record_artifact_checksum(rel_k, abs_v)

        # Generate 15-section scientific research report
        report_path = self.output_dir / "report.md"
        ResearchReportGenerator.generate_report(
            manifest=self.manifest,
            trials_df=trials_df,
            comparisons_df=comparisons_df if not comparisons_df.empty else None,
            output_path=report_path,
        )
        self.manifest.record_artifact_checksum("report.md", report_path)

        # Final manifest save with all checksums
        self.manifest.save(self.manifest_path)

        return {
            "experiment_id": self.experiment_id,
            "output_dir": str(self.output_dir),
            "manifest_path": str(self.manifest_path),
            "total_trials": len(trial_plans),
            "completed": self.manifest.completed_repetitions,
            "failed": self.manifest.failed_repetitions,
            "report_path": str(report_path),
        }
