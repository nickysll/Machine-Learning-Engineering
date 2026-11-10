
"""Registra a simulação de Concept Drift no MLflow."""

import json
from pathlib import Path

import mlflow

BASE_DIR = Path(__file__).resolve().parents[1]
REPORT_PATH = BASE_DIR / "reports" / "concept_drift_comparison.json"

def main():
    if not REPORT_PATH.exists():
        raise FileNotFoundError(
            "Execute primeiro: python -m src.simulate_concept_drift"
        )

    results = json.loads(REPORT_PATH.read_text(encoding="utf-8"))

    mlflow.set_tracking_uri(f"sqlite:///{BASE_DIR / 'mlflow.db'}")
    mlflow.set_experiment("fase-04-concept-drift")

    for scenario, metrics in results["scenarios"].items():
        with mlflow.start_run(run_name=f"concept-{scenario}"):
            mlflow.log_params({
                "scenario": scenario,
                "simulation_type": "synthetic_conditional_label_shift",
                "features_unchanged": True,
                "model_retrained": False,
                "affected_subgroup": "PAY_0 <= 0 and LIMIT_BAL <= 100000",
            })

            mlflow.log_metrics({
                **metrics,
                "records": results["records"],
                "group_size": results["group_size"],
                "eligible_negative_labels": results[
                    "eligible_negative_labels"
                ],
            })

            mlflow.set_tag("drift_type", "CONCEPT_DRIFT_SYNTHETIC")
            mlflow.set_tag(
                "interpretation",
                "Synthetic labels; not observed real-world concept drift",
            )

            mlflow.log_artifact(str(REPORT_PATH), artifact_path="reports")

        print(f"{scenario}: registrado no MLflow")

    print("\nExperimento fase-04-concept-drift concluído.")


if __name__ == "__main__":
    main()
