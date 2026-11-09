
from pathlib import Path
import json

import mlflow

BASE_DIR = Path(__file__).resolve().parents[1]
REPORTS = BASE_DIR / "reports"

mlflow.set_tracking_uri(
    f"sqlite:///{BASE_DIR / 'mlflow.db'}"
)

mlflow.set_experiment("fase-04-credit-drift")

predictions = json.loads(
    (REPORTS / "prediction_comparison.json").read_text()
)

changes = json.loads(
    (REPORTS / "classification_changes.json").read_text()
)

for scenario in ["baseline", "moderate", "severe"]:
    drift = json.loads(
        (REPORTS / f"drift_ks_{scenario}.json").read_text()
    )

    metrics = {
        "predicted_positive_rate":
            predictions[scenario]["predicted_positive_rate"],
        "mean_predicted_risk":
            predictions[scenario]["mean_predicted_risk"],
        "records":
            predictions[scenario]["records"],
        "drifted_features_raw":
            sum(v["p_value"] < 0.05 for v in drift.values()),
        "drifted_features":
            sum(v["p_value"] < 0.05 / len(drift) for v in drift.values()),
        "max_ks":
            max(v["ks_statistic"] for v in drift.values()),
        "classification_change_rate":
            changes.get(scenario, {}).get("change_rate", 0),
    }

    with mlflow.start_run(run_name=f"drift-{scenario}"):
        mlflow.log_param("scenario", scenario)
        mlflow.log_param("drift_test", "KS")
        mlflow.log_param("significance_level", 0.05)
        mlflow.log_param("multiple_testing_correction", "bonferroni")
        mlflow.log_param("adjusted_alpha", 0.05 / len(drift))

        mlflow.log_metrics(metrics)

        mlflow.log_artifact(
            str(REPORTS / f"drift_ks_{scenario}.json")
        )

        html = REPORTS / f"drift_{scenario}.html"
        if html.exists():
            mlflow.log_artifact(str(html))

        print(f"{scenario}: registrado no MLflow")
