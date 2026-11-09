
from pathlib import Path
from datetime import datetime, timezone
from time import perf_counter
import json
import logging

import mlflow
import pandas as pd

from src.contracts import validate_credit_batch


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
REPORT_DIR = BASE_DIR / "reports"
LOG_DIR = BASE_DIR / "logs"

LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    filename=LOG_DIR / "pipeline.log",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

mlflow.set_tracking_uri(
    f"sqlite:///{BASE_DIR / 'mlflow.db'}"
)
mlflow.set_experiment("fase-04-pipeline-monitoring")

SCENARIOS = ["baseline", "moderate", "severe"]

# Limiares experimentais para o prototipo
MAX_CLASSIFICATION_CHANGE = 0.05
MAX_DRIFTED_FEATURES = 0


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def monitor_scenario(scenario):
    start = perf_counter()
    alerts = []
    status = "SUCCESS"
    records = 0

    try:
        df = pd.read_csv(
            DATA_DIR / f"production_{scenario}.csv"
        )
        records = len(df)

        # Etapa 1: contrato de qualidade
        validate_credit_batch(df)

        # Etapa 2: resultados estatisticos
        drift = load_json(
            REPORT_DIR / f"drift_ks_{scenario}.json"
        )
        changes = load_json(
            REPORT_DIR / "classification_changes.json"
        )
        predictions = load_json(
            REPORT_DIR / "prediction_comparison.json"
        )

        # Correção de Bonferroni
        alpha = 0.05
        adjusted_alpha = alpha / len(drift)

        drifted_raw = sum(
            item["p_value"] < alpha
            for item in drift.values()
        )

        drifted = sum(
            item["p_value"] < adjusted_alpha
            for item in drift.values()
        )

        change_rate = changes.get(
            scenario, {}
        ).get("change_rate", 0.0)

        positive_rate = predictions[
            scenario
        ]["predicted_positive_rate"]

        if predictions[scenario]["records"] != records:
            raise ValueError(
                "Volume do relatório diferente do dataset"
            )

        # Regras de alerta
        if drifted > MAX_DRIFTED_FEATURES:
            alerts.append("DATA_DRIFT")

        if change_rate > MAX_CLASSIFICATION_CHANGE:
            alerts.append("PREDICTION_CHANGE")

        if alerts:
            status = "WARNING"

        duration = perf_counter() - start

        # Rastreabilidade no MLflow
        with mlflow.start_run(
            run_name=f"pipeline-{scenario}"
        ):
            mlflow.log_param("scenario", scenario)

            mlflow.log_metrics({
                "records_processed": records,
                "duration_seconds": duration,
                "drifted_features": drifted,
                "drifted_features_raw": drifted_raw,
                "classification_change_rate": change_rate,
                "predicted_positive_rate": positive_rate,
                "alert_count": len(alerts),
            })
            mlflow.set_tag("pipeline_status", status)
            mlflow.set_tag(
                "alerts", ",".join(alerts) or "NONE"
            )

        result = {
            "timestamp_utc": datetime.now(
                timezone.utc
            ).isoformat(),
            "scenario": scenario,
            "status": status,
            "records": records,
            "duration_seconds": round(duration, 4),
            "drifted_features": drifted,
            "classification_change_rate": change_rate,
            "alerts": alerts,
        }

    except Exception as exc:
        status = "FAILED"
        duration = perf_counter() - start

        logging.exception(
            "Falha no monitoramento do cenario %s",
            scenario,
        )

        # Registra falhas no MLflow sem impedir o log local
        try:
            mlflow.start_run(
                run_name=f"pipeline-{scenario}-failed"
            )
            try:
                mlflow.log_param("scenario", scenario)
                mlflow.log_metrics({
                    "records_processed": records,
                    "duration_seconds": duration,
                })
                mlflow.set_tag("pipeline_status", "FAILED")
                mlflow.set_tag(
                    "error_type", type(exc).__name__
                )
                mlflow.set_tag(
                    "error_message", str(exc)[:500]
                )
            finally:
                mlflow.end_run(status="FAILED")
        except Exception:
            logging.exception(
                "Falha ao registrar erro no MLflow"
            )

        result = {
            "timestamp_utc": datetime.now(
                timezone.utc
            ).isoformat(),
            "scenario": scenario,
            "status": status,
            "records": records,
            "duration_seconds": round(duration, 4),
            "alerts": alerts,
            "error_type": type(exc).__name__,
            "error_message": str(exc),
        }

    # Log estruturado em JSON Lines
    with (LOG_DIR / "pipeline.jsonl").open(
        "a", encoding="utf-8"
    ) as file:
        file.write(
            json.dumps(result, ensure_ascii=False) + "\n"
        )

    logging.info(
        "Cenario=%s | Status=%s | Registros=%s",
        scenario,
        result["status"],
        result["records"],
    )

    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result


if __name__ == "__main__":
    results = [
        monitor_scenario(s)
        for s in SCENARIOS
    ]

    if any(r["status"] == "FAILED" for r in results):
        raise SystemExit(1)
