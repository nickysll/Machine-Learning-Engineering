import json
from contextlib import nullcontext
from unittest.mock import MagicMock

import pandas as pd
import pytest

from src import monitor_pipeline as mp


@pytest.fixture
def isolated_monitor(monkeypatch, tmp_path):
    monkeypatch.setattr(mp, "LOG_DIR", tmp_path)
    monkeypatch.setattr(mp, "DATA_DIR", tmp_path)
    monkeypatch.setattr(mp, "REPORT_DIR", tmp_path)

    monkeypatch.setattr(mp.mlflow, "start_run", lambda **kwargs: nullcontext())
    monkeypatch.setattr(mp.mlflow, "log_param", MagicMock())
    monkeypatch.setattr(mp.mlflow, "log_metrics", MagicMock())
    monkeypatch.setattr(mp.mlflow, "end_run", MagicMock())
    monkeypatch.setattr(mp.mlflow, "set_tag", MagicMock())

    monkeypatch.setattr(mp, "validate_credit_batch", lambda df: df)

    return tmp_path


@pytest.mark.parametrize(
    "scenario,change_rate,p_values,expected_status,expected_alerts",
    [
        ("baseline", 0.0, [0.8, 0.9], "SUCCESS", []),
        ("moderate", 0.0352, [0.001, 0.9], "WARNING", ["DATA_DRIFT"]),
        (
            "severe",
            0.0847,
            [0.001, 0.002],
            "WARNING",
            ["DATA_DRIFT", "PREDICTION_CHANGE"],
        ),
    ],
)
def test_monitoring_scenarios(
    isolated_monitor,
    monkeypatch,
    scenario,
    change_rate,
    p_values,
    expected_status,
    expected_alerts,
):
    df = pd.DataFrame({"ID": [1, 2, 3]})

    monkeypatch.setattr(mp.pd, "read_csv", lambda path: df)

    reports = {
        f"drift_ks_{scenario}.json": {
            f"feature_{i}": {"p_value": p}
            for i, p in enumerate(p_values)
        },
        "classification_changes.json": {
            scenario: {"change_rate": change_rate}
        },
        "prediction_comparison.json": {
            scenario: {
                "records": len(df),
                "predicted_positive_rate": 0.45,
            }
        },
    }

    monkeypatch.setattr(
        mp,
        "load_json",
        lambda path: reports[path.name],
    )

    result = mp.monitor_scenario(scenario)

    assert result["status"] == expected_status
    assert result["alerts"] == expected_alerts
    assert result["records"] == 3

    logged_metrics = mp.mlflow.log_metrics.call_args.args[0]
    assert logged_metrics["alert_count"] == len(expected_alerts)
    assert logged_metrics["records_processed"] == 3

    saved = json.loads(
        (isolated_monitor / "pipeline.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()[-1]
    )

    assert saved["status"] == expected_status
    assert saved["alerts"] == expected_alerts


def test_monitoring_failure(isolated_monitor, monkeypatch):
    def fail_read_csv(path):
        raise FileNotFoundError("Arquivo de teste inexistente")

    monkeypatch.setattr(mp.pd, "read_csv", fail_read_csv)

    result = mp.monitor_scenario("baseline")

    assert result["status"] == "FAILED"
    assert result["error_type"] == "FileNotFoundError"
    assert "Arquivo de teste inexistente" in result["error_message"]
    assert result["records"] == 0

    saved = json.loads(
        (isolated_monitor / "pipeline.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()[-1]
    )

    assert saved["status"] == "FAILED"
    mp.mlflow.set_tag.assert_any_call(
        "pipeline_status", "FAILED"
    )
    mp.mlflow.set_tag.assert_any_call(
        "error_type", "FileNotFoundError"
    )
    mp.mlflow.end_run.assert_called_once_with(
        status="FAILED"
    )



def test_monitoring_mlflow_unavailable(
    isolated_monitor, monkeypatch
):
    def fail_read_csv(path):
        raise FileNotFoundError("Arquivo inexistente")

    def fail_mlflow(**kwargs):
        raise ConnectionError("MLflow indisponivel")

    monkeypatch.setattr(mp.pd, "read_csv", fail_read_csv)
    monkeypatch.setattr(mp.mlflow, "start_run", fail_mlflow)

    result = mp.monitor_scenario("baseline")

    assert result["status"] == "FAILED"
    assert result["error_type"] == "FileNotFoundError"

    saved = json.loads(
        (isolated_monitor / "pipeline.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()[-1]
    )

    assert saved["status"] == "FAILED"
    mp.mlflow.end_run.assert_not_called()
