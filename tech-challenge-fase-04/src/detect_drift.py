
from pathlib import Path
import json

import pandas as pd
from scipy.stats import ks_2samp

from evidently import Report
from evidently.presets import DataDriftPreset


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
REPORT_DIR = BASE_DIR / "reports"

REPORT_DIR.mkdir(exist_ok=True)

FEATURES = [
    "LIMIT_BAL",
    "AGE",
    "PAY_0",
    "PAY_2",
    "BILL_AMT1",
    "PAY_AMT1",
]

SCENARIOS = ["baseline", "moderate", "severe"]

reference = pd.read_csv(DATA_DIR / "train.csv")[FEATURES]


def run_drift_analysis(scenario: str):
    current = pd.read_csv(
        DATA_DIR / f"production_{scenario}.csv"
    )[FEATURES]

    # Relatório Evidently
    report = Report([
        DataDriftPreset()
    ])

    snapshot = report.run(
        current_data=current,
        reference_data=reference,
    )

    snapshot.save_html(
        str(REPORT_DIR / f"drift_{scenario}.html")
    )

    # Teste Kolmogorov-Smirnov
    results = {}

    for feature in FEATURES:
        statistic, p_value = ks_2samp(
            reference[feature],
            current[feature],
        )

        results[feature] = {
            "ks_statistic": float(statistic),
            "p_value": float(p_value),
            "drift_detected": bool(p_value < 0.05),
        }

    output = REPORT_DIR / f"drift_ks_{scenario}.json"

    with open(output, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=4)

    print(f"\nCenário: {scenario}")

    for feature, result in results.items():
        status = (
            "DRIFT"
            if result["drift_detected"]
            else "ESTÁVEL"
        )

        print(
            f"{feature}: {status} | "
            f"KS={result['ks_statistic']:.4f} | "
            f"p={result['p_value']:.6g}"
        )


if __name__ == "__main__":
    for scenario in SCENARIOS:
        run_drift_analysis(scenario)
