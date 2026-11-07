
from pathlib import Path
import json

import joblib
import pandas as pd

from src.config import FEATURES
from src.contracts import validate_credit_batch

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
REPORT_DIR = BASE_DIR / "reports"

model = joblib.load(
    BASE_DIR / "models" / "baseline_credit.joblib"
)

scenarios = ["baseline", "moderate", "severe"]
results = {}

for scenario in scenarios:
    df = pd.read_csv(
        DATA_DIR / f"production_{scenario}.csv"
    )

    # Confirma a qualidade antes de gerar previsões
    validate_credit_batch(df)

    X = df[FEATURES]

    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]

    results[scenario] = {
        "records": len(df),
        "predicted_positive_rate": round(
            float(predictions.mean()), 4
        ),
        "mean_predicted_risk": round(
            float(probabilities.mean()), 4
        ),
        "high_risk_count": int(predictions.sum()),
    }

    print(f"\nCenário: {scenario}")
    print(
        f"Previsões positivas: "
        f"{results[scenario]['predicted_positive_rate']:.2%}"
    )
    print(
        f"Risco médio previsto: "
        f"{results[scenario]['mean_predicted_risk']:.2%}"
    )

REPORT_DIR.mkdir(exist_ok=True)

(REPORT_DIR / "prediction_comparison.json").write_text(
    json.dumps(results, indent=4, ensure_ascii=False),
    encoding="utf-8",
)
