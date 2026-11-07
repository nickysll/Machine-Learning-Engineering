
from pathlib import Path
import json

import joblib
import pandas as pd

from src.config import FEATURES

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
REPORT_DIR = BASE_DIR / "reports"

model = joblib.load(
    BASE_DIR / "models" / "baseline_credit.joblib"
)

predictions = {}

for scenario in ["baseline", "moderate", "severe"]:
    df = pd.read_csv(
        DATA_DIR / f"production_{scenario}.csv"
    )

    predictions[scenario] = pd.Series(
        model.predict(df[FEATURES]),
        index=df["ID"],
    ).sort_index()

baseline = predictions["baseline"]
results = {}

for scenario in ["moderate", "severe"]:
    current = predictions[scenario]

    assert baseline.index.equals(current.index)

    changed = baseline != current
    increased = (baseline == 0) & (current == 1)
    decreased = (baseline == 1) & (current == 0)

    results[scenario] = {
        "total_records": int(len(baseline)),
        "changed_classifications": int(changed.sum()),
        "change_rate": round(float(changed.mean()), 4),
        "changed_to_positive": int(increased.sum()),
        "changed_to_negative": int(decreased.sum()),
    }

    print(f"\nCenário: {scenario}")
    print(f"Classificações alteradas: {changed.sum()}")
    print(f"Taxa de mudança: {changed.mean():.2%}")
    print(f"Classe 0 → 1: {increased.sum()}")
    print(f"Classe 1 → 0: {decreased.sum()}")

REPORT_DIR.mkdir(exist_ok=True)

(REPORT_DIR / "classification_changes.json").write_text(
    json.dumps(results, indent=4),
    encoding="utf-8",
)
