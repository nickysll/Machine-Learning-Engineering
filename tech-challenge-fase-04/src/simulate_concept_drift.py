
"""Simulação acadêmica de Concept Drift com rótulos sintéticos."""

import json

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, recall_score, roc_auc_score
from src.config import DATA_DIR, MODEL_DIR, REPORT_DIR, FEATURES, TARGET, SEED
from src.contracts import validate_credit_batch


def simulate_labels(reference, eligible, random_values, probability):
    """Altera apenas rótulos selecionados, preservando todas as features."""
    simulated = reference.copy()
    flip_mask = eligible & (random_values < probability)
    simulated.loc[flip_mask, TARGET] = 1

    assert simulated[FEATURES].equals(reference[FEATURES])
    assert simulated["ID"].equals(reference["ID"])

    return simulated, int(flip_mask.sum())


def main():
    reference = pd.read_csv(DATA_DIR / "production_holdout.csv")
    validate_credit_batch(reference)

    if not reference[TARGET].isin([0, 1]).all():
        raise ValueError("O target precisa ser binário.")

    model = joblib.load(MODEL_DIR / "baseline_credit.joblib")

    # Mesmo modelo e mesmas características nos três cenários
    predictions = model.predict(reference[FEATURES])
    probabilities = model.predict_proba(reference[FEATURES])[:, 1]

    # Subgrupo escolhido para uma perturbação condicional sintética
    group = (
        (reference["PAY_0"] <= 0)
        & (reference["LIMIT_BAL"] <= 100000)
    )

    # Apenas exemplos originalmente negativos podem ser convertidos
    eligible = group & reference[TARGET].eq(0)

    if eligible.sum() == 0:
        raise ValueError("Nenhum registro elegível para simulação.")

    # Mesmos números aleatórios para comparar os cenários
    rng = np.random.default_rng(SEED)
    random_values = rng.random(len(reference))

    scenarios = {
        "baseline": 0.00,
        "moderate": 0.15,
        "severe": 0.30,
    }

    results = {
        "experiment": "synthetic_concept_drift",
        "records": len(reference),
        "group_size": int(group.sum()),
        "eligible_negative_labels": int(eligible.sum()),
        "features_unchanged": True,
        "method": "Conditional synthetic label flipping",
        "scenarios": {},
    }

    for name, probability in scenarios.items():
        simulated, changed = simulate_labels(
            reference, eligible, random_values, probability
        )

        y = simulated[TARGET]

        metrics = {
            "flip_probability": probability,
            "labels_changed": changed,
            "overall_default_rate": round(float(y.mean()), 4),
            "group_default_rate": round(float(y[group].mean()), 4),
            "outside_group_default_rate": round(float(y[~group].mean()), 4),
            "accuracy": round(float(accuracy_score(y, predictions)), 4),
            "recall": round(float(recall_score(y, predictions)), 4),
            "roc_auc": round(float(roc_auc_score(y, probabilities)), 4),
        }

        results["scenarios"][name] = metrics

        print(f"\nCenário: {name.upper()}")
        print(f"Rótulos alterados: {changed}")
        print(f"Inadimplência do subgrupo: {metrics['group_default_rate']:.2%}")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"ROC-AUC: {metrics['roc_auc']:.4f}")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    output = REPORT_DIR / "concept_drift_comparison.json"
    output.write_text(
        json.dumps(results, indent=4, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"\nRelatório salvo: {output}")


if __name__ == "__main__":
    main()
