
"""Testes da simulação sintética de Concept Drift."""

import numpy as np
import pandas as pd

from src.config import FEATURES, TARGET
from src.simulate_concept_drift import simulate_labels


def test_simulation_preserves_features_and_changes_only_eligible():
    reference = pd.DataFrame({
        "ID": [1, 2, 3, 4],
        **{feature: [10, 20, 30, 40] for feature in FEATURES},
        TARGET: [0, 0, 1, 0],
    })

    original = reference.copy(deep=True)

    eligible = pd.Series([True, True, False, False])
    random_values = np.array([0.10, 0.70, 0.20, 0.30])

    moderate, changed_m = simulate_labels(
        reference, eligible, random_values, 0.50
    )

    severe, changed_s = simulate_labels(
        reference, eligible, random_values, 1.00
    )

    # Quantidade de rótulos alterados
    assert changed_m == 1
    assert changed_s == 2

    # Rótulos esperados
    assert moderate[TARGET].tolist() == [1, 0, 1, 0]
    assert severe[TARGET].tolist() == [1, 1, 1, 0]

    # Nenhuma característica ou ID pode mudar
    for simulated in [moderate, severe]:
        assert simulated.drop(columns=TARGET).equals(
            reference.drop(columns=TARGET)
        )

    # A base original não pode ser modificada
    assert reference.equals(original)


def test_baseline_does_not_change_labels():
    reference = pd.DataFrame({
        "ID": [1, 2],
        **{feature: [10, 20] for feature in FEATURES},
        TARGET: [0, 1],
    })

    eligible = pd.Series([True, False])
    random_values = np.array([0.20, 0.80])

    baseline, changed = simulate_labels(
        reference, eligible, random_values, 0.00
    )

    assert changed == 0
    assert baseline.equals(reference)
