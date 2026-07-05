import pytest

from evaluation.metrics import compute_classification_metrics


def test_compute_classification_metrics_returns_expected_values() -> None:
    """Test if classification metrics are computed correctly."""
    y_true = [1, 0, 1, 0]
    y_pred = [1, 0, 0, 0]
    y_proba = [0.9, 0.2, 0.4, 0.1]

    result = compute_classification_metrics(y_true, y_pred, y_proba)

    assert result["accuracy"] == pytest.approx(0.75)
    assert result["precision"] == pytest.approx(1.0)
    assert result["recall"] == pytest.approx(0.5)
    assert result["f1_score"] == pytest.approx(0.666666, rel=1e-3)
    assert result["roc_auc"] == pytest.approx(1.0)
