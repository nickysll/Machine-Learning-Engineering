from pathlib import Path

import joblib


MODEL_PATH = Path("models/model.joblib")


def test_model_file_exists() -> None:
    assert MODEL_PATH.exists()


def test_model_prediction() -> None:
    model = joblib.load(MODEL_PATH)

    text = (
        "Patient presents with persistent chest pain "
        "and cardiovascular symptoms."
    )

    prediction = model.predict([text])[0]

    assert prediction in [1, 2, 3, 4, 5]


def test_model_probability() -> None:
    model = joblib.load(MODEL_PATH)

    text = (
        "Patient presents with persistent chest pain "
        "and cardiovascular symptoms."
    )

    probabilities = model.predict_proba([text])[0]

    assert len(probabilities) == 5
    assert abs(probabilities.sum() - 1.0) < 1e-6