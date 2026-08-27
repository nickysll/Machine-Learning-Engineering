from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_health() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict() -> None:
    payload = {
        "text": (
            "Patient with cardiovascular disease and persistent "
            "chest pain requiring medical evaluation."
        )
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200

    data = response.json()

    assert "prediction" in data
    assert "confidence" in data
    assert 1 <= data["prediction"] <= 5
    assert 0 <= data["confidence"] <= 1


def test_predict_invalid_text() -> None:
    payload = {
        "text": "short"
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 422