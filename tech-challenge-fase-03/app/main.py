from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException

from app.schemas import PredictionRequest, PredictionResponse


BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "models" / "model.joblib"


app = FastAPI(
    title="Medical Text Classification API",
    description="API for classifying medical abstracts using a Machine Learning model.",
    version="1.0.0",
)


if not MODEL_PATH.exists():
    raise RuntimeError(
        f"Model not found at {MODEL_PATH}. "
        "Run the training pipeline before starting the API."
    )

model = joblib.load(MODEL_PATH)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    try:
        prediction = model.predict([request.text])[0]
        probabilities = model.predict_proba([request.text])[0]

        confidence = probabilities.max()

        return PredictionResponse(
            prediction=int(prediction),
            confidence=float(confidence),
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Error while generating prediction.",
        ) from exc