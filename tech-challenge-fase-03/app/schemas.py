from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=10,
        description="Medical abstract used for classification.",
    )


class PredictionResponse(BaseModel):
    prediction: int
    confidence: float