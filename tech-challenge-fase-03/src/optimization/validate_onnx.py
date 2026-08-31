from pathlib import Path

import joblib
import numpy as np
import onnxruntime as ort
import pandas as pd

MODEL_PATH = Path("models/model.joblib")
ONNX_PATH = Path("models/model.onnx")
TEST_PATH = Path("data/raw/test.parquet")

TEXT_COLUMN = "medical_abstract"


def main() -> None:
    print("Loading models and test data...")

    pipeline = joblib.load(MODEL_PATH)

    tfidf = pipeline.named_steps["tfidf"]

    onnx_session = ort.InferenceSession(
        str(ONNX_PATH),
        providers=["CPUExecutionProvider"],
    )

    test_df = pd.read_parquet(TEST_PATH)

    texts = test_df[TEXT_COLUMN].head(100).tolist()

    # Baseline completo
    sklearn_predictions = pipeline.predict(texts)

    # Mesmo TF-IDF do modelo original
    features = tfidf.transform(texts)

    # ONNX Runtime recebe float32
    onnx_input = (
        features
        .astype(np.float32)
        .toarray()
    )

    input_name = onnx_session.get_inputs()[0].name

    onnx_outputs = onnx_session.run(
        None,
        {
            input_name: onnx_input,
        },
    )

    onnx_predictions = np.asarray(
        onnx_outputs[0]
    ).ravel()

    matches = (
        sklearn_predictions == onnx_predictions
    )

    agreement = matches.mean()

    print("\nPrediction comparison:")
    print(f"Samples compared: {len(texts)}")
    print(f"Matching predictions: {matches.sum()}")
    print(f"Agreement: {agreement:.2%}")

    print("\nFirst 10 predictions:")

    for index in range(10):
        print(
            f"{index}: "
            f"sklearn={sklearn_predictions[index]} | "
            f"onnx={onnx_predictions[index]}"
        )

    if agreement != 1.0:
        raise ValueError(
            "ONNX predictions do not fully match "
            "the scikit-learn pipeline."
        )

    print("\nONNX model validated successfully.")


if __name__ == "__main__":
    main()