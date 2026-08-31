from pathlib import Path

import joblib
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

MODEL_PATH = Path("models/model.joblib")
ONNX_PATH = Path("models/model.onnx")


def main() -> None:
    print("Loading scikit-learn pipeline...")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}. "
            "Run the training pipeline first."
        )

    pipeline = joblib.load(MODEL_PATH)

    tfidf = pipeline.named_steps["tfidf"]
    classifier = pipeline.named_steps["classifier"]

    n_features = len(tfidf.get_feature_names_out())

    print(f"TF-IDF features: {n_features}")
    print("Converting Logistic Regression classifier to ONNX...")

    initial_types = [
        (
            "input",
            FloatTensorType([None, n_features]),
        )
    ]

    onnx_model = convert_sklearn(
        classifier,
        initial_types=initial_types,
        target_opset=17,
        options={
            id(classifier): {
                "zipmap": False,
            }
        },
    )

    ONNX_PATH.write_bytes(
        onnx_model.SerializeToString()
    )

    print("Conversion completed successfully.")
    print(f"ONNX model saved to: {ONNX_PATH}")
    print(
        f"Model size: "
        f"{ONNX_PATH.stat().st_size / (1024 * 1024):.2f} MB"
    )


if __name__ == "__main__":
    main()