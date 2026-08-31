import json
import time
from pathlib import Path

import joblib
import numpy as np
import onnxruntime as ort
import pandas as pd

MODEL_PATH = Path("models/model.joblib")
ONNX_PATH = Path("models/model.onnx")
TEST_PATH = Path("data/raw/test.parquet")
RESULT_PATH = Path("models/benchmark_classifier.json")

TEXT_COLUMN = "medical_abstract"

N_SAMPLES = 300
WARMUP_SAMPLES = 20


def calculate_metrics(
    latencies_ms: list[float],
) -> dict[str, float]:
    values = np.array(latencies_ms)

    return {
        "mean_ms": float(values.mean()),
        "p50_ms": float(np.percentile(values, 50)),
        "p95_ms": float(np.percentile(values, 95)),
        "p99_ms": float(np.percentile(values, 99)),
        "throughput_requests_per_second": float(
            1000 / values.mean()
        ),
    }


def main() -> None:
    print("Loading models and data...")

    pipeline = joblib.load(MODEL_PATH)

    tfidf = pipeline.named_steps["tfidf"]
    classifier = pipeline.named_steps["classifier"]

    session = ort.InferenceSession(
        str(ONNX_PATH),
        providers=["CPUExecutionProvider"],
    )

    input_name = session.get_inputs()[0].name

    test_df = pd.read_parquet(TEST_PATH)

    texts = (
        test_df[TEXT_COLUMN]
        .head(N_SAMPLES + WARMUP_SAMPLES)
        .tolist()
    )

    print("Precomputing TF-IDF features...")

    features_sparse = tfidf.transform(texts)

    features_dense = (
        features_sparse
        .astype(np.float32)
        .toarray()
    )

    # Warm-up
    for index in range(WARMUP_SAMPLES):
        classifier.predict(
            features_sparse[index]
        )

        session.run(
            None,
            {
                input_name:
                features_dense[index:index + 1]
            },
        )

    sklearn_latencies = []
    onnx_latencies = []

    print("Benchmarking classifiers...")

    for index in range(
        WARMUP_SAMPLES,
        len(texts),
    ):
        sparse_sample = features_sparse[index]

        dense_sample = features_dense[
            index:index + 1
        ]

        start = time.perf_counter()

        classifier.predict(sparse_sample)

        sklearn_latencies.append(
            (time.perf_counter() - start) * 1000
        )

        start = time.perf_counter()

        session.run(
            None,
            {input_name: dense_sample},
        )

        onnx_latencies.append(
            (time.perf_counter() - start) * 1000
        )

    sklearn_metrics = calculate_metrics(
        sklearn_latencies
    )

    onnx_metrics = calculate_metrics(
        onnx_latencies
    )

    improvement = {
        "mean_latency_reduction_percent": float(
            (
                sklearn_metrics["mean_ms"]
                - onnx_metrics["mean_ms"]
            )
            / sklearn_metrics["mean_ms"]
            * 100
        ),
        "throughput_improvement_percent": float(
            (
                onnx_metrics[
                    "throughput_requests_per_second"
                ]
                - sklearn_metrics[
                    "throughput_requests_per_second"
                ]
            )
            / sklearn_metrics[
                "throughput_requests_per_second"
            ]
            * 100
        ),
    }

    results = {
        "benchmark_type": "classifier_only",
        "samples": N_SAMPLES,
        "sklearn": sklearn_metrics,
        "onnx_runtime": onnx_metrics,
        "improvement": improvement,
    }

    RESULT_PATH.write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )

    print("\nCLASSIFIER-ONLY RESULTS")
    print("=" * 60)

    print("\nScikit-learn Logistic Regression")

    for metric, value in sklearn_metrics.items():
        print(f"{metric}: {value:.4f}")

    print("\nONNX Runtime Logistic Regression")

    for metric, value in onnx_metrics.items():
        print(f"{metric}: {value:.4f}")

    print("\nImprovement")
    print(
        "Mean latency reduction: "
        f"{improvement['mean_latency_reduction_percent']:.2f}%"
    )
    print(
        "Throughput improvement: "
        f"{improvement['throughput_improvement_percent']:.2f}%"
    )

    print(
        f"\nBenchmark saved to: {RESULT_PATH}"
    )


if __name__ == "__main__":
    main()