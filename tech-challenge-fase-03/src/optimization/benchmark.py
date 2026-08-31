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
RESULT_PATH = Path("models/benchmark.json")

TEXT_COLUMN = "medical_abstract"

N_SAMPLES = 300
WARMUP_SAMPLES = 20


def calculate_metrics(latencies_ms: list[float]) -> dict[str, float]:
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


def benchmark_sklearn(
    pipeline,
    texts: list[str],
) -> dict[str, float]:
    latencies = []

    for text in texts:
        start = time.perf_counter()

        pipeline.predict([text])

        elapsed = (
            time.perf_counter() - start
        ) * 1000

        latencies.append(elapsed)

    return calculate_metrics(latencies)


def benchmark_onnx(
    tfidf,
    session: ort.InferenceSession,
    texts: list[str],
) -> dict[str, float]:
    latencies = []

    input_name = session.get_inputs()[0].name

    for text in texts:
        start = time.perf_counter()

        features = tfidf.transform([text])

        onnx_input = (
            features
            .astype(np.float32)
            .toarray()
        )

        session.run(
            None,
            {input_name: onnx_input},
        )

        elapsed = (
            time.perf_counter() - start
        ) * 1000

        latencies.append(elapsed)

    return calculate_metrics(latencies)


def main() -> None:
    print("Loading models and test data...")

    pipeline = joblib.load(MODEL_PATH)

    tfidf = pipeline.named_steps["tfidf"]

    session = ort.InferenceSession(
        str(ONNX_PATH),
        providers=["CPUExecutionProvider"],
    )

    test_df = pd.read_parquet(TEST_PATH)

    texts = (
        test_df[TEXT_COLUMN]
        .head(N_SAMPLES + WARMUP_SAMPLES)
        .tolist()
    )

    warmup_texts = texts[:WARMUP_SAMPLES]
    benchmark_texts = texts[WARMUP_SAMPLES:]

    print(
        f"Warm-up requests: {len(warmup_texts)}"
    )
    print(
        f"Benchmark requests: {len(benchmark_texts)}"
    )

    print("\nWarming up scikit-learn...")

    for text in warmup_texts:
        pipeline.predict([text])

    print("Warming up ONNX Runtime...")

    input_name = session.get_inputs()[0].name

    for text in warmup_texts:
        features = tfidf.transform([text])

        onnx_input = (
            features
            .astype(np.float32)
            .toarray()
        )

        session.run(
            None,
            {input_name: onnx_input},
        )

    print("\nBenchmarking scikit-learn...")

    sklearn_metrics = benchmark_sklearn(
        pipeline,
        benchmark_texts,
    )

    print("Benchmarking ONNX Runtime...")

    onnx_metrics = benchmark_onnx(
        tfidf,
        session,
        benchmark_texts,
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
        "samples": len(benchmark_texts),
        "sklearn": sklearn_metrics,
        "onnx_runtime": onnx_metrics,
        "improvement": improvement,
    }

    RESULT_PATH.write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )

    print("\nRESULTS")
    print("=" * 60)

    print("\nScikit-learn")
    for metric, value in sklearn_metrics.items():
        print(f"{metric}: {value:.4f}")

    print("\nONNX Runtime")
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