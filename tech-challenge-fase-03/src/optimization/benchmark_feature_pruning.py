import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline

TRAIN_PATH = Path("data/raw/train.parquet")
TEST_PATH = Path("data/raw/test.parquet")
RESULT_PATH = Path("models/feature_pruning_benchmark.json")

TEXT_COLUMN = "medical_abstract"
TARGET_COLUMN = "condition_label"
N_BENCHMARK_SAMPLES = 1000
N_WARMUP_SAMPLES = 50

CONFIGURATIONS = {
    "baseline": None,
    "max_features_20000": 20_000,
    "max_features_10000": 10_000,
    "max_features_5000": 5_000,
}


def build_pipeline(max_features: int | None) -> Pipeline:
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    stop_words="english",
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.95,
                    sublinear_tf=True,
                    max_features=max_features,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )


def calculate_latency_metrics(
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


def benchmark_pipeline(
    pipeline: Pipeline,
    texts: list[str],
) -> dict[str, float]:
    warmup_texts = texts[:N_WARMUP_SAMPLES]
    benchmark_texts = texts[
        N_WARMUP_SAMPLES:
        N_WARMUP_SAMPLES + N_BENCHMARK_SAMPLES
    ]

    for text in warmup_texts:
        pipeline.predict([text])

    latencies = []

    for text in benchmark_texts:
        start = time.perf_counter()

        pipeline.predict([text])

        elapsed_ms = (
            time.perf_counter() - start
        ) * 1000

        latencies.append(elapsed_ms)

    return calculate_latency_metrics(latencies)


def main() -> None:
    print("Loading datasets...")

    train_df = pd.read_parquet(TRAIN_PATH)
    test_df = pd.read_parquet(TEST_PATH)

    x_train = train_df[TEXT_COLUMN]
    y_train = train_df[TARGET_COLUMN]

    x_test = test_df[TEXT_COLUMN]
    y_test = test_df[TARGET_COLUMN]

    benchmark_texts = (
        x_test
        .head(
            N_WARMUP_SAMPLES
            + N_BENCHMARK_SAMPLES
        )
        .tolist()
    )

    results = {}

    for name, max_features in CONFIGURATIONS.items():
        print("\n" + "=" * 60)
        print(f"Configuration: {name}")
        print(f"max_features: {max_features}")

        pipeline = build_pipeline(max_features)

        print("Training model...")
        pipeline.fit(x_train, y_train)

        predictions = pipeline.predict(x_test)

        accuracy = accuracy_score(
            y_test,
            predictions,
        )

        macro_f1 = f1_score(
            y_test,
            predictions,
            average="macro",
        )

        vocabulary_size = len(
            pipeline
            .named_steps["tfidf"]
            .vocabulary_
        )

        print("Benchmarking latency...")

        latency_metrics = benchmark_pipeline(
            pipeline,
            benchmark_texts,
        )

        results[name] = {
            "max_features": max_features,
            "vocabulary_size": vocabulary_size,
            "accuracy": float(accuracy),
            "macro_f1": float(macro_f1),
            **latency_metrics,
        }

        print(f"Vocabulary size: {vocabulary_size}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Macro F1: {macro_f1:.4f}")

        for metric, value in latency_metrics.items():
            print(f"{metric}: {value:.4f}")

    baseline = results["baseline"]

    for name, metrics in results.items():
        if name == "baseline":
            continue

        metrics["mean_latency_reduction_percent"] = float(
            (
                baseline["mean_ms"]
                - metrics["mean_ms"]
            )
            / baseline["mean_ms"]
            * 100
        )

        metrics["p95_latency_reduction_percent"] = float(
            (
                baseline["p95_ms"]
                - metrics["p95_ms"]
            )
            / baseline["p95_ms"]
            * 100
        )

        metrics["accuracy_difference_points"] = float(
            (
                metrics["accuracy"]
                - baseline["accuracy"]
            )
            * 100
        )

        metrics["macro_f1_difference"] = float(
            metrics["macro_f1"]
            - baseline["macro_f1"]
        )

    RESULT_PATH.write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    for name, metrics in results.items():
        print(f"\n{name}")
        print(
            f"Vocabulary: {metrics['vocabulary_size']}"
        )
        print(
            f"Accuracy: {metrics['accuracy']:.4f}"
        )
        print(
            f"Macro F1: {metrics['macro_f1']:.4f}"
        )
        print(
            f"Mean latency: {metrics['mean_ms']:.4f} ms"
        )
        print(
            f"P95: {metrics['p95_ms']:.4f} ms"
        )
        print(
            "Throughput: "
            f"{metrics['throughput_requests_per_second']:.2f} req/s"
        )

        if name != "baseline":
            print(
                "Mean latency reduction: "
                f"{metrics['mean_latency_reduction_percent']:.2f}%"
            )
            print(
                "P95 reduction: "
                f"{metrics['p95_latency_reduction_percent']:.2f}%"
            )
            print(
                "Accuracy difference: "
                f"{metrics['accuracy_difference_points']:.2f} p.p."
            )

    print(
        f"\nResults saved to: {RESULT_PATH}"
    )


if __name__ == "__main__":
    main()