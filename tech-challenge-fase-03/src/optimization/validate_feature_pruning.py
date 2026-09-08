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
RESULT_PATH = Path("models/feature_pruning_validation.json")

N_WARMUP = 50
N_SAMPLES = 1000
N_ROUNDS = 5


def build_pipeline(max_features: int | None) -> Pipeline:
    return Pipeline(
        [
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


def benchmark(
    pipeline: Pipeline,
    texts: list[str],
) -> dict[str, float]:
    for text in texts[:N_WARMUP]:
        pipeline.predict([text])

    latencies = []

    benchmark_texts = texts[
        N_WARMUP:N_WARMUP + N_SAMPLES
    ]

    for text in benchmark_texts:
        start = time.perf_counter_ns()

        pipeline.predict([text])

        elapsed_ms = (
            time.perf_counter_ns() - start
        ) / 1_000_000

        latencies.append(elapsed_ms)

    values = np.array(latencies)

    return {
        "mean_ms": float(values.mean()),
        "p50_ms": float(np.percentile(values, 50)),
        "p95_ms": float(np.percentile(values, 95)),
        "p99_ms": float(np.percentile(values, 99)),
        "throughput": float(1000 / values.mean()),
    }


def summarize_rounds(
    rounds: list[dict[str, float]],
) -> dict[str, float]:
    return {
        "median_mean_ms": float(
            np.median([r["mean_ms"] for r in rounds])
        ),
        "median_p50_ms": float(
            np.median([r["p50_ms"] for r in rounds])
        ),
        "median_p95_ms": float(
            np.median([r["p95_ms"] for r in rounds])
        ),
        "median_p99_ms": float(
            np.median([r["p99_ms"] for r in rounds])
        ),
        "median_throughput": float(
            np.median([r["throughput"] for r in rounds])
        ),
    }


def main() -> None:
    print("Loading data...")

    train_df = pd.read_parquet(TRAIN_PATH)
    test_df = pd.read_parquet(TEST_PATH)

    x_train = train_df["medical_abstract"]
    y_train = train_df["condition_label"]

    x_test = test_df["medical_abstract"]
    y_test = test_df["condition_label"]

    texts = x_test.head(
        N_WARMUP + N_SAMPLES
    ).tolist()

    print("Training baseline...")
    baseline = build_pipeline(None)
    baseline.fit(x_train, y_train)

    print("Training optimized model...")
    optimized = build_pipeline(10_000)
    optimized.fit(x_train, y_train)

    baseline_pred = baseline.predict(x_test)
    optimized_pred = optimized.predict(x_test)

    quality = {
        "baseline": {
            "accuracy": float(
                accuracy_score(y_test, baseline_pred)
            ),
            "macro_f1": float(
                f1_score(
                    y_test,
                    baseline_pred,
                    average="macro",
                )
            ),
            "vocabulary_size": len(
                baseline.named_steps["tfidf"].vocabulary_
            ),
        },
        "optimized": {
            "accuracy": float(
                accuracy_score(y_test, optimized_pred)
            ),
            "macro_f1": float(
                f1_score(
                    y_test,
                    optimized_pred,
                    average="macro",
                )
            ),
            "vocabulary_size": len(
                optimized.named_steps["tfidf"].vocabulary_
            ),
        },
    }

    baseline_rounds = []
    optimized_rounds = []

    for round_number in range(1, N_ROUNDS + 1):
        print(f"\nRound {round_number}/{N_ROUNDS}")

        # Alterna a ordem para reduzir viés de execução.
        if round_number % 2 == 1:
            baseline_result = benchmark(
                baseline,
                texts,
            )
            optimized_result = benchmark(
                optimized,
                texts,
            )
        else:
            optimized_result = benchmark(
                optimized,
                texts,
            )
            baseline_result = benchmark(
                baseline,
                texts,
            )

        baseline_rounds.append(baseline_result)
        optimized_rounds.append(optimized_result)

        print(
            "Baseline mean: "
            f"{baseline_result['mean_ms']:.4f} ms | "
            "P95: "
            f"{baseline_result['p95_ms']:.4f} ms"
        )

        print(
            "Optimized mean: "
            f"{optimized_result['mean_ms']:.4f} ms | "
            "P95: "
            f"{optimized_result['p95_ms']:.4f} ms"
        )

    baseline_summary = summarize_rounds(
        baseline_rounds
    )
    optimized_summary = summarize_rounds(
        optimized_rounds
    )

    mean_reduction = (
        (
            baseline_summary["median_mean_ms"]
            - optimized_summary["median_mean_ms"]
        )
        / baseline_summary["median_mean_ms"]
        * 100
    )

    p95_reduction = (
        (
            baseline_summary["median_p95_ms"]
            - optimized_summary["median_p95_ms"]
        )
        / baseline_summary["median_p95_ms"]
        * 100
    )

    throughput_improvement = (
        (
            optimized_summary["median_throughput"]
            - baseline_summary["median_throughput"]
        )
        / baseline_summary["median_throughput"]
        * 100
    )

    results = {
        "quality": quality,
        "baseline_rounds": baseline_rounds,
        "optimized_rounds": optimized_rounds,
        "baseline_summary": baseline_summary,
        "optimized_summary": optimized_summary,
        "improvement": {
            "mean_latency_reduction_percent": float(
                mean_reduction
            ),
            "p95_latency_reduction_percent": float(
                p95_reduction
            ),
            "throughput_improvement_percent": float(
                throughput_improvement
            ),
        },
    }

    RESULT_PATH.write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)

    print(
        "Baseline vocabulary:",
        quality["baseline"]["vocabulary_size"],
    )
    print(
        "Optimized vocabulary:",
        quality["optimized"]["vocabulary_size"],
    )

    print(
        "\nBaseline accuracy:",
        f"{quality['baseline']['accuracy']:.4f}",
    )
    print(
        "Optimized accuracy:",
        f"{quality['optimized']['accuracy']:.4f}",
    )

    print(
        "\nBaseline Macro F1:",
        f"{quality['baseline']['macro_f1']:.4f}",
    )
    print(
        "Optimized Macro F1:",
        f"{quality['optimized']['macro_f1']:.4f}",
    )

    print(
        "\nBaseline median mean latency:",
        f"{baseline_summary['median_mean_ms']:.4f} ms",
    )
    print(
        "Optimized median mean latency:",
        f"{optimized_summary['median_mean_ms']:.4f} ms",
    )

    print(
        "\nBaseline median P95:",
        f"{baseline_summary['median_p95_ms']:.4f} ms",
    )
    print(
        "Optimized median P95:",
        f"{optimized_summary['median_p95_ms']:.4f} ms",
    )

    print(
        "\nMean latency reduction:",
        f"{mean_reduction:.2f}%",
    )
    print(
        "P95 latency reduction:",
        f"{p95_reduction:.2f}%",
    )
    print(
        "Throughput improvement:",
        f"{throughput_improvement:.2f}%",
    )

    print(
        "\nResults saved to:",
        RESULT_PATH,
    )


if __name__ == "__main__":
    main()