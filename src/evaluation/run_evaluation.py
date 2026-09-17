import json
import tempfile
import time
from pathlib import Path

from datasets import load_dataset

from src.ocr.tesseract import TesseractOCR
from src.extraction.llm import LLMExtractor
from src.pipeline.document import DocumentPipeline

from src.evaluation.metrics import (
    FIELDS,
    evaluate_receipt,
)

SPLIT = "test"
MODEL = "openai/gpt-5.6-luna"
DATASET_NAME = "jsdnrs/ICDAR2019-SROIE"

NUM_SAMPLES = 1

RESULTS_DIR = Path("evaluation_results")

PREDICTIONS_FILE = RESULTS_DIR / "predictions.jsonl"
REPORT_FILE = RESULTS_DIR / "evaluation_report.json"


def load_completed_keys() -> set[str]:
    """
    Load receipt IDs that have already been processed.

    This allows the evaluation to resume if it is interrupted.
    """

    if not PREDICTIONS_FILE.exists():
        return set()

    completed = set()

    with PREDICTIONS_FILE.open(
        "r",
        encoding="utf-8",
    ) as file:

        for line in file:
            if not line.strip():
                continue

            record = json.loads(line)

            completed.add(record["key"])

    return completed


def save_result(record: dict) -> None:
    """
    Append one evaluation result to the JSONL file immediately.
    """

    with PREDICTIONS_FILE.open(
        "a",
        encoding="utf-8",
    ) as file:

        file.write(
            json.dumps(
                record,
                ensure_ascii=False,
            )
            + "\n"
        )


def calculate_summary(results: list[dict]) -> dict:
    """
    Calculate aggregate metrics across successful predictions.
    """

    summary = {}

    for field in FIELDS:

        f1_scores = [
            result["metrics"][field]["f1"]
            for result in results
        ]

        exact_scores = [
            result["metrics"][field]["exact_match"]
            for result in results
        ]

        if not f1_scores:
            continue

        summary[field] = {
            "f1": sum(f1_scores) / len(f1_scores),
            "exact_match": (
                sum(exact_scores)
                / len(exact_scores)
            ),
        }

    # Overall macro average across fields
    if summary:

        overall_f1 = sum(
            metrics["f1"]
            for metrics in summary.values()
        ) / len(summary)

        overall_exact = sum(
            metrics["exact_match"]
            for metrics in summary.values()
        ) / len(summary)

        summary["overall"] = {
            "f1": overall_f1,
            "exact_match": overall_exact,
        }

    return summary


def main():

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Loading SROIE dataset...")

    dataset = load_dataset(
        DATASET_NAME,
        split=SPLIT,
    )

    total_samples = min(
        NUM_SAMPLES,
        len(dataset),
    )

    print(
        f"Test samples available: "
        f"{len(dataset)}"
    )

    print(
        f"Evaluation target: "
        f"{total_samples}"
    )

    completed_keys = load_completed_keys()

    print(
        f"Already completed: "
        f"{len(completed_keys)}"
    )

    print(
        f"Remaining: "
        f"{total_samples - len(completed_keys)}"
    )

    if len(completed_keys) >= total_samples:
        print(
            "\nAll requested samples have "
            "already been processed."
        )
        return

    ocr = TesseractOCR()

    extractor = LLMExtractor(
        model=MODEL
    )

    pipeline = DocumentPipeline(
        ocr=ocr,
        extractor=extractor,
    )

    successful = 0
    failed = 0

    for index in range(total_samples):

        example = dataset[index]

        key = example["key"]

        if key in completed_keys:
            continue

        print(
            f"\n[{index + 1}/{total_samples}] "
            f"{key}"
        )

        image = example["image"]
        ground_truth = example["entities"]

        start_time = time.perf_counter()

        try:

            with tempfile.TemporaryDirectory() as temp_dir:

                image_path = (
                    Path(temp_dir)
                    / "receipt.png"
                )

                image.save(image_path)

                pipeline_result = pipeline.process(
                    str(image_path)
                )

                prediction = pipeline_result.receipt
                ocr_result = pipeline_result.ocr_result

            elapsed = (
                time.perf_counter()
                - start_time
            )

            metrics = evaluate_receipt(
                prediction,
                ground_truth,
            )

            record = {
                "key": key,
                "status": "success",
                "latency_seconds": elapsed,

                "ocr": {
                    "method": ocr_result.ocr_method,
                    "word_count": ocr_result.word_count,
                    "average_confidence": (
                        ocr_result.average_confidence
                    ),
                },

                "prediction": prediction.model_dump(),
                "ground_truth": ground_truth,
                "metrics": metrics,
            }

            save_result(record)

            successful += 1

            print(
                f"Status: SUCCESS | "
                f"Time: {elapsed:.2f}s"
            )

            for field in FIELDS:

                field_metrics = metrics[field]

                print(
                    f"  {field}: "
                    f"F1={field_metrics['f1']:.3f}, "
                    f"Exact="
                    f"{field_metrics['exact_match']}"
                )

        except Exception as exc:

            elapsed = (
                time.perf_counter()
                - start_time
            )

            record = {
                "key": key,
                "status": "error",
                "latency_seconds": elapsed,
                "error": str(exc),
            }

            save_result(record)

            failed += 1

            print(
                f"Status: ERROR | "
                f"Time: {elapsed:.2f}s"
            )

            print(
                f"Error: {exc}"
            )

    # --------------------------------------------------
    # Load every saved result
    # --------------------------------------------------

    all_results = []

    with PREDICTIONS_FILE.open(
        "r",
        encoding="utf-8",
    ) as file:

        for line in file:

            if not line.strip():
                continue

            record = json.loads(line)

            if record["status"] == "success":
                all_results.append(record)

    # --------------------------------------------------
    # Calculate aggregate metrics
    # --------------------------------------------------

    summary = calculate_summary(
        all_results
    )

    latencies = [
        result["latency_seconds"]
        for result in all_results
    ]

    report = {
        "dataset": "jsdnrs/ICDAR2019-SROIE",
        "split": "test",
        "model": MODEL,
        "requested_samples": total_samples,
        "successful_samples": len(all_results),
        "failed_samples": (
            total_samples - len(all_results)
        ),
        "metrics": summary,
        "latency": {
            "average_seconds": (
                sum(latencies) / len(latencies)
                if latencies
                else None
            ),
            "min_seconds": (
                min(latencies)
                if latencies
                else None
            ),
            "max_seconds": (
                max(latencies)
                if latencies
                else None
            ),
        },
    }

    with REPORT_FILE.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            report,
            file,
            indent=2,
        )

    # --------------------------------------------------
    # Final summary
    # --------------------------------------------------

    print("\n")
    print("=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)

    print(
        f"Successful: "
        f"{len(all_results)}"
    )

    print(
        f"Failed: "
        f"{total_samples - len(all_results)}"
    )

    print("\nMETRICS")

    for field, metrics in summary.items():

        print(
            f"{field:10s} "
            f"F1={metrics['f1']:.3f} "
            f"Exact={metrics['exact_match']:.3f}"
        )

    print("\nLATENCY")

    if latencies:

        print(
            f"Average: "
            f"{sum(latencies) / len(latencies):.2f}s"
        )

        print(
            f"Min: "
            f"{min(latencies):.2f}s"
        )

        print(
            f"Max: "
            f"{max(latencies):.2f}s"
        )

    print("\nSaved:")
    print(PREDICTIONS_FILE)
    print(REPORT_FILE)


if __name__ == "__main__":
    main()