import json
from pathlib import Path

from .metrics import FIELDS


PREDICTIONS_FILE = Path(
    "evaluation_results/predictions.jsonl"
)


def main():
    records = []

    with PREDICTIONS_FILE.open(
        "r",
        encoding="utf-8",
    ) as file:

        for line in file:

            if not line.strip():
                continue

            record = json.loads(line)

            if record["status"] == "success":
                records.append(record)

    print("=" * 70)
    print("ERROR ANALYSIS")
    print("=" * 70)

    print(f"\nSuccessful predictions: {len(records)}")

    # --------------------------------------------------
    # Field statistics
    # --------------------------------------------------

    print("\nFIELD ERROR RATES")

    for field in FIELDS:

        exact_correct = sum(
            result["metrics"][field]["exact_match"]
            for result in records
        )

        total = len(records)

        exact_match_rate = exact_correct / total
        error_rate = 1 - exact_match_rate

        average_f1 = sum(
            result["metrics"][field]["f1"]
            for result in records
        ) / total

        print(
            f"{field:10s} "
            f"F1={average_f1:.3f} "
            f"Exact={exact_match_rate:.3f} "
            f"Errors={total - exact_correct}"
        )

    # --------------------------------------------------
    # Worst examples by field
    # --------------------------------------------------

    for field in FIELDS:

        print("\n" + "=" * 70)
        print(f"WORST {field.upper()} PREDICTIONS")
        print("=" * 70)

        sorted_records = sorted(
            records,
            key=lambda record: record["metrics"][field]["f1"],
        )

        for record in sorted_records[:10]:

            metrics = record["metrics"][field]

            print(f"\nReceipt: {record['key']}")
            print(f"F1: {metrics['f1']:.3f}")
            print(
                f"Prediction: "
                f"{metrics['prediction']}"
            )
            print(
                f"Ground truth: "
                f"{metrics['ground_truth']}"
            )

    # --------------------------------------------------
    # Low-quality overall predictions
    # --------------------------------------------------

    print("\n" + "=" * 70)
    print("LOWEST OVERALL RECEIPTS")
    print("=" * 70)

    receipt_scores = []

    for record in records:

        scores = [
            record["metrics"][field]["f1"]
            for field in FIELDS
        ]

        average = sum(scores) / len(scores)

        receipt_scores.append(
            (average, record)
        )

    receipt_scores.sort(
        key=lambda item: item[0]
    )

    for average, record in receipt_scores[:20]:

        print(
            f"\nReceipt: {record['key']} "
            f"Overall F1={average:.3f}"
        )

        for field in FIELDS:

            metrics = record["metrics"][field]

            print(
                f"  {field:10s}: "
                f"{metrics['f1']:.3f} | "
                f"Pred={metrics['prediction']} | "
                f"GT={metrics['ground_truth']}"
            )


if __name__ == "__main__":
    main()