from pathlib import Path

from datasets import load_dataset


RECEIPT_KEYS = [
    "X51005577191",
    "X51005663309",
    "X51005719863",
    "X51005749904",
    "X51006619328",
    "X51006619703",
]


OUTPUT_DIR = Path(
    "evaluation_results/failed_images"
)


def main():
    print("Loading SROIE dataset...")

    dataset = load_dataset(
        "jsdnrs/ICDAR2019-SROIE"
    )

    test_dataset = dataset["test"]

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    saved = 0

    for example in test_dataset:

        if example["key"] not in RECEIPT_KEYS:
            continue

        image = example["image"]

        output_path = (
            OUTPUT_DIR
            / f"{example['key']}.png"
        )

        image.save(output_path)

        print(f"Saved: {output_path}")

        saved += 1

    print(
        f"\nSaved {saved} failed receipt images."
    )


if __name__ == "__main__":
    main()