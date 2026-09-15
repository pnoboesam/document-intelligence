import tempfile
from pathlib import Path

import cv2
import numpy as np
from datasets import load_dataset
from PIL import Image

from src.ocr.tesseract import TesseractOCR


RECEIPT_KEYS = [
    "X51005577191",
    "X51005663309",
    "X51005719863",
    "X51005749904",
    "X51006619328",
    "X51006619703",
]


def preprocess_grayscale(image: Image.Image) -> Image.Image:
    return image.convert("L")


def preprocess_contrast(image: Image.Image) -> Image.Image:
    gray = np.array(image.convert("L"))

    # Improve local contrast using CLAHE.
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8),
    )

    enhanced = clahe.apply(gray)

    return Image.fromarray(enhanced)


def preprocess_threshold(image: Image.Image) -> Image.Image:
    gray = np.array(image.convert("L"))

    # Local adaptive thresholding.
    thresholded = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        11,
    )

    return Image.fromarray(thresholded)


def run_ocr(
    image: Image.Image,
    psm: int,
    temp_dir: str,
) -> tuple[int, float, str]:

    image_path = (
        Path(temp_dir)
        / f"receipt_psm_{psm}.png"
    )

    image.save(image_path)

    ocr = TesseractOCR()

    # Configure Tesseract's page segmentation mode.
    import pytesseract

    result = pytesseract.image_to_data(
        image,
        config=f"--psm {psm}",
        output_type=pytesseract.Output.DICT,
    )

    words = []

    for i, text in enumerate(result["text"]):

        text = text.strip()

        if not text:
            continue

        confidence = float(
            result["conf"][i]
        )

        if confidence < 0:
            continue

        words.append(
            (
                text,
                confidence / 100,
            )
        )

    text = " ".join(
        word[0]
        for word in words
    )

    average_confidence = (
        sum(word[1] for word in words)
        / len(words)
        if words
        else 0.0
    )

    return (
        len(words),
        average_confidence,
        text,
    )


def main():

    print("Loading SROIE dataset...")

    dataset = load_dataset(
        "jsdnrs/ICDAR2019-SROIE"
    )

    test_dataset = dataset["test"]

    examples = {
        example["key"]: example
        for example in test_dataset
        if example["key"] in RECEIPT_KEYS
    }

    for key in RECEIPT_KEYS:

        print("\n" + "=" * 80)
        print(f"RECEIPT: {key}")
        print("=" * 80)

        example = examples[key]
        image = example["image"]

        with tempfile.TemporaryDirectory() as temp_dir:

            configurations = {
                "original_psm3": (
                    image,
                    3,
                ),
                "original_psm6": (
                    image,
                    6,
                ),
                "original_psm11": (
                    image,
                    11,
                ),
                "grayscale_psm6": (
                    preprocess_grayscale(image),
                    6,
                ),
                "contrast_psm6": (
                    preprocess_contrast(image),
                    6,
                ),
                "threshold_psm6": (
                    preprocess_threshold(image),
                    6,
                ),
            }

            for name, (
                processed_image,
                psm,
            ) in configurations.items():

                words, confidence, text = run_ocr(
                    processed_image,
                    psm,
                    temp_dir,
                )

                print(
                    f"\n--- {name} ---"
                )

                print(
                    f"Words: {words}"
                )

                print(
                    f"Confidence: "
                    f"{confidence:.3f}"
                )

                print(
                    f"Text:\n{text[:1000]}"
                )


if __name__ == "__main__":
    main()