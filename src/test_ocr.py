from ocr.tesseract import TesseractOCR


RECEIPTS = [
    ("X51005577191", "evaluation_results/failed_images/X51005577191.png"),
    ("X51005663309", "evaluation_results/failed_images/X51005663309.png"),
    ("X51005719863", "evaluation_results/failed_images/X51005719863.png"),
    ("X51005749904", "evaluation_results/failed_images/X51005749904.png"),
    ("X51006619328", "evaluation_results/failed_images/X51006619328.png"),
    ("X51006619703", "evaluation_results/failed_images/X51006619703.png"),
]


def main():
    ocr = TesseractOCR()

    for key, image_path in RECEIPTS:

        print("\n" + "=" * 70)
        print(f"RECEIPT: {key}")
        print("=" * 70)

        result = ocr.extract(image_path)

        print(f"Method: {result.ocr_method}")
        print(f"Words: {result.word_count}")
        print(
            f"Average confidence: "
            f"{result.average_confidence:.3f}"
        )

        print("\nOCR TEXT:")
        print(result.text)


if __name__ == "__main__":
    main()