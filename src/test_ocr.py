from ocr.tesseract import TesseractOCR


def main():
    ocr = TesseractOCR(
        tesseract_cmd=r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    )

    result = ocr.extract("data/raw/receipt.png")

    print("\n--- OCR TEXT ---")
    print(result.text)

    print("\n--- OCR WORDS ---")

    for word in result.words:
        print(
            f"text={word.text!r}, "
            f"bbox={word.bbox}, "
            f"confidence={word.confidence:.2f}"
        )


if __name__ == "__main__":
    main()