from ocr.tesseract import TesseractOCR
from extraction.llm import LLMExtractor
from pipeline.document import DocumentPipeline

IMAGE_PATH = "data/raw/receipt.png"
MODEL = "openai/gpt-5.6-luna"


def main():

    # OCR
    ocr = TesseractOCR()

    extractor = LLMExtractor(
        model=MODEL
    )
    
    pipeline = DocumentPipeline(
        ocr=ocr,
        extractor=extractor
    )
    
    result = pipeline.process(IMAGE_PATH)

    print("\n--- OCR ---")
    print(f"Method: {result.ocr_result.ocr_method}")
    print(f"Words: {result.ocr_result.word_count}")
    print(f"Average confidence: {result.ocr_result.average_confidence:.3f}")

    print("\n--- EXTRACTED RECEIPT ---")
    print(result.receipt)

    print("\n--- JSON ---")
    print(result.receipt.model_dump_json(indent=2))


if __name__ == "__main__":
    main()