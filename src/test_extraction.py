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
    
    receipt = pipeline.process(IMAGE_PATH)

    # 3. Structured result
    print("\n--- EXTRACTED RECEIPT ---")
    print(receipt)

    print("\n--- JSON ---")
    print(receipt.model_dump_json(indent=2))


if __name__ == "__main__":
    main()