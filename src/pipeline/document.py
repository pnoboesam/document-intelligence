from dataclasses import dataclass

from ocr.tesseract import TesseractOCR
from extraction.llm import LLMExtractor
from schemas.receipt import Receipt


@dataclass
class DocumentPipeline:
    ocr: TesseractOCR
    extractor: LLMExtractor

    def process(self, image_path: str) -> Receipt:
        # Step 1: OCR
        ocr_result = self.ocr.extract(image_path)

        # Step 2: Extract structured information
        receipt = self.extractor.extract(
            ocr_text=ocr_result.text
        )

        # Step 3: Return validated structured data
        return receipt