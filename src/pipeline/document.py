from dataclasses import dataclass

from src.ocr.tesseract import TesseractOCR
from src.extraction.llm import LLMExtractor
from src.schemas.receipt import Receipt
from src.ocr.base import OCRResult


@dataclass
class PipelineResult:
    receipt: Receipt
    ocr_result: OCRResult


@dataclass
class DocumentPipeline:
    ocr: TesseractOCR
    extractor: LLMExtractor

    def process(self, image_path: str) -> PipelineResult:

        # OCR
        ocr_result = self.ocr.extract(image_path)

        # Extract structured information from OCR text
        receipt = self.extractor.extract(
            ocr_text=ocr_result.text
        )

        return receipt