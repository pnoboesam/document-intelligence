from PIL import Image

import os
import cv2
import numpy as np
import pytesseract
from pytesseract import Output

from .base import OCRResult, OCRWord, OCREngine


PATH = os.getenv("TESSERACT_CMD", "tesseract")

class TesseractOCR(OCREngine):

    def __init__(
        self,
        tesseract_cmd: str = PATH,
        confidence_threshold: float = 0.40,
        word_threshold: int = 20,
    ):
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

        self.confidence_threshold = confidence_threshold
        self.word_threshold = word_threshold

    def extract(self, image_path: str) -> OCRResult:
        image = Image.open(image_path)

        # --------------------------------------------------
        # 1. Original OCR
        # --------------------------------------------------
        original_result = self._run_ocr(
            image=image,
            method="original",
        )

        # --------------------------------------------------
        # 2. Check OCR quality
        # --------------------------------------------------
        if not self._needs_fallback(original_result):
            return original_result

        # --------------------------------------------------
        # 3. Contrast-enhanced OCR fallback
        # --------------------------------------------------
        enhanced_image = self._enhance_contrast(image)

        fallback_result = self._run_ocr(
            image=enhanced_image,
            method="contrast_fallback",
        )

        if self._is_better_result(
            fallback_result,
            original_result,
        ):
            return fallback_result

        return original_result

    def _run_ocr(
        self,
        image: Image.Image,
        method: str,
    ) -> OCRResult:

        data = pytesseract.image_to_data(
            image,
            output_type=Output.DICT,
            config="--psm 6",
        )

        words = []

        for i, text in enumerate(data["text"]):

            text = text.strip()

            if not text:
                continue

            confidence = float(data["conf"][i])

            if confidence < 0:
                continue

            x = int(data["left"][i])
            y = int(data["top"][i])
            width = int(data["width"][i])
            height = int(data["height"][i])

            bbox = (
                x,
                y,
                x + width,
                y + height,
            )

            words.append(
                OCRWord(
                    text=text,
                    bbox=bbox,
                    confidence=confidence / 100,
                )
            )

        return OCRResult(
            words=words,
            ocr_method=method,
        )

    def _needs_fallback(self, result: OCRResult) -> bool:

        if result.word_count <= self.word_threshold:
            return True

        if result.average_confidence <= self.confidence_threshold:
            return True

        return False

    def _enhance_contrast(
        self,
        image: Image.Image,
    ) -> Image.Image:

        grayscale = np.array(
            image.convert("L")
        )

        clahe = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(8, 8),
        )

        enhanced = clahe.apply(grayscale)

        return Image.fromarray(enhanced)

    def _is_better_result(
        self,
        candidate: OCRResult,
        current: OCRResult,
    ) -> bool:

        return (
            self._quality_score(candidate)
            > self._quality_score(current)
        )

    def _quality_score(self, result: OCRResult) -> float:

        confidence_score = result.average_confidence

        word_score = min(
            result.word_count / 100,
            1.0,
        )

        return (
            0.7 * confidence_score
            + 0.3 * word_score
        )