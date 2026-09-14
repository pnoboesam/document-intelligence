from PIL import Image
import pytesseract
from pytesseract import Output

from .base import OCRResult, OCRWord, OCREngine


PATH=r"C:\Program Files\Tesseract-OCR\tesseract.exe"

class TesseractOCR(OCREngine):
    def __init__(self, tesseract_cmd: str=PATH):
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

    def extract(self, image_path: str) -> OCRResult:
        image = Image.open(image_path)

        data = pytesseract.image_to_data(
            image,
            output_type=Output.DICT
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

        return OCRResult(words=words)