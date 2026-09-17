from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class OCRWord:
    text: str
    bbox: tuple[int, int, int, int]
    confidence: float


@dataclass
class OCRResult:
    words: list[OCRWord]
    ocr_method: str = "original"

    @property
    def text(self) -> str:
        return " ".join(word.text for word in self.words)

    @property
    def word_count(self) -> int:
        return len(self.words)

    @property
    def average_confidence(self) -> float:
        if not self.words:
            return 0.0

        return sum(word.confidence for word in self.words) / len(self.words)


class OCREngine(ABC):

    @abstractmethod
    def extract(self, image_path: str) -> OCRResult:
        pass