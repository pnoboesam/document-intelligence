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

    @property
    def text(self) -> str:
        return " ".join(word.text for word in self.words)


class OCREngine(ABC):

    @abstractmethod
    def extract(self, image_path: str) -> OCRResult:
        pass