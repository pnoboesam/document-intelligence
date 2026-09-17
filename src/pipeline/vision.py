from dataclasses import dataclass

from src.extraction.vision_model import VisionExtractor
from src.schemas.receipt import Receipt


@dataclass
class VisionPipeline:

    extractor: VisionExtractor

    def process(
        self,
        image_path: str,
    ) -> Receipt:

        receipt = self.extractor.extract(
            image_path
        )

        return receipt