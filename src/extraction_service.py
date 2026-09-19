from src.pipeline.document import DocumentPipeline
from src.pipeline.vision import VisionPipeline

class ExtractionService:
    def __init__(self, document_pipeline: DocumentPipeline, vision_pipeline: VisionPipeline):
        self.document_pipeline = document_pipeline
        self.vision_pipeline = vision_pipeline

    def extract(self, image_path: str, method: str):
        if method == "ocr":
            return self.document_pipeline.process(image_path)

        if method == "vision":
            return self.vision_pipeline.process(image_path)

        raise ValueError(f"Unsupported extraction method: {method}")
    