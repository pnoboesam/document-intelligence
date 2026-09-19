import os
import tempfile
from fastapi import APIRouter, UploadFile, File, Form, HTTPException

from src.extraction_service import ExtractionService
from src.pipeline.document import DocumentPipeline
from src.pipeline.vision import VisionPipeline

from src.ocr.tesseract import TesseractOCR
from src.extraction.llm import LLMExtractor
from src.extraction.vision_model import VisionExtractor

from src.api.schemas import ExtractionResponse, Method


router = APIRouter(prefix="/api/v1/extraction", tags=["extraction"])

document_pipeline = DocumentPipeline(
    ocr=TesseractOCR(),
    extractor=LLMExtractor(model="openai/gpt-5.6-luna")
)

vision_pipeline = VisionPipeline(
    extractor=VisionExtractor()
)

extraction = ExtractionService(
    document_pipeline,
    vision_pipeline
)


@router.post("/", response_model=ExtractionResponse)
async def extract(
    image: UploadFile = File(...),
    method: Method = Form(...)
):
    if not image.filename:
        raise HTTPException(status_code=400, detail="No file provided.")

    allowed_types = {"image/jpeg", "image/png", "image/webp"}

    if image.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type. Use JPEG, PNG, or WebP."
        )

    
    suffix = os.path.splitext(image.filename)[1]

    try:
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix
        ) as temp_file:
            temp_file.write(await image.read())
            image_path = temp_file.name

        try:
            return extraction.extract(image_path, method)
        
        except Exception:
            raise HTTPException(
                status_code=500,
                detail="Document extraction failed."
            )
        
        finally:
            os.remove(image_path)

    except HTTPException:
        raise

    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Failed to process uploaded file."
        )

