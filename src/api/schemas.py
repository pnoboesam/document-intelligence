from pydantic import BaseModel
from enum import Enum

class Method(str, Enum):
    ocr = "ocr"
    vision = "vision"

class ExtractionResponse(BaseModel):
    company: str
    address: str
    date: str
    total: float