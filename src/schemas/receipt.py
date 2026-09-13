from pydantic import BaseModel

class Receipt(BaseModel):
    company: str
    address: str
    date: str
    total: float

