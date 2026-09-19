from fastapi import FastAPI

from src.api.routes.health import router as health_router
from src.api.routes.extraction import router as extraction_router

app = FastAPI()

app.include_router(health_router)
app.include_router(extraction_router)
