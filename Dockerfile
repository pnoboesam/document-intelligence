FROM python:3.14-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libgl1 \
    libxcb1 \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml uv.lock ./

RUN pip install uv && uv sync --frozen

COPY . .

EXPOSE 8000

CMD ["sh", "-c", "uv run uvicorn src.api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]