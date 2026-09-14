from openai import OpenAI
from dotenv import load_dotenv

import os

from schemas.receipt import Receipt


load_dotenv()


class LLMExtractor:

    def __init__(self, model: str):
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ["OPENROUTER_API_KEY"],
            timeout=60.0,
        )

        self.model = model

    def extract(self, ocr_text: str) -> Receipt:

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": """
                You are a document information extraction system.

Your task is to extract receipt information from OCR text.

Rules:
- Extract only information supported by the OCR text.
- Do not invent missing information.
- If a field cannot be determined, return an empty string.
- Extract the merchant/company name, not a product or brand name.
- Return the date in DD/MM/YYYY format only.
- Do not include the time in the date field.
- The total must be the receipt's final total, not subtotal, tax,
  item price, cash tendered, or change.
                """
                },
                {
                    "role": "user",
                    "content": ocr_text,
                },
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "receipt",
                    "strict": True,
                    "schema": Receipt.model_json_schema(),
                },
            },
        )

        content = response.choices[0].message.content

        return Receipt.model_validate_json(content)