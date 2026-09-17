import base64
import mimetypes
import os

from dotenv import load_dotenv
from openai import OpenAI

from src.schemas.receipt import Receipt


load_dotenv()


class VisionExtractor:

    def __init__(
        self,
        model: str = "openai/gpt-5.6-luna",
    ):
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ["OPENROUTER_API_KEY"],
            timeout=60.0,
        )

        self.model = model

    def extract(self, image_path: str) -> Receipt:

        image_data = self._encode_image(
            image_path
        )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": """
You are a document information extraction system.

Your task is to extract receipt information
from the provided receipt image.

Rules:
- Extract only information visible in the receipt.
- Do not invent missing information.
- If a field cannot be determined, return an empty string.
- Extract the merchant/company name, not a product or brand name.
- Return the date in DD/MM/YYYY format only.
- Do not include the time in the date field.
- The total must be the receipt's final total,
  not subtotal, tax, item price, cash tendered,
  or change.
""",
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Extract the company, address, "
                                "date, and final total from "
                                "this receipt."
                            ),
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_data,
                            },
                        },
                    ],
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

    def _encode_image(
        self,
        image_path: str,
    ) -> str:

        mime_type, _ = mimetypes.guess_type(
            image_path
        )

        if mime_type is None:
            mime_type = "image/png"

        with open(image_path, "rb") as image_file:
            encoded_image = base64.b64encode(
                image_file.read()
            ).decode("utf-8")

        return (
            f"data:{mime_type};base64,"
            f"{encoded_image}"
        )