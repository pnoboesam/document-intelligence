from extraction.vision_model import VisionExtractor
from pipeline.vision import VisionPipeline


IMAGE_PATH = "data/raw/receipt_low.png"


def main():

    extractor = VisionExtractor(
        model="openai/gpt-5.6-luna"
    )

    pipeline = VisionPipeline(
        extractor=extractor
    )

    receipt = pipeline.process(
        IMAGE_PATH
    )

    print("\n--- VISION EXTRACTION ---")
    print(receipt)

    print("\n--- JSON ---")
    print(
        receipt.model_dump_json(
            indent=2
        )
    )


if __name__ == "__main__":
    main()