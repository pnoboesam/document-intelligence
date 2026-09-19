from fastapi.testclient import TestClient

from src.api.main import app


client = TestClient(app)

IMAGE_PATH = "data/raw/receipt.png"


def test_vision_extraction():
    with open(IMAGE_PATH, "rb") as image:
        response = client.post(
            "/api/v1/extraction/",
            files={
                "image": ("receipt.png", image, "image/png")
            },
            data={"method": "vision"},
        )

    assert response.status_code == 200

    data = response.json()

    assert "company" in data
    assert "address" in data
    assert "date" in data
    assert "total" in data


def test_ocr_extraction():
    with open(IMAGE_PATH, "rb") as image:
        response = client.post(
            "/api/v1/extraction/",
            files={
                "image": ("receipt.png", image, "image/png")
            },
            data={"method": "ocr"},
        )

    assert response.status_code == 200

    data = response.json()

    assert "company" in data
    assert "address" in data
    assert "date" in data
    assert "total" in data


def test_unsupported_file_type():
    response = client.post(
        "/api/v1/extraction/",
        files={
            "image": (
                "test.txt",
                b"this is not an image",
                "text/plain",
            )
        },
        data={"method": "vision"},
    )

    assert response.status_code == 400


def test_missing_file():
    response = client.post(
        "/api/v1/extraction/",
        data={"method": "vision"},
    )

    assert response.status_code == 422


def test_invalid_method():
    with open(IMAGE_PATH, "rb") as image:
        response = client.post(
            "/api/v1/extraction/",
            files={
                "image": ("receipt.png", image, "image/png")
            },
            data={"method": "invalid"},
        )

    assert response.status_code == 422