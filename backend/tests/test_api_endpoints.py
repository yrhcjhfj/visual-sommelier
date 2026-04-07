"""Tests for FastAPI endpoints."""

import base64

import pytest
from fastapi.testclient import TestClient

from app.api import routes
from app.main import app
from app.models.device import DeviceAnalysisResult, DeviceContext
from app.models.explanation import Explanation, Step


@pytest.fixture
def client(monkeypatch):
    """Create a test client with mocked startup side effects."""

    monkeypatch.setattr("app.main.gpu_manager.check_cuda_availability", lambda: False)
    monkeypatch.setattr("app.main.gpu_manager.cleanup", lambda: None)
    monkeypatch.setattr("app.main.AdapterFactory.cleanup_all", lambda: None)

    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def device_context_payload():
    """Create a JSON-serializable device context payload."""

    return {
        "device_type": "washing_machine",
        "brand": "TestBrand",
        "model": "Model123",
        "detected_controls": [],
        "safety_warnings": [],
    }


def test_analyze_endpoint_returns_analysis(client, monkeypatch):
    """Analyze endpoint returns service result."""

    expected = DeviceAnalysisResult(
        device_type="washing_machine",
        confidence=0.92,
        brand="TestBrand",
        model="Model123",
        suggested_categories=[],
        detected_controls=[],
    )

    monkeypatch.setattr(routes.image_analysis_service, "analyze_device", lambda image: expected)

    response = client.post(
        "/api/analyze",
        files={"file": ("device.png", b"fake-image-bytes", "image/png")},
    )

    assert response.status_code == 200
    assert response.json()["device_type"] == "washing_machine"
    assert response.json()["confidence"] == 0.92


def test_analyze_endpoint_rejects_non_image_upload(client):
    """Analyze endpoint validates uploaded content type."""

    response = client.post(
        "/api/analyze",
        files={"file": ("device.txt", b"not-an-image", "text/plain")},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Uploaded file must be an image"


def test_explain_endpoint_returns_explanation(client, monkeypatch, device_context_payload):
    """Explain endpoint returns generated explanation."""

    expected = Explanation(
        text="Explanation text\n\nDisclaimer text",
        steps=None,
        warnings=["warning"],
        confidence=0.8,
        sources=["LLaVA Vision-Language Model"],
    )

    def mock_generate_explanation(image, question, device_context, language):
        assert image == b"test-image"
        assert question == "What does this do?"
        assert device_context == DeviceContext(**device_context_payload)
        assert language == "ru"
        return expected

    monkeypatch.setattr(routes.explanation_service, "generate_explanation", mock_generate_explanation)

    response = client.post(
        "/api/explain",
        json={
            "question": "What does this do?",
            "device_context": device_context_payload,
            "language": "ru",
            "image_base64": base64.b64encode(b"test-image").decode("ascii"),
        },
    )

    assert response.status_code == 200
    assert response.json()["text"].startswith("Explanation text")


def test_explain_endpoint_rejects_invalid_base64(client, device_context_payload):
    """Explain endpoint validates base64 payloads."""

    response = client.post(
        "/api/explain",
        json={
            "question": "What does this do?",
            "device_context": device_context_payload,
            "image_base64": "%%%invalid%%%",
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Invalid image_base64 payload"


def test_instructions_endpoint_returns_steps(client, monkeypatch, device_context_payload):
    """Instructions endpoint returns generated steps."""

    expected_steps = [
        Step(number=1, description="Open the door", warning=None),
        Step(number=2, description="Press start", warning="Be careful"),
    ]

    def mock_generate_instructions(task, device_context, language, image):
        assert task == "Start the wash cycle"
        assert device_context == DeviceContext(**device_context_payload)
        assert language == "en"
        assert image is None
        return expected_steps

    monkeypatch.setattr(routes.explanation_service, "generate_instructions", mock_generate_instructions)

    response = client.post(
        "/api/instructions",
        json={
            "task": "Start the wash cycle",
            "device_context": device_context_payload,
        },
    )

    assert response.status_code == 200
    assert len(response.json()["steps"]) == 2
    assert response.json()["steps"][0]["number"] == 1


def test_clarify_endpoint_returns_text(client, monkeypatch, device_context_payload):
    """Clarify endpoint returns generated clarification text."""

    step_payload = {
        "number": 1,
        "description": "Press the start button",
        "warning": None,
        "highlighted_area": None,
        "completed": False,
    }

    def mock_clarify_step(step, question, device_context, language, image):
        assert step.number == 1
        assert question == "Which one is start?"
        assert device_context == DeviceContext(**device_context_payload)
        assert language == "en"
        assert image is None
        return "Use the large button on the right."

    monkeypatch.setattr(routes.explanation_service, "clarify_step", mock_clarify_step)

    response = client.post(
        "/api/clarify",
        json={
            "step": step_payload,
            "question": "Which one is start?",
            "device_context": device_context_payload,
        },
    )

    assert response.status_code == 200
    assert response.json() == {"text": "Use the large button on the right."}
