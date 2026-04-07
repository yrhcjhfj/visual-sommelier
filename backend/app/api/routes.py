"""API routes for image analysis and explanation flows."""

import base64
import binascii
import logging
from typing import List, Optional

from fastapi import APIRouter, File, HTTPException, UploadFile, status
from pydantic import BaseModel, Field

from ..config import settings
from ..models.device import DeviceAnalysisResult, DeviceContext
from ..models.explanation import Explanation, Step
from ..services.explanation_service import ExplanationService
from ..services.image_analysis_service import ImageAnalysisService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["api"])

image_analysis_service = ImageAnalysisService()
explanation_service = ExplanationService()


class ExplainRequest(BaseModel):
    """Request payload for explanation generation."""

    question: str = Field(..., min_length=1)
    device_context: DeviceContext
    language: str = Field(default="en", min_length=2, max_length=5)
    image_base64: Optional[str] = None


class InstructionsRequest(BaseModel):
    """Request payload for instruction generation."""

    task: str = Field(..., min_length=1)
    device_context: DeviceContext
    language: str = Field(default="en", min_length=2, max_length=5)
    image_base64: Optional[str] = None


class ClarifyRequest(BaseModel):
    """Request payload for step clarification."""

    step: Step
    question: str = Field(..., min_length=1)
    device_context: DeviceContext
    language: str = Field(default="en", min_length=2, max_length=5)
    image_base64: Optional[str] = None


class StepsResponse(BaseModel):
    """Response payload for generated instructions."""

    steps: List[Step]


class ClarifyResponse(BaseModel):
    """Response payload for step clarification."""

    text: str


def _validate_image_size(image_bytes: bytes) -> None:
    """Validate uploaded image size against configured limit."""

    max_size_bytes = settings.max_image_size_mb * 1024 * 1024
    if len(image_bytes) > max_size_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Image exceeds max size of {settings.max_image_size_mb} MB",
        )


def _decode_image_base64(image_base64: Optional[str]) -> Optional[bytes]:
    """Decode optional base64 image payload."""

    if image_base64 is None:
        return None

    payload = image_base64.strip()
    if not payload:
        return None

    if "," in payload and payload.startswith("data:"):
        payload = payload.split(",", maxsplit=1)[1]

    try:
        image_bytes = base64.b64decode(payload, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid image_base64 payload",
        ) from exc

    _validate_image_size(image_bytes)
    return image_bytes


@router.post("/analyze", response_model=DeviceAnalysisResult)
async def analyze_image(file: UploadFile = File(...)) -> DeviceAnalysisResult:
    """Analyze an uploaded device image."""

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file must be an image",
        )

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded image is empty",
        )

    _validate_image_size(image_bytes)

    try:
        return image_analysis_service.analyze_device(image_bytes)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Failed to analyze image")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to analyze image",
        ) from exc


@router.post("/explain", response_model=Explanation)
async def explain_device(request: ExplainRequest) -> Explanation:
    """Generate an explanation for a device question."""

    image_bytes = _decode_image_base64(request.image_base64)

    try:
        return explanation_service.generate_explanation(
            image=image_bytes or b"",
            question=request.question,
            device_context=request.device_context,
            language=request.language,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Failed to generate explanation")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate explanation",
        ) from exc


@router.post("/instructions", response_model=StepsResponse)
async def generate_instructions(request: InstructionsRequest) -> StepsResponse:
    """Generate step-by-step instructions for a task."""

    image_bytes = _decode_image_base64(request.image_base64)

    try:
        steps = explanation_service.generate_instructions(
            task=request.task,
            device_context=request.device_context,
            language=request.language,
            image=image_bytes,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Failed to generate instructions")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate instructions",
        ) from exc

    return StepsResponse(steps=steps)


@router.post("/clarify", response_model=ClarifyResponse)
async def clarify_step(request: ClarifyRequest) -> ClarifyResponse:
    """Clarify a specific instruction step."""

    image_bytes = _decode_image_base64(request.image_base64)

    try:
        text = explanation_service.clarify_step(
            step=request.step,
            question=request.question,
            device_context=request.device_context,
            language=request.language,
            image=image_bytes,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Failed to clarify step")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to clarify step",
        ) from exc

    return ClarifyResponse(text=text)
