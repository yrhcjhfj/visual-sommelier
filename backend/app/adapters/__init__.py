"""Adapters for CV and LLM providers."""

from .base import (
    CVProviderAdapter,
    LLMProviderAdapter,
    Label,
    TextAnnotation,
    DetectedObject
)
from .yolo_adapter import YOLOAdapter
from .easyocr_adapter import EasyOCRAdapter
from .llava_adapter import LLaVAAdapter
from .factory import (
    AdapterFactory,
    get_yolo_adapter,
    get_ocr_adapter,
    get_llava_adapter
)

__all__ = [
    # Base classes
    "CVProviderAdapter",
    "LLMProviderAdapter",
    "Label",
    "TextAnnotation",
    "DetectedObject",
    # Implementations
    "YOLOAdapter",
    "EasyOCRAdapter",
    "LLaVAAdapter",
    # Factory
    "AdapterFactory",
    "get_yolo_adapter",
    "get_ocr_adapter",
    "get_llava_adapter",
]
