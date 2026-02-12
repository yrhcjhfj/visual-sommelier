"""Base adapter interfaces for CV and LLM providers."""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class Label:
    """Detected label/class."""
    name: str
    confidence: float


@dataclass
class TextAnnotation:
    """Detected text with location."""
    text: str
    confidence: float
    bounding_box: tuple[float, float, float, float]  # (x, y, width, height)


@dataclass
class DetectedObject:
    """Detected object with bounding box."""
    class_name: str
    confidence: float
    bounding_box: tuple[float, float, float, float]  # (x, y, width, height)


class CVProviderAdapter(ABC):
    """Abstract interface for Computer Vision providers."""
    
    @abstractmethod
    def detect_labels(self, image_bytes: bytes) -> List[Label]:
        """Detect objects and labels in the image.
        
        Args:
            image_bytes: Image data as bytes
            
        Returns:
            List of detected labels with confidence scores
        """
        pass
    
    @abstractmethod
    def detect_text(self, image_bytes: bytes) -> List[TextAnnotation]:
        """Recognize text in the image.
        
        Args:
            image_bytes: Image data as bytes
            
        Returns:
            List of detected text annotations with locations
        """
        pass
    
    @abstractmethod
    def detect_objects(self, image_bytes: bytes) -> List[DetectedObject]:
        """Detect objects with bounding boxes.
        
        Args:
            image_bytes: Image data as bytes
            
        Returns:
            List of detected objects with bounding boxes
        """
        pass
    
    @abstractmethod
    def load_model(self) -> None:
        """Load the model into memory/GPU."""
        pass
    
    @abstractmethod
    def unload_model(self) -> None:
        """Unload the model from memory/GPU."""
        pass
    
    @abstractmethod
    def is_loaded(self) -> bool:
        """Check if model is currently loaded."""
        pass


class LLMProviderAdapter(ABC):
    """Abstract interface for LLM providers."""
    
    @abstractmethod
    def generate_completion(
        self,
        prompt: str,
        image: Optional[bytes] = None,
        language: str = "en",
        max_tokens: int = 512
    ) -> str:
        """Generate text completion.
        
        Args:
            prompt: Text prompt
            image: Optional image data for vision-language models
            language: Target language code
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated text response
        """
        pass
    
    @abstractmethod
    def generate_structured_output(
        self,
        prompt: str,
        schema: Dict[str, Any],
        language: str = "en"
    ) -> Dict[str, Any]:
        """Generate structured output matching a schema.
        
        Args:
            prompt: Text prompt
            schema: JSON schema for output structure
            language: Target language code
            
        Returns:
            Structured output as dictionary
        """
        pass
    
    @abstractmethod
    def load_model(self) -> None:
        """Load the model into memory/GPU."""
        pass
    
    @abstractmethod
    def unload_model(self) -> None:
        """Unload the model from memory/GPU."""
        pass
    
    @abstractmethod
    def is_loaded(self) -> bool:
        """Check if model is currently loaded."""
        pass
