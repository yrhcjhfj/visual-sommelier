"""Explanation and instruction-related data models."""

from pydantic import BaseModel, Field, field_validator
from typing import List, Optional
from backend.app.models.device import BoundingBox


class Step(BaseModel):
    """Represents a single step in instructions."""
    
    number: int = Field(..., ge=1, description="Step number (1-indexed)")
    description: str = Field(..., min_length=1, description="Description of the step")
    warning: Optional[str] = Field(None, description="Safety warning for this step, if any")
    highlighted_area: Optional[BoundingBox] = Field(
        None,
        description="Area to highlight on the image for this step"
    )
    completed: bool = Field(default=False, description="Whether the step has been completed")
    
    @field_validator('number')
    @classmethod
    def validate_step_number(cls, v: int) -> int:
        """Ensure step number is positive."""
        if v < 1:
            raise ValueError("Step number must be at least 1")
        return v
    
    @field_validator('description')
    @classmethod
    def validate_description(cls, v: str) -> str:
        """Ensure description is not empty."""
        if not v or not v.strip():
            raise ValueError("Step description cannot be empty")
        return v


class Explanation(BaseModel):
    """Represents an explanation or set of instructions."""
    
    text: str = Field(..., min_length=1, description="Main explanation text")
    steps: Optional[List[Step]] = Field(None, description="Step-by-step instructions, if applicable")
    warnings: List[str] = Field(default_factory=list, description="Safety warnings")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in the explanation (0.0 - 1.0)")
    sources: List[str] = Field(default_factory=list, description="Sources of information")
    
    @field_validator('text')
    @classmethod
    def validate_text(cls, v: str) -> str:
        """Ensure explanation text is not empty."""
        if not v or not v.strip():
            raise ValueError("Explanation text cannot be empty")
        return v
    
    @field_validator('confidence')
    @classmethod
    def validate_confidence(cls, v: float) -> float:
        """Ensure confidence is between 0 and 1."""
        if not 0.0 <= v <= 1.0:
            raise ValueError("Confidence must be between 0.0 and 1.0")
        return v
    
    @field_validator('steps')
    @classmethod
    def validate_steps_order(cls, v: Optional[List[Step]]) -> Optional[List[Step]]:
        """Ensure steps are in sequential order if provided."""
        if v is not None and len(v) > 0:
            for i, step in enumerate(v, start=1):
                if step.number != i:
                    raise ValueError(f"Steps must be in sequential order. Expected step {i}, got {step.number}")
        return v
