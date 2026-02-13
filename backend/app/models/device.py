"""Device-related data models."""

from pydantic import BaseModel, Field, field_validator
from typing import List, Optional


class BoundingBox(BaseModel):
    """Represents a bounding box for detected objects or controls."""
    
    x: float = Field(..., ge=0.0, description="X coordinate (top-left)")
    y: float = Field(..., ge=0.0, description="Y coordinate (top-left)")
    width: float = Field(..., gt=0.0, description="Width of the bounding box")
    height: float = Field(..., gt=0.0, description="Height of the bounding box")
    
    @field_validator('x', 'y', 'width', 'height')
    @classmethod
    def validate_positive(cls, v: float) -> float:
        """Ensure coordinates and dimensions are valid."""
        if v < 0:
            raise ValueError("Coordinates and dimensions must be non-negative")
        return v


class Control(BaseModel):
    """Represents a detected control element on a device."""
    
    id: str = Field(..., min_length=1, description="Unique identifier for the control")
    type: str = Field(..., min_length=1, description="Type of control (button, knob, switch, lever, etc.)")
    label: Optional[str] = Field(None, description="Text label on the control, if any")
    bounding_box: BoundingBox = Field(..., description="Location of the control on the image")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Detection confidence score (0.0 - 1.0)")
    
    @field_validator('confidence')
    @classmethod
    def validate_confidence(cls, v: float) -> float:
        """Ensure confidence is between 0 and 1."""
        if not 0.0 <= v <= 1.0:
            raise ValueError("Confidence must be between 0.0 and 1.0")
        return v


class DeviceAnalysisResult(BaseModel):
    """Result of device image analysis."""
    
    device_type: str = Field(..., min_length=1, description="Type of device detected")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Detection confidence (0.0 - 1.0)")
    brand: Optional[str] = Field(None, description="Brand of the device, if detected")
    model: Optional[str] = Field(None, description="Model of the device, if detected")
    suggested_categories: List[str] = Field(
        default_factory=list,
        description="Suggested categories when confidence is low"
    )
    detected_controls: List[Control] = Field(
        default_factory=list,
        description="List of detected control elements"
    )
    
    @field_validator('confidence')
    @classmethod
    def validate_confidence(cls, v: float) -> float:
        """Ensure confidence is between 0 and 1."""
        if not 0.0 <= v <= 1.0:
            raise ValueError("Confidence must be between 0.0 and 1.0")
        return v
    
    @field_validator('device_type')
    @classmethod
    def validate_device_type(cls, v: str) -> str:
        """Ensure device type is not empty."""
        if not v or not v.strip():
            raise ValueError("Device type cannot be empty")
        return v.strip()


class DeviceContext(BaseModel):
    """Context information about a device for a session."""
    
    device_type: str = Field(..., min_length=1, description="Type of device")
    brand: Optional[str] = Field(None, description="Brand of the device")
    model: Optional[str] = Field(None, description="Model of the device")
    detected_controls: List[Control] = Field(
        default_factory=list,
        description="List of detected controls"
    )
    safety_warnings: List[str] = Field(
        default_factory=list,
        description="Safety warnings for this device type"
    )
