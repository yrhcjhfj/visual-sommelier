"""Session and message-related data models."""

from pydantic import BaseModel, Field, field_validator
from typing import List, Optional
from datetime import datetime
from enum import Enum


class MessageRole(str, Enum):
    """Role of a message in a conversation."""
    
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class Message(BaseModel):
    """Represents a message in a conversation."""
    
    role: MessageRole = Field(..., description="Role of the message sender")
    content: str = Field(..., min_length=1, description="Content of the message")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="When the message was created")
    image_ref: Optional[str] = Field(None, description="Reference to an image, if attached")
    
    @field_validator('content')
    @classmethod
    def validate_content(cls, v: str) -> str:
        """Ensure content is not empty."""
        if not v or not v.strip():
            raise ValueError("Message content cannot be empty")
        return v


class Session(BaseModel):
    """Represents a user session with a device."""
    
    id: str = Field(..., min_length=1, description="Unique session identifier")
    user_id: str = Field(..., min_length=1, description="User identifier")
    device_type: str = Field(..., min_length=1, description="Type of device in this session")
    device_image_url: str = Field(..., min_length=1, description="URL or path to the device image")
    messages: List[Message] = Field(default_factory=list, description="Conversation messages")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Session creation time")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last update time")
    device_context: Optional['DeviceContext'] = Field(None, description="Device context information")
    
    @field_validator('id', 'user_id', 'device_type', 'device_image_url')
    @classmethod
    def validate_not_empty(cls, v: str) -> str:
        """Ensure required string fields are not empty."""
        if not v or not v.strip():
            raise ValueError("Field cannot be empty")
        return v.strip()


# Import DeviceContext for type hint resolution
from backend.app.models.device import DeviceContext
Session.model_rebuild()
