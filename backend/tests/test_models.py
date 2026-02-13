"""Tests for Pydantic data models."""

import pytest
import json
from datetime import datetime
from pydantic import ValidationError
from backend.app.models import (
    BoundingBox,
    Control,
    DeviceAnalysisResult,
    DeviceContext,
    Message,
    MessageRole,
    Session,
    Explanation,
    Step,
)


class TestBoundingBox:
    """Tests for BoundingBox model."""
    
    def test_valid_bounding_box(self):
        """Test creating a valid bounding box."""
        bbox = BoundingBox(x=10.0, y=20.0, width=100.0, height=50.0)
        assert bbox.x == 10.0
        assert bbox.y == 20.0
        assert bbox.width == 100.0
        assert bbox.height == 50.0
    
    def test_negative_coordinates_rejected(self):
        """Test that negative coordinates are rejected."""
        with pytest.raises(ValidationError):
            BoundingBox(x=-10.0, y=20.0, width=100.0, height=50.0)
    
    def test_zero_width_rejected(self):
        """Test that zero or negative width is rejected."""
        with pytest.raises(ValidationError):
            BoundingBox(x=10.0, y=20.0, width=0.0, height=50.0)
    
    def test_negative_height_rejected(self):
        """Test that negative height is rejected."""
        with pytest.raises(ValidationError):
            BoundingBox(x=10.0, y=20.0, width=100.0, height=-50.0)
    
    def test_serialization(self):
        """Test serialization to dict."""
        bbox = BoundingBox(x=10.0, y=20.0, width=100.0, height=50.0)
        data = bbox.model_dump()
        assert data == {"x": 10.0, "y": 20.0, "width": 100.0, "height": 50.0}
    
    def test_deserialization(self):
        """Test deserialization from dict."""
        data = {"x": 10.0, "y": 20.0, "width": 100.0, "height": 50.0}
        bbox = BoundingBox(**data)
        assert bbox.x == 10.0
        assert bbox.y == 20.0
    
    def test_json_serialization(self):
        """Test JSON serialization."""
        bbox = BoundingBox(x=10.0, y=20.0, width=100.0, height=50.0)
        json_str = bbox.model_dump_json()
        assert isinstance(json_str, str)
        data = json.loads(json_str)
        assert data["x"] == 10.0
    
    def test_json_deserialization(self):
        """Test JSON deserialization."""
        json_str = '{"x": 10.0, "y": 20.0, "width": 100.0, "height": 50.0}'
        bbox = BoundingBox.model_validate_json(json_str)
        assert bbox.x == 10.0
        assert bbox.width == 100.0


class TestControl:
    """Tests for Control model."""
    
    def test_valid_control(self):
        """Test creating a valid control."""
        bbox = BoundingBox(x=10.0, y=20.0, width=100.0, height=50.0)
        control = Control(
            id="btn1",
            type="button",
            label="Power",
            bounding_box=bbox,
            confidence=0.95
        )
        assert control.id == "btn1"
        assert control.type == "button"
        assert control.label == "Power"
        assert control.confidence == 0.95
    
    def test_confidence_out_of_range(self):
        """Test that confidence outside [0, 1] is rejected."""
        bbox = BoundingBox(x=10.0, y=20.0, width=100.0, height=50.0)
        with pytest.raises(ValidationError):
            Control(
                id="btn1",
                type="button",
                bounding_box=bbox,
                confidence=1.5
            )
    
    def test_confidence_negative(self):
        """Test that negative confidence is rejected."""
        bbox = BoundingBox(x=10.0, y=20.0, width=100.0, height=50.0)
        with pytest.raises(ValidationError):
            Control(
                id="btn1",
                type="button",
                bounding_box=bbox,
                confidence=-0.1
            )
    
    def test_empty_id_rejected(self):
        """Test that empty id is rejected."""
        bbox = BoundingBox(x=10.0, y=20.0, width=100.0, height=50.0)
        with pytest.raises(ValidationError):
            Control(
                id="",
                type="button",
                bounding_box=bbox,
                confidence=0.9
            )
    
    def test_optional_label(self):
        """Test that label is optional."""
        bbox = BoundingBox(x=10.0, y=20.0, width=100.0, height=50.0)
        control = Control(
            id="btn1",
            type="button",
            bounding_box=bbox,
            confidence=0.9
        )
        assert control.label is None
    
    def test_serialization(self):
        """Test serialization to dict."""
        bbox = BoundingBox(x=10.0, y=20.0, width=100.0, height=50.0)
        control = Control(
            id="btn1",
            type="button",
            label="Power",
            bounding_box=bbox,
            confidence=0.95
        )
        data = control.model_dump()
        assert data["id"] == "btn1"
        assert data["type"] == "button"
        assert data["confidence"] == 0.95
        assert "bounding_box" in data
    
    def test_deserialization(self):
        """Test deserialization from dict."""
        data = {
            "id": "btn1",
            "type": "button",
            "label": "Power",
            "bounding_box": {"x": 10.0, "y": 20.0, "width": 100.0, "height": 50.0},
            "confidence": 0.95
        }
        control = Control(**data)
        assert control.id == "btn1"
        assert control.bounding_box.x == 10.0


class TestDeviceAnalysisResult:
    """Tests for DeviceAnalysisResult model."""
    
    def test_valid_device_analysis(self):
        """Test creating a valid device analysis result."""
        result = DeviceAnalysisResult(
            device_type="washing_machine",
            confidence=0.85,
            brand="Samsung",
            model="WF45R6100",
            suggested_categories=["appliance", "laundry"],
            detected_controls=[]
        )
        assert result.device_type == "washing_machine"
        assert result.confidence == 0.85
        assert result.brand == "Samsung"
    
    def test_empty_device_type_rejected(self):
        """Test that empty device type is rejected."""
        with pytest.raises(ValidationError):
            DeviceAnalysisResult(
                device_type="",
                confidence=0.85
            )
    
    def test_whitespace_device_type_rejected(self):
        """Test that whitespace-only device type is rejected."""
        with pytest.raises(ValidationError):
            DeviceAnalysisResult(
                device_type="   ",
                confidence=0.85
            )
    
    def test_confidence_validation(self):
        """Test confidence validation."""
        with pytest.raises(ValidationError):
            DeviceAnalysisResult(
                device_type="remote",
                confidence=1.5
            )
    
    def test_default_empty_lists(self):
        """Test that lists default to empty."""
        result = DeviceAnalysisResult(
            device_type="remote",
            confidence=0.9
        )
        assert result.suggested_categories == []
        assert result.detected_controls == []
    
    def test_serialization(self):
        """Test serialization to dict."""
        result = DeviceAnalysisResult(
            device_type="washing_machine",
            confidence=0.85,
            brand="Samsung"
        )
        data = result.model_dump()
        assert data["device_type"] == "washing_machine"
        assert data["confidence"] == 0.85
        assert data["brand"] == "Samsung"
    
    def test_deserialization(self):
        """Test deserialization from dict."""
        data = {
            "device_type": "washing_machine",
            "confidence": 0.85,
            "brand": "Samsung",
            "model": "WF45R6100",
            "suggested_categories": ["appliance"],
            "detected_controls": []
        }
        result = DeviceAnalysisResult(**data)
        assert result.device_type == "washing_machine"
        assert result.brand == "Samsung"


class TestDeviceContext:
    """Tests for DeviceContext model."""
    
    def test_valid_device_context(self):
        """Test creating a valid device context."""
        context = DeviceContext(
            device_type="washing_machine",
            brand="Samsung",
            model="WF45R6100",
            detected_controls=[],
            safety_warnings=["Unplug before maintenance"]
        )
        assert context.device_type == "washing_machine"
        assert context.brand == "Samsung"
        assert len(context.safety_warnings) == 1
    
    def test_default_empty_lists(self):
        """Test that lists default to empty."""
        context = DeviceContext(device_type="remote")
        assert context.detected_controls == []
        assert context.safety_warnings == []
    
    def test_serialization(self):
        """Test serialization to dict."""
        context = DeviceContext(
            device_type="washing_machine",
            brand="Samsung"
        )
        data = context.model_dump()
        assert data["device_type"] == "washing_machine"
        assert data["brand"] == "Samsung"
    
    def test_deserialization(self):
        """Test deserialization from dict."""
        data = {
            "device_type": "washing_machine",
            "brand": "Samsung",
            "model": "WF45R6100",
            "detected_controls": [],
            "safety_warnings": ["Be careful"]
        }
        context = DeviceContext(**data)
        assert context.device_type == "washing_machine"


class TestMessage:
    """Tests for Message model."""
    
    def test_valid_message(self):
        """Test creating a valid message."""
        msg = Message(
            role=MessageRole.USER,
            content="How do I use this button?"
        )
        assert msg.role == MessageRole.USER
        assert msg.content == "How do I use this button?"
        assert isinstance(msg.timestamp, datetime)
    
    def test_empty_content_rejected(self):
        """Test that empty content is rejected."""
        with pytest.raises(ValidationError):
            Message(role=MessageRole.USER, content="")
    
    def test_whitespace_content_rejected(self):
        """Test that whitespace-only content is rejected."""
        with pytest.raises(ValidationError):
            Message(role=MessageRole.USER, content="   ")
    
    def test_default_timestamp(self):
        """Test that timestamp is auto-generated."""
        msg = Message(role=MessageRole.USER, content="Test")
        assert isinstance(msg.timestamp, datetime)
    
    def test_optional_image_ref(self):
        """Test that image_ref is optional."""
        msg = Message(role=MessageRole.USER, content="Test")
        assert msg.image_ref is None
    
    def test_all_roles(self):
        """Test all message roles."""
        for role in [MessageRole.USER, MessageRole.ASSISTANT, MessageRole.SYSTEM]:
            msg = Message(role=role, content="Test")
            assert msg.role == role
    
    def test_serialization(self):
        """Test serialization to dict."""
        msg = Message(
            role=MessageRole.USER,
            content="How do I use this button?",
            image_ref="/images/device.jpg"
        )
        data = msg.model_dump()
        assert data["role"] == "user"
        assert data["content"] == "How do I use this button?"
        assert data["image_ref"] == "/images/device.jpg"
    
    def test_deserialization(self):
        """Test deserialization from dict."""
        data = {
            "role": "user",
            "content": "How do I use this button?",
            "timestamp": datetime.utcnow().isoformat(),
            "image_ref": None
        }
        msg = Message(**data)
        assert msg.role == MessageRole.USER
        assert msg.content == "How do I use this button?"


class TestSession:
    """Tests for Session model."""
    
    def test_valid_session(self):
        """Test creating a valid session."""
        session = Session(
            id="sess123",
            user_id="user456",
            device_type="remote_control",
            device_image_url="/images/remote.jpg"
        )
        assert session.id == "sess123"
        assert session.user_id == "user456"
        assert len(session.messages) == 0
        assert isinstance(session.created_at, datetime)
    
    def test_empty_id_rejected(self):
        """Test that empty id is rejected."""
        with pytest.raises(ValidationError):
            Session(
                id="",
                user_id="user456",
                device_type="remote",
                device_image_url="/images/remote.jpg"
            )
    
    def test_empty_user_id_rejected(self):
        """Test that empty user_id is rejected."""
        with pytest.raises(ValidationError):
            Session(
                id="sess123",
                user_id="",
                device_type="remote",
                device_image_url="/images/remote.jpg"
            )
    
    def test_default_timestamps(self):
        """Test that timestamps are auto-generated."""
        session = Session(
            id="sess123",
            user_id="user456",
            device_type="remote",
            device_image_url="/images/remote.jpg"
        )
        assert isinstance(session.created_at, datetime)
        assert isinstance(session.updated_at, datetime)
    
    def test_default_empty_messages(self):
        """Test that messages default to empty list."""
        session = Session(
            id="sess123",
            user_id="user456",
            device_type="remote",
            device_image_url="/images/remote.jpg"
        )
        assert session.messages == []
    
    def test_optional_device_context(self):
        """Test that device_context is optional."""
        session = Session(
            id="sess123",
            user_id="user456",
            device_type="remote",
            device_image_url="/images/remote.jpg"
        )
        assert session.device_context is None
    
    def test_serialization(self):
        """Test serialization to dict."""
        session = Session(
            id="sess123",
            user_id="user456",
            device_type="remote",
            device_image_url="/images/remote.jpg"
        )
        data = session.model_dump()
        assert data["id"] == "sess123"
        assert data["user_id"] == "user456"
        assert data["device_type"] == "remote"
    
    def test_deserialization(self):
        """Test deserialization from dict."""
        data = {
            "id": "sess123",
            "user_id": "user456",
            "device_type": "remote",
            "device_image_url": "/images/remote.jpg",
            "messages": [],
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat(),
            "device_context": None
        }
        session = Session(**data)
        assert session.id == "sess123"
        assert session.user_id == "user456"


class TestStep:
    """Tests for Step model."""
    
    def test_valid_step(self):
        """Test creating a valid step."""
        step = Step(
            number=1,
            description="Press the power button",
            warning="Ensure device is unplugged"
        )
        assert step.number == 1
        assert step.description == "Press the power button"
        assert step.completed is False
    
    def test_invalid_step_number(self):
        """Test that step number < 1 is rejected."""
        with pytest.raises(ValidationError):
            Step(number=0, description="Invalid step")
    
    def test_negative_step_number(self):
        """Test that negative step number is rejected."""
        with pytest.raises(ValidationError):
            Step(number=-1, description="Invalid step")
    
    def test_empty_description_rejected(self):
        """Test that empty description is rejected."""
        with pytest.raises(ValidationError):
            Step(number=1, description="")
    
    def test_default_completed_false(self):
        """Test that completed defaults to False."""
        step = Step(number=1, description="Test step")
        assert step.completed is False
    
    def test_optional_warning(self):
        """Test that warning is optional."""
        step = Step(number=1, description="Test step")
        assert step.warning is None
    
    def test_optional_highlighted_area(self):
        """Test that highlighted_area is optional."""
        step = Step(number=1, description="Test step")
        assert step.highlighted_area is None
    
    def test_with_highlighted_area(self):
        """Test step with highlighted area."""
        bbox = BoundingBox(x=10.0, y=20.0, width=100.0, height=50.0)
        step = Step(
            number=1,
            description="Press here",
            highlighted_area=bbox
        )
        assert step.highlighted_area.x == 10.0
    
    def test_serialization(self):
        """Test serialization to dict."""
        step = Step(
            number=1,
            description="Press the power button",
            warning="Be careful"
        )
        data = step.model_dump()
        assert data["number"] == 1
        assert data["description"] == "Press the power button"
        assert data["completed"] is False
    
    def test_deserialization(self):
        """Test deserialization from dict."""
        data = {
            "number": 1,
            "description": "Press the power button",
            "warning": "Be careful",
            "highlighted_area": None,
            "completed": False
        }
        step = Step(**data)
        assert step.number == 1
        assert step.description == "Press the power button"


class TestExplanation:
    """Tests for Explanation model."""
    
    def test_valid_explanation(self):
        """Test creating a valid explanation."""
        explanation = Explanation(
            text="This button turns on the device",
            confidence=0.9,
            warnings=["Be careful with electrical devices"]
        )
        assert explanation.text == "This button turns on the device"
        assert explanation.confidence == 0.9
        assert len(explanation.warnings) == 1
    
    def test_steps_sequential_order(self):
        """Test that steps must be in sequential order."""
        step1 = Step(number=1, description="First step")
        step2 = Step(number=2, description="Second step")
        
        explanation = Explanation(
            text="Instructions",
            confidence=0.9,
            steps=[step1, step2]
        )
        assert len(explanation.steps) == 2
    
    def test_steps_wrong_order_rejected(self):
        """Test that steps in wrong order are rejected."""
        step1 = Step(number=1, description="First step")
        step3 = Step(number=3, description="Third step")
        
        with pytest.raises(ValidationError):
            Explanation(
                text="Instructions",
                confidence=0.9,
                steps=[step1, step3]
            )
    
    def test_empty_text_rejected(self):
        """Test that empty text is rejected."""
        with pytest.raises(ValidationError):
            Explanation(text="", confidence=0.9)
    
    def test_confidence_validation(self):
        """Test confidence validation."""
        with pytest.raises(ValidationError):
            Explanation(text="Test", confidence=1.5)
    
    def test_default_empty_lists(self):
        """Test that lists default to empty."""
        explanation = Explanation(text="Test", confidence=0.9)
        assert explanation.warnings == []
        assert explanation.sources == []
    
    def test_optional_steps(self):
        """Test that steps are optional."""
        explanation = Explanation(text="Test", confidence=0.9)
        assert explanation.steps is None
    
    def test_serialization(self):
        """Test serialization to dict."""
        explanation = Explanation(
            text="This button turns on the device",
            confidence=0.9,
            warnings=["Be careful"]
        )
        data = explanation.model_dump()
        assert data["text"] == "This button turns on the device"
        assert data["confidence"] == 0.9
        assert len(data["warnings"]) == 1
    
    def test_deserialization(self):
        """Test deserialization from dict."""
        data = {
            "text": "This button turns on the device",
            "confidence": 0.9,
            "warnings": ["Be careful"],
            "steps": None,
            "sources": []
        }
        explanation = Explanation(**data)
        assert explanation.text == "This button turns on the device"
        assert explanation.confidence == 0.9
    
    def test_json_round_trip(self):
        """Test JSON serialization and deserialization round trip."""
        step1 = Step(number=1, description="First step")
        step2 = Step(number=2, description="Second step")
        
        original = Explanation(
            text="Instructions",
            confidence=0.9,
            steps=[step1, step2],
            warnings=["Be careful"],
            sources=["Manual"]
        )
        
        json_str = original.model_dump_json()
        restored = Explanation.model_validate_json(json_str)
        
        assert restored.text == original.text
        assert restored.confidence == original.confidence
        assert len(restored.steps) == 2
        assert restored.steps[0].number == 1
        assert restored.steps[1].number == 2

