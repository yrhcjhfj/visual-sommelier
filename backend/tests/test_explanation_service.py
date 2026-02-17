"""Tests for ExplanationService."""

import pytest
from unittest.mock import Mock, patch
from backend.app.services.explanation_service import ExplanationService
from backend.app.models.device import DeviceContext, Control, BoundingBox
from backend.app.models.explanation import Explanation, Step


@pytest.fixture
def explanation_service():
    """Create an ExplanationService instance."""
    return ExplanationService()


@pytest.fixture
def device_context():
    """Create a sample device context."""
    return DeviceContext(
        device_type="washing_machine",
        brand="TestBrand",
        model="Model123",
        detected_controls=[
            Control(
                id="ctrl1",
                type="button",
                label="Start",
                bounding_box=BoundingBox(x=0.1, y=0.2, width=0.05, height=0.05),
                confidence=0.9
            )
        ],
        safety_warnings=[]
    )


@pytest.fixture
def mock_llm_adapter():
    """Create a mock LLM adapter."""
    mock = Mock()
    mock.generate_completion = Mock(return_value="This is a test explanation.")
    return mock


class TestExplanationService:
    """Test suite for ExplanationService."""
    
    def test_initialization(self, explanation_service):
        """Test that service initializes correctly."""
        assert explanation_service is not None
        assert explanation_service._llm_adapter is None
    
    def test_detect_safety_concerns_electrical_device(self, explanation_service, device_context):
        """Test safety concern detection for electrical devices."""
        warnings = explanation_service._detect_safety_concerns(device_context, "How do I use this?")
        
        assert len(warnings) > 0
        assert any("ELECTRICAL" in w for w in warnings)
    
    def test_detect_safety_concerns_dangerous_operation(self, explanation_service):
        """Test safety concern detection for dangerous operations."""
        context = DeviceContext(
            device_type="microwave",
            detected_controls=[],
            safety_warnings=[]
        )
        
        warnings = explanation_service._detect_safety_concerns(context, "How do I repair this?")
        
        assert len(warnings) >= 1
        assert any("CAUTION" in w or "ELECTRICAL" in w for w in warnings)
    
    def test_add_disclaimer(self, explanation_service):
        """Test that disclaimer is added to text."""
        text = "This is a test explanation."
        result = explanation_service._add_disclaimer(text)
        
        assert "Disclaimer" in result
        assert text in result
    
    @patch('backend.app.services.explanation_service.ExplanationService._get_llm_adapter')
    def test_generate_explanation(self, mock_get_adapter, explanation_service, device_context, mock_llm_adapter):
        """Test explanation generation."""
        mock_get_adapter.return_value = mock_llm_adapter
        
        image = b"fake_image_data"
        question = "What does the start button do?"
        
        result = explanation_service.generate_explanation(
            image=image,
            question=question,
            device_context=device_context,
            language="en"
        )
        
        assert isinstance(result, Explanation)
        assert len(result.text) > 0
        assert "Disclaimer" in result.text
        assert len(result.warnings) > 0  # Should have electrical warning
        assert result.confidence > 0
    
    @patch('backend.app.services.explanation_service.ExplanationService._get_llm_adapter')
    def test_generate_instructions(self, mock_get_adapter, explanation_service, device_context, mock_llm_adapter):
        """Test instruction generation."""
        mock_llm_adapter.generate_completion = Mock(
            return_value="Step 1: Open the door\nStep 2: Load the clothes\nStep 3: Close the door"
        )
        mock_get_adapter.return_value = mock_llm_adapter
        
        task = "How to start a wash cycle"
        
        result = explanation_service.generate_instructions(
            task=task,
            device_context=device_context,
            language="en"
        )
        
        assert isinstance(result, list)
        assert len(result) > 0
        assert all(isinstance(step, Step) for step in result)
        assert result[0].number == 1
    
    def test_parse_steps_from_text(self, explanation_service):
        """Test parsing steps from text."""
        text = """
        Step 1: First action to take
        Step 2: Second action to take
        Step 3: Third action to take
        """
        
        steps = explanation_service._parse_steps_from_text(text, [])
        
        assert len(steps) == 3
        assert steps[0].number == 1
        assert steps[1].number == 2
        assert steps[2].number == 3
        assert "First action" in steps[0].description
    
    def test_parse_steps_with_safety_warnings(self, explanation_service):
        """Test that safety warnings are added to dangerous steps."""
        text = """
        Step 1: Unplug the device from the power outlet
        Step 2: Remove the back panel with a screwdriver
        Step 3: Check the internal wiring
        """
        
        steps = explanation_service._parse_steps_from_text(text, [])
        
        # Steps involving electrical work should have warnings
        assert any(step.warning is not None for step in steps)
    
    @patch('backend.app.services.explanation_service.ExplanationService._get_llm_adapter')
    def test_clarify_step(self, mock_get_adapter, explanation_service, device_context, mock_llm_adapter):
        """Test step clarification."""
        mock_get_adapter.return_value = mock_llm_adapter
        
        step = Step(
            number=1,
            description="Load the clothes into the drum",
            warning=None
        )
        question = "How much clothes should I load?"
        
        result = explanation_service.clarify_step(
            step=step,
            question=question,
            device_context=device_context,
            language="en"
        )
        
        assert isinstance(result, str)
        assert len(result) > 0
        assert "Disclaimer" in result
    
    def test_build_explanation_prompt(self, explanation_service, device_context):
        """Test explanation prompt building."""
        question = "What does this button do?"
        warnings = ["Test warning"]
        
        prompt = explanation_service._build_explanation_prompt(
            question=question,
            device_context=device_context,
            warnings=warnings
        )
        
        assert question in prompt
        assert device_context.device_type in prompt
        assert device_context.brand in prompt
    
    def test_build_instructions_prompt(self, explanation_service, device_context):
        """Test instructions prompt building."""
        task = "Start a wash cycle"
        warnings = ["Test warning"]
        
        prompt = explanation_service._build_instructions_prompt(
            task=task,
            device_context=device_context,
            warnings=warnings
        )
        
        assert task in prompt
        assert device_context.device_type in prompt
        assert "step" in prompt.lower()
    
    def test_build_clarification_prompt(self, explanation_service, device_context):
        """Test clarification prompt building."""
        step = Step(
            number=2,
            description="Press the start button",
            warning="Be careful"
        )
        question = "Which button is the start button?"
        
        prompt = explanation_service._build_clarification_prompt(
            step=step,
            question=question,
            device_context=device_context
        )
        
        assert question in prompt
        assert step.description in prompt
        assert str(step.number) in prompt
        assert step.warning in prompt
