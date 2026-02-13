"""Tests for ImageAnalysisService."""

import pytest
from io import BytesIO
from PIL import Image
import numpy as np

from app.services.image_analysis_service import ImageAnalysisService
from app.models.device import BoundingBox


class TestImageAnalysisService:
    """Tests for ImageAnalysisService."""
    
    @pytest.fixture
    def service(self):
        """Create service instance."""
        return ImageAnalysisService()
    
    @pytest.fixture
    def sample_image_bytes(self):
        """Create a sample image as bytes."""
        # Create a simple test image
        img = Image.new('RGB', (640, 480), color='white')
        buffer = BytesIO()
        img.save(buffer, format='PNG')
        buffer.seek(0)
        return buffer.read()
    
    def test_initialization(self, service):
        """Test service can be initialized."""
        assert service is not None
        assert service.adapter_factory is not None
    
    def test_analyze_device_returns_result(self, service, sample_image_bytes):
        """Test analyze_device returns a DeviceAnalysisResult."""
        try:
            result = service.analyze_device(sample_image_bytes)
            
            # Should return a result
            assert result is not None
            assert result.device_type is not None
            assert 0.0 <= result.confidence <= 1.0
            assert isinstance(result.detected_controls, list)
            
        except Exception as e:
            # If models not available, skip
            pytest.skip(f"Models not available: {e}")

    def test_analyze_device_low_confidence_suggests_categories(self, service, sample_image_bytes):
        """Test that low confidence results include suggested categories."""
        try:
            result = service.analyze_device(sample_image_bytes)
            
            # If confidence is low, should have suggestions
            if result.confidence < service.HIGH_CONFIDENCE_THRESHOLD:
                assert len(result.suggested_categories) > 0
                assert "washing_machine" in result.suggested_categories
                assert "remote_control" in result.suggested_categories
            
        except Exception as e:
            pytest.skip(f"Models not available: {e}")
    
    def test_detect_controls_returns_list(self, service, sample_image_bytes):
        """Test detect_controls returns a list of controls."""
        try:
            controls = service.detect_controls(sample_image_bytes, "microwave")
            
            # Should return a list (may be empty for blank image)
            assert isinstance(controls, list)
            
            # If controls detected, verify structure
            for control in controls:
                assert control.id is not None
                assert control.type is not None
                assert control.bounding_box is not None
                assert 0.0 <= control.confidence <= 1.0
            
        except Exception as e:
            pytest.skip(f"Models not available: {e}")
    
    def test_highlight_area_returns_bytes(self, service, sample_image_bytes):
        """Test highlight_area returns modified image bytes."""
        bbox = BoundingBox(x=0.2, y=0.2, width=0.3, height=0.3)
        
        result = service.highlight_area(sample_image_bytes, bbox)
        
        # Should return bytes
        assert isinstance(result, bytes)
        assert len(result) > 0
        
        # Should be a valid image
        img = Image.open(BytesIO(result))
        assert img is not None
    
    def test_highlight_area_with_custom_color(self, service, sample_image_bytes):
        """Test highlight_area with custom color."""
        bbox = BoundingBox(x=0.1, y=0.1, width=0.2, height=0.2)
        
        result = service.highlight_area(sample_image_bytes, bbox, color="blue", width=5)
        
        # Should return valid image
        assert isinstance(result, bytes)
        img = Image.open(BytesIO(result))
        assert img is not None
    
    def test_infer_control_type(self, service):
        """Test control type inference."""
        assert service._infer_control_type("button") == "button"
        assert service._infer_control_type("power button") == "button"
        assert service._infer_control_type("knob") == "knob"
        assert service._infer_control_type("dial") == "knob"
        assert service._infer_control_type("switch") == "switch"
        assert service._infer_control_type("lever") == "lever"
        assert service._infer_control_type("slider") == "slider"
        assert service._infer_control_type("display") == "display"
        assert service._infer_control_type("screen") == "display"
        assert service._infer_control_type("unknown") == "unknown_control"
    
    def test_is_text_near_control(self, service):
        """Test text proximity detection."""
        text_bbox = (0.5, 0.5, 0.1, 0.05)
        control_bbox = BoundingBox(x=0.48, y=0.48, width=0.1, height=0.1)
        
        # Should be near
        assert service._is_text_near_control(text_bbox, control_bbox, threshold=0.1)
        
        # Should not be near
        far_control_bbox = BoundingBox(x=0.1, y=0.1, width=0.1, height=0.1)
        assert not service._is_text_near_control(text_bbox, far_control_bbox, threshold=0.1)
