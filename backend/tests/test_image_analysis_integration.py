"""Integration tests for ImageAnalysisService with real adapters."""

import pytest
from io import BytesIO
from PIL import Image, ImageDraw
import os

from app.services.image_analysis_service import ImageAnalysisService
from app.models.device import BoundingBox


@pytest.mark.skipif(
    not os.path.exists("yolov8n.pt"),
    reason="YOLO model not available"
)
class TestImageAnalysisIntegration:
    """Integration tests with real models."""
    
    @pytest.fixture
    def service(self):
        """Create service instance."""
        return ImageAnalysisService()
    
    @pytest.fixture
    def device_image_bytes(self):
        """Create a more realistic device image."""
        # Create an image with some shapes that might look like controls
        img = Image.new('RGB', (640, 480), color='lightgray')
        draw = ImageDraw.Draw(img)
        
        # Draw some button-like shapes
        draw.rectangle([100, 100, 150, 130], fill='red', outline='black')
        draw.rectangle([200, 100, 250, 130], fill='green', outline='black')
        draw.rectangle([300, 100, 350, 130], fill='blue', outline='black')
        
        # Draw some text-like elements
        draw.text((110, 110), "ON", fill='white')
        draw.text((210, 110), "OFF", fill='white')
        
        buffer = BytesIO()
        img.save(buffer, format='PNG')
        buffer.seek(0)
        return buffer.read()

    def test_full_device_analysis_workflow(self, service, device_image_bytes):
        """Test complete workflow: analyze device -> detect controls -> highlight area."""
        # Step 1: Analyze device (Requirement 1.1)
        result = service.analyze_device(device_image_bytes)
        
        assert result is not None
        assert result.device_type is not None
        assert 0.0 <= result.confidence <= 1.0
        
        # Step 2: Verify low confidence handling (Requirement 1.3)
        if result.confidence < service.HIGH_CONFIDENCE_THRESHOLD:
            assert len(result.suggested_categories) > 0
            # Should include standard categories
            assert any(cat in result.suggested_categories 
                      for cat in ["washing_machine", "remote_control", "microwave"])
        
        # Step 3: Verify controls detected (Requirement 1.1, 2.4)
        assert isinstance(result.detected_controls, list)
        
        # Step 4: If controls detected, highlight one (Requirement 2.4)
        if len(result.detected_controls) > 0:
            control = result.detected_controls[0]
            highlighted = service.highlight_area(
                device_image_bytes,
                control.bounding_box
            )
            
            assert isinstance(highlighted, bytes)
            assert len(highlighted) > 0
            
            # Verify it's a valid image
            img = Image.open(BytesIO(highlighted))
            assert img is not None
    
    def test_detect_controls_with_labels(self, service, device_image_bytes):
        """Test control detection includes text labels when available."""
        controls = service.detect_controls(device_image_bytes, "remote_control")
        
        # Should return a list
        assert isinstance(controls, list)
        
        # Each control should have proper structure
        for control in controls:
            assert control.id is not None
            assert len(control.id) > 0
            assert control.type is not None
            assert control.bounding_box is not None
            assert 0.0 <= control.confidence <= 1.0
            
            # Bounding box should be valid
            bbox = control.bounding_box
            assert 0.0 <= bbox.x <= 1.0
            assert 0.0 <= bbox.y <= 1.0
            assert 0.0 < bbox.width <= 1.0
            assert 0.0 < bbox.height <= 1.0
    
    def test_highlight_multiple_areas(self, service, device_image_bytes):
        """Test highlighting multiple areas on the same image."""
        # Create multiple bounding boxes
        bbox1 = BoundingBox(x=0.1, y=0.1, width=0.2, height=0.15)
        bbox2 = BoundingBox(x=0.4, y=0.1, width=0.2, height=0.15)
        
        # Highlight first area
        result1 = service.highlight_area(device_image_bytes, bbox1, color="red")
        assert isinstance(result1, bytes)
        
        # Highlight second area on the modified image
        result2 = service.highlight_area(result1, bbox2, color="blue")
        assert isinstance(result2, bytes)
        
        # Should be a valid image
        img = Image.open(BytesIO(result2))
        assert img is not None
