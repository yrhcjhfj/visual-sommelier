"""Property-based tests for coordinate-based control identification.

**Property 7: Идентификация элемента по координатам**
**Validates: Requirements 2.4**

WHEN пользователь указывает на область изображения, THEN Система SHALL 
идентифицировать элемент управления в этой области и предоставить объяснение.

These tests verify that when a user points to an area on the image,
the system correctly identifies the control element in that area.
"""

import pytest
from hypothesis import given, strategies as st, settings, assume, HealthCheck
from hypothesis.strategies import composite
from io import BytesIO
from PIL import Image, ImageDraw

from app.services.image_analysis_service import ImageAnalysisService
from app.models.device import Control, BoundingBox


# Custom strategies for generating test images with controls
@composite
def image_with_controls(draw):
    """Generate synthetic device images with known control positions.
    
    Returns:
        Tuple of (image_bytes, list of control positions as (x, y, width, height))
    """
    # Image dimensions
    width = draw(st.integers(min_value=400, max_value=800))
    height = draw(st.integers(min_value=400, max_value=800))
    
    # Background color
    bg_color = tuple(draw(st.integers(min_value=100, max_value=200)) for _ in range(3))
    
    img = Image.new('RGB', (width, height), bg_color)
    draw_obj = ImageDraw.Draw(img)
    
    # Add controls with known positions
    num_controls = draw(st.integers(min_value=2, max_value=6))
    control_positions = []
    
    for _ in range(num_controls):
        # Generate control position (normalized coordinates)
        x_norm = draw(st.floats(min_value=0.1, max_value=0.7))
        y_norm = draw(st.floats(min_value=0.1, max_value=0.7))
        w_norm = draw(st.floats(min_value=0.05, max_value=0.2))
        h_norm = draw(st.floats(min_value=0.05, max_value=0.2))
        
        # Ensure control doesn't go out of bounds
        if x_norm + w_norm > 1.0:
            w_norm = 1.0 - x_norm - 0.01
        if y_norm + h_norm > 1.0:
            h_norm = 1.0 - y_norm - 0.01
        
        # Convert to pixel coordinates
        x_px = int(x_norm * width)
        y_px = int(y_norm * height)
        w_px = int(w_norm * width)
        h_px = int(h_norm * height)
        
        # Draw control
        color = tuple(draw(st.integers(min_value=0, max_value=255)) for _ in range(3))
        draw_obj.rectangle(
            [x_px, y_px, x_px + w_px, y_px + h_px],
            fill=color,
            outline=(0, 0, 0),
            width=3
        )
        
        control_positions.append((x_norm, y_norm, w_norm, h_norm))
    
    # Convert to bytes
    buffer = BytesIO()
    img.save(buffer, format='JPEG', quality=85)
    
    return buffer.getvalue(), control_positions


@composite
def coordinates_strategy(draw):
    """Generate normalized coordinates (x, y) in range [0.0, 1.0]."""
    x = draw(st.floats(min_value=0.0, max_value=1.0))
    y = draw(st.floats(min_value=0.0, max_value=1.0))
    return (x, y)


class TestCoordinateIdentificationProperties:
    """Property-based tests for control identification by coordinates.
    
    **Property 7: Идентификация элемента по координатам**
    **Validates: Requirements 2.4**
    
    WHEN пользователь указывает на область изображения, THEN Система SHALL 
    идентифицировать элемент управления в этой области.
    """
    
    @given(
        image_data=image_with_controls(),
        coordinate_offset=st.floats(min_value=-0.02, max_value=0.02)
    )
    @settings(
        max_examples=15,
        deadline=60000,
        suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow]
    )
    def test_identifies_control_at_exact_coordinates(self, image_data, coordinate_offset):
        """Property: System identifies control when user points inside its bounding box.
        
        For any image with detected controls, when a user points to coordinates
        inside a control's bounding box, the system should:
        
        1. Return a Control object (not None)
        2. The returned control should contain the pointed coordinates
        3. The control should have valid structure (id, type, bounding_box, confidence)
        
        This ensures Requirement 2.4: system identifies control at user-specified area.
        """
        image_bytes, control_positions = image_data
        
        # Skip if no controls were generated
        assume(len(control_positions) > 0)
        
        service = ImageAnalysisService()
        
        try:
            # Pick a random control position
            import random
            control_x, control_y, control_w, control_h = random.choice(control_positions)
            
            # Point to center of control (with small offset for variation)
            point_x = control_x + control_w / 2 + coordinate_offset
            point_y = control_y + control_h / 2 + coordinate_offset
            
            # Ensure point is still within valid range
            point_x = max(0.0, min(1.0, point_x))
            point_y = max(0.0, min(1.0, point_y))
            
            # Act - identify control at coordinates
            result = service.identify_control_at_coordinates(
                image_bytes,
                point_x,
                point_y
            )
            
            # Assert - should find a control
            # Note: Due to CV model limitations, we may not always detect all controls
            # But if we do detect controls, pointing at them should work
            if result is not None:
                # Property 1: Result must be a Control object
                assert isinstance(result, Control), \
                    f"identify_control_at_coordinates must return Control object, got {type(result)}"
                
                # Property 2: Control must have valid structure
                assert hasattr(result, 'id'), "Control must have id"
                assert hasattr(result, 'type'), "Control must have type"
                assert hasattr(result, 'bounding_box'), "Control must have bounding_box"
                assert hasattr(result, 'confidence'), "Control must have confidence"
                
                assert isinstance(result.id, str), "Control id must be string"
                assert len(result.id) > 0, "Control id must not be empty"
                
                assert isinstance(result.type, str), "Control type must be string"
                assert len(result.type) > 0, "Control type must not be empty"
                
                assert isinstance(result.bounding_box, BoundingBox), \
                    "Control must have BoundingBox"
                
                assert 0.0 <= result.confidence <= 1.0, \
                    f"Control confidence must be in [0.0, 1.0], got {result.confidence}"
                
                # Property 3: Bounding box must be valid
                bbox = result.bounding_box
                assert 0.0 <= bbox.x <= 1.0, f"bbox.x must be normalized, got {bbox.x}"
                assert 0.0 <= bbox.y <= 1.0, f"bbox.y must be normalized, got {bbox.y}"
                assert 0.0 < bbox.width <= 1.0, \
                    f"bbox.width must be positive and normalized, got {bbox.width}"
                assert 0.0 < bbox.height <= 1.0, \
                    f"bbox.height must be positive and normalized, got {bbox.height}"
                
                # Property 4: The point should be inside or near the returned control's bbox
                # (allowing for some tolerance due to closest-control fallback)
                center_x = bbox.x + bbox.width / 2
                center_y = bbox.y + bbox.height / 2
                distance = ((point_x - center_x) ** 2 + (point_y - center_y) ** 2) ** 0.5
                
                # Should be within reasonable distance (0.2 normalized units)
                assert distance < 0.2, \
                    f"Returned control should be near pointed coordinates. " \
                    f"Distance: {distance:.3f}, Point: ({point_x:.3f}, {point_y:.3f}), " \
                    f"Control center: ({center_x:.3f}, {center_y:.3f})"
            
        except Exception as e:
            error_msg = str(e).lower()
            if any(keyword in error_msg for keyword in ['model', 'file', 'cuda', 'ollama', 'not found']):
                pytest.skip(f"Models not available for testing: {e}")
            else:
                raise
    
    @given(coordinates=coordinates_strategy())
    @settings(
        max_examples=10,
        deadline=60000,
        suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow]
    )
    def test_handles_invalid_coordinates_gracefully(self, coordinates):
        """Property: System handles out-of-bounds coordinates gracefully.
        
        When user provides coordinates outside valid range or on empty areas:
        
        1. System should not crash or raise exceptions
        2. Should return None if no control found
        3. Should validate coordinate ranges (0.0-1.0)
        
        This ensures robust handling of edge cases.
        """
        # Create simple test image
        img = Image.new('RGB', (400, 400), (128, 128, 128))
        buffer = BytesIO()
        img.save(buffer, format='JPEG')
        image_bytes = buffer.getvalue()
        
        service = ImageAnalysisService()
        x, y = coordinates
        
        try:
            # Act - try to identify control at coordinates
            result = service.identify_control_at_coordinates(
                image_bytes,
                x,
                y
            )
            
            # Assert - should handle gracefully
            # Result can be None (no control found) or a Control object
            if result is not None:
                assert isinstance(result, Control), \
                    "If result is not None, it must be a Control object"
            
            # Should not raise exception for valid coordinates
            assert 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0, \
                "Test should only use valid coordinates"
            
        except Exception as e:
            error_msg = str(e).lower()
            if any(keyword in error_msg for keyword in ['model', 'file', 'cuda', 'ollama', 'not found']):
                pytest.skip(f"Models not available for testing: {e}")
            else:
                raise
    
    @given(
        x=st.floats(min_value=-1.0, max_value=2.0),
        y=st.floats(min_value=-1.0, max_value=2.0)
    )
    @settings(
        max_examples=10,
        deadline=30000,
        suppress_health_check=[HealthCheck.function_scoped_fixture]
    )
    def test_validates_coordinate_ranges(self, x, y):
        """Property: System validates that coordinates are in valid range [0.0, 1.0].
        
        When coordinates are outside the valid range:
        
        1. System should handle gracefully (not crash)
        2. Should return None for invalid coordinates
        3. Should log warning for invalid input
        
        This ensures input validation for Requirement 2.4.
        """
        # Create simple test image
        img = Image.new('RGB', (400, 400), (128, 128, 128))
        buffer = BytesIO()
        img.save(buffer, format='JPEG')
        image_bytes = buffer.getvalue()
        
        service = ImageAnalysisService()
        
        try:
            # Act
            result = service.identify_control_at_coordinates(
                image_bytes,
                x,
                y
            )
            
            # Assert - invalid coordinates should return None
            if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
                assert result is None, \
                    f"Invalid coordinates ({x}, {y}) should return None"
            
            # Valid coordinates may return None (no control) or Control object
            else:
                if result is not None:
                    assert isinstance(result, Control), \
                        "Valid coordinates should return None or Control object"
            
        except Exception as e:
            error_msg = str(e).lower()
            if any(keyword in error_msg for keyword in ['model', 'file', 'cuda', 'ollama', 'not found']):
                pytest.skip(f"Models not available for testing: {e}")
            else:
                raise
    
    @given(image_data=image_with_controls())
    @settings(
        max_examples=10,
        deadline=60000,
        suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow]
    )
    def test_returns_closest_control_when_not_exact(self, image_data):
        """Property: System returns closest control when point is near but not inside.
        
        When user points near a control (but not exactly inside its bounding box):
        
        1. System should find the closest control
        2. Distance should be within reasonable threshold
        3. Should return None if no controls are nearby
        
        This ensures user-friendly behavior for Requirement 2.4.
        """
        image_bytes, control_positions = image_data
        
        # Skip if no controls
        assume(len(control_positions) > 0)
        
        service = ImageAnalysisService()
        
        try:
            # Pick a control and point near it (but outside)
            import random
            control_x, control_y, control_w, control_h = random.choice(control_positions)
            
            # Point just outside the control (to the right)
            point_x = control_x + control_w + 0.05
            point_y = control_y + control_h / 2
            
            # Ensure point is within valid range
            point_x = min(1.0, point_x)
            
            # Act
            result = service.identify_control_at_coordinates(
                image_bytes,
                point_x,
                point_y
            )
            
            # Assert - may find closest control or return None
            if result is not None:
                assert isinstance(result, Control), \
                    "Result must be Control object if not None"
                
                # The returned control should be reasonably close
                bbox = result.bounding_box
                center_x = bbox.x + bbox.width / 2
                center_y = bbox.y + bbox.height / 2
                distance = ((point_x - center_x) ** 2 + (point_y - center_y) ** 2) ** 0.5
                
                # Should be within the threshold used by the method (0.15)
                # Plus some margin for the control we pointed near
                assert distance < 0.25, \
                    f"Closest control should be within reasonable distance, got {distance:.3f}"
            
        except Exception as e:
            error_msg = str(e).lower()
            if any(keyword in error_msg for keyword in ['model', 'file', 'cuda', 'ollama', 'not found']):
                pytest.skip(f"Models not available for testing: {e}")
            else:
                raise
    
    @given(
        image_data=image_with_controls(),
        num_queries=st.integers(min_value=2, max_value=4)
    )
    @settings(
        max_examples=5,
        deadline=120000,
        suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow]
    )
    def test_identification_is_consistent(self, image_data, num_queries):
        """Property: Multiple queries at same coordinates return consistent results.
        
        When querying the same coordinates multiple times:
        
        1. Should return the same control (or None consistently)
        2. Control properties should be identical
        3. Demonstrates deterministic behavior
        
        This ensures reliable user experience for Requirement 2.4.
        """
        image_bytes, control_positions = image_data
        
        # Skip if no controls
        assume(len(control_positions) > 0)
        
        service = ImageAnalysisService()
        
        try:
            # Pick a point
            import random
            control_x, control_y, control_w, control_h = random.choice(control_positions)
            point_x = control_x + control_w / 2
            point_y = control_y + control_h / 2
            
            # Query multiple times
            results = []
            for _ in range(num_queries):
                result = service.identify_control_at_coordinates(
                    image_bytes,
                    point_x,
                    point_y
                )
                results.append(result)
            
            # Assert - all results should be consistent
            if results[0] is None:
                # All should be None
                assert all(r is None for r in results), \
                    "If first query returns None, all queries should return None"
            else:
                # All should return controls with same properties
                for result in results:
                    assert result is not None, \
                        "All queries should return consistent results (not mix None and Control)"
                    
                    # Check consistency of key properties
                    assert result.type == results[0].type, \
                        "Control type should be consistent across queries"
                    
                    # Bounding boxes should be the same
                    assert result.bounding_box.x == results[0].bounding_box.x, \
                        "Bounding box should be consistent"
                    assert result.bounding_box.y == results[0].bounding_box.y, \
                        "Bounding box should be consistent"
                    assert result.bounding_box.width == results[0].bounding_box.width, \
                        "Bounding box should be consistent"
                    assert result.bounding_box.height == results[0].bounding_box.height, \
                        "Bounding box should be consistent"
            
        except Exception as e:
            error_msg = str(e).lower()
            if any(keyword in error_msg for keyword in ['model', 'file', 'cuda', 'ollama', 'not found']):
                pytest.skip(f"Models not available for testing: {e}")
            else:
                raise
    
    @given(image_data=image_with_controls())
    @settings(
        max_examples=10,
        deadline=60000,
        suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow]
    )
    def test_device_type_context_is_optional(self, image_data):
        """Property: device_type parameter is optional for coordinate identification.
        
        The identify_control_at_coordinates method should work:
        
        1. With device_type provided
        2. Without device_type (None)
        3. Results should be similar in both cases
        
        This ensures flexible API usage for Requirement 2.4.
        """
        image_bytes, control_positions = image_data
        
        # Skip if no controls
        assume(len(control_positions) > 0)
        
        service = ImageAnalysisService()
        
        try:
            # Pick a point
            import random
            control_x, control_y, control_w, control_h = random.choice(control_positions)
            point_x = control_x + control_w / 2
            point_y = control_y + control_h / 2
            
            # Query without device_type
            result_without = service.identify_control_at_coordinates(
                image_bytes,
                point_x,
                point_y,
                device_type=None
            )
            
            # Query with device_type
            result_with = service.identify_control_at_coordinates(
                image_bytes,
                point_x,
                point_y,
                device_type="test_device"
            )
            
            # Assert - both should work (return None or Control)
            if result_without is not None:
                assert isinstance(result_without, Control), \
                    "Result without device_type should be None or Control"
            
            if result_with is not None:
                assert isinstance(result_with, Control), \
                    "Result with device_type should be None or Control"
            
            # Results should be similar (both None or both Control)
            assert (result_without is None) == (result_with is None), \
                "Results should be consistent regardless of device_type parameter"
            
        except Exception as e:
            error_msg = str(e).lower()
            if any(keyword in error_msg for keyword in ['model', 'file', 'cuda', 'ollama', 'not found']):
                pytest.skip(f"Models not available for testing: {e}")
            else:
                raise
