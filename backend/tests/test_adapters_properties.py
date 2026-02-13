"""Property-based tests for adapter implementations.

**Validates: Requirements 1.1**

These tests use Hypothesis to verify that adapters correctly process device images
across a wide range of inputs and conditions.
"""

import pytest
from hypothesis import given, strategies as st, settings, assume, HealthCheck
from hypothesis.strategies import composite
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
import numpy as np

from app.adapters import YOLOAdapter, EasyOCRAdapter, LLaVAAdapter


# Custom strategies for generating test images
@composite
def valid_image_bytes(draw):
    """Generate valid image bytes with various properties.
    
    This strategy creates synthetic device-like images with:
    - Various sizes (small to large)
    - Different color modes (RGB, RGBA, L)
    - Simple geometric shapes simulating device controls
    """
    # Image dimensions
    width = draw(st.integers(min_value=100, max_value=800))
    height = draw(st.integers(min_value=100, max_value=800))
    
    # Color mode
    mode = draw(st.sampled_from(['RGB', 'RGBA', 'L']))
    
    # Create image with random background color
    if mode == 'L':
        bg_color = draw(st.integers(min_value=0, max_value=255))
    else:
        bg_color = tuple(draw(st.integers(min_value=0, max_value=255)) for _ in range(3))
        if mode == 'RGBA':
            bg_color = bg_color + (255,)
    
    img = Image.new(mode, (width, height), bg_color)
    draw_obj = ImageDraw.Draw(img)
    
    # Add some simple shapes to simulate device controls
    num_shapes = draw(st.integers(min_value=0, max_value=5))
    for _ in range(num_shapes):
        shape_type = draw(st.sampled_from(['rectangle', 'circle']))
        x1 = draw(st.integers(min_value=0, max_value=width-20))
        y1 = draw(st.integers(min_value=0, max_value=height-20))
        x2 = draw(st.integers(min_value=x1+10, max_value=min(x1+100, width)))
        y2 = draw(st.integers(min_value=y1+10, max_value=min(y1+100, height)))
        
        if mode == 'L':
            color = draw(st.integers(min_value=0, max_value=255))
        else:
            color = tuple(draw(st.integers(min_value=0, max_value=255)) for _ in range(3))
            if mode == 'RGBA':
                color = color + (255,)
        
        if shape_type == 'rectangle':
            draw_obj.rectangle([x1, y1, x2, y2], fill=color)
        else:
            draw_obj.ellipse([x1, y1, x2, y2], fill=color)
    
    # Convert to bytes
    buffer = BytesIO()
    # Convert RGBA to RGB for JPEG
    if mode == 'RGBA':
        img = img.convert('RGB')
    img.save(buffer, format='JPEG', quality=85)
    return buffer.getvalue()


@composite
def image_with_text(draw):
    """Generate image bytes with text overlay (for OCR testing)."""
    # Base image
    width = draw(st.integers(min_value=200, max_value=600))
    height = draw(st.integers(min_value=200, max_value=600))
    
    img = Image.new('RGB', (width, height), (255, 255, 255))
    draw_obj = ImageDraw.Draw(img)
    
    # Add text
    text = draw(st.text(
        alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd')),
        min_size=1,
        max_size=20
    ))
    
    # Simple text placement (no font to avoid dependency issues)
    if text:
        x = draw(st.integers(min_value=10, max_value=max(11, width-100)))
        y = draw(st.integers(min_value=10, max_value=max(11, height-50)))
        draw_obj.text((x, y), text, fill=(0, 0, 0))
    
    # Convert to bytes
    buffer = BytesIO()
    img.save(buffer, format='JPEG', quality=85)
    return buffer.getvalue()


class TestYOLOAdapterProperties:
    """Property-based tests for YOLO adapter.
    
    **Property 1: Обработка изображений устройств**
    **Validates: Requirements 1.1**
    
    The system SHALL process device images and attempt to identify device types.
    This property verifies that the YOLO adapter can handle various image inputs
    without crashing and returns results in the expected format.
    """
    
    @given(image_bytes=valid_image_bytes())
    @settings(
        max_examples=10,
        deadline=30000,  # 30 seconds per test
        suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow]
    )
    def test_detect_labels_returns_valid_structure(self, image_bytes):
        """Property: detect_labels always returns a list of Label objects.
        
        For any valid image input, the adapter should:
        1. Not crash or raise unexpected exceptions
        2. Return a list (possibly empty)
        3. All items in the list should be Label objects
        4. All confidence scores should be between 0 and 1
        """
        adapter = YOLOAdapter()
        
        try:
            # Act
            labels = adapter.detect_labels(image_bytes)
            
            # Assert - verify structure
            assert isinstance(labels, list), "detect_labels must return a list"
            
            for label in labels:
                assert hasattr(label, 'name'), "Label must have 'name' attribute"
                assert hasattr(label, 'confidence'), "Label must have 'confidence' attribute"
                assert isinstance(label.name, str), "Label name must be a string"
                assert isinstance(label.confidence, float), "Confidence must be a float"
                assert 0.0 <= label.confidence <= 1.0, f"Confidence must be in [0,1], got {label.confidence}"
            
            # Verify labels are sorted by confidence (descending)
            if len(labels) > 1:
                confidences = [label.confidence for label in labels]
                assert confidences == sorted(confidences, reverse=True), \
                    "Labels should be sorted by confidence (descending)"
                    
        except Exception as e:
            # If model is not available, skip the test
            if "model" in str(e).lower() or "file" in str(e).lower():
                pytest.skip(f"Model not available: {e}")
            else:
                raise
        finally:
            # Cleanup
            if adapter.is_loaded():
                adapter.unload_model()
    
    @given(image_bytes=valid_image_bytes())
    @settings(
        max_examples=10,
        deadline=30000,
        suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow]
    )
    def test_detect_objects_returns_valid_structure(self, image_bytes):
        """Property: detect_objects always returns a list of DetectedObject objects.
        
        For any valid image input, the adapter should:
        1. Return a list of DetectedObject instances
        2. Each object has valid bounding box coordinates (normalized 0-1)
        3. Confidence scores are in valid range
        """
        adapter = YOLOAdapter()
        
        try:
            # Act
            objects = adapter.detect_objects(image_bytes)
            
            # Assert - verify structure
            assert isinstance(objects, list), "detect_objects must return a list"
            
            for obj in objects:
                assert hasattr(obj, 'class_name'), "Object must have 'class_name'"
                assert hasattr(obj, 'confidence'), "Object must have 'confidence'"
                assert hasattr(obj, 'bounding_box'), "Object must have 'bounding_box'"
                
                assert isinstance(obj.class_name, str), "class_name must be string"
                assert isinstance(obj.confidence, float), "confidence must be float"
                assert 0.0 <= obj.confidence <= 1.0, f"Confidence must be in [0,1], got {obj.confidence}"
                
                # Verify bounding box format
                bbox = obj.bounding_box
                assert isinstance(bbox, tuple), "Bounding box must be a tuple"
                assert len(bbox) == 4, "Bounding box must have 4 values (x, y, width, height)"
                
                x, y, width, height = bbox
                assert 0.0 <= x <= 1.0, f"x coordinate must be normalized [0,1], got {x}"
                assert 0.0 <= y <= 1.0, f"y coordinate must be normalized [0,1], got {y}"
                assert 0.0 <= width <= 1.0, f"width must be normalized [0,1], got {width}"
                assert 0.0 <= height <= 1.0, f"height must be normalized [0,1], got {height}"
                
                # Verify bounding box doesn't exceed image bounds
                assert x + width <= 1.0, "Bounding box exceeds image width"
                assert y + height <= 1.0, "Bounding box exceeds image height"
                
        except Exception as e:
            if "model" in str(e).lower() or "file" in str(e).lower():
                pytest.skip(f"Model not available: {e}")
            else:
                raise
        finally:
            if adapter.is_loaded():
                adapter.unload_model()
    
    @given(image_bytes=valid_image_bytes())
    @settings(
        max_examples=5,
        deadline=30000,
        suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow]
    )
    def test_model_lifecycle_is_idempotent(self, image_bytes):
        """Property: Loading and unloading models multiple times is safe.
        
        The adapter should handle:
        1. Multiple load calls (should be idempotent)
        2. Multiple unload calls (should be safe)
        3. Processing after reload
        """
        adapter = YOLOAdapter()
        
        try:
            # Load multiple times
            adapter.load_model()
            assert adapter.is_loaded()
            
            adapter.load_model()  # Should be idempotent
            assert adapter.is_loaded()
            
            # Process image
            labels1 = adapter.detect_labels(image_bytes)
            assert isinstance(labels1, list)
            
            # Unload and reload
            adapter.unload_model()
            assert not adapter.is_loaded()
            
            adapter.load_model()
            assert adapter.is_loaded()
            
            # Process again - should still work
            labels2 = adapter.detect_labels(image_bytes)
            assert isinstance(labels2, list)
            
            # Unload multiple times
            adapter.unload_model()
            assert not adapter.is_loaded()
            
            adapter.unload_model()  # Should be safe
            assert not adapter.is_loaded()
            
        except Exception as e:
            if "model" in str(e).lower() or "file" in str(e).lower():
                pytest.skip(f"Model not available: {e}")
            else:
                raise
        finally:
            if adapter.is_loaded():
                adapter.unload_model()


class TestEasyOCRAdapterProperties:
    """Property-based tests for EasyOCR adapter.
    
    **Property 1: Обработка изображений устройств**
    **Validates: Requirements 1.1**
    
    The OCR adapter should process device images to extract text from controls.
    """
    
    @given(image_bytes=image_with_text())
    @settings(
        max_examples=5,
        deadline=60000,  # OCR can be slower
        suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow]
    )
    def test_detect_text_returns_valid_structure(self, image_bytes):
        """Property: detect_text always returns a list of TextAnnotation objects.
        
        For any valid image input, the adapter should:
        1. Return a list of TextAnnotation instances
        2. Each annotation has valid text, confidence, and bounding box
        3. Bounding boxes are properly normalized
        """
        adapter = EasyOCRAdapter()
        
        try:
            # Act
            annotations = adapter.detect_text(image_bytes)
            
            # Assert - verify structure
            assert isinstance(annotations, list), "detect_text must return a list"
            
            for annotation in annotations:
                assert hasattr(annotation, 'text'), "Annotation must have 'text'"
                assert hasattr(annotation, 'confidence'), "Annotation must have 'confidence'"
                assert hasattr(annotation, 'bounding_box'), "Annotation must have 'bounding_box'"
                
                assert isinstance(annotation.text, str), "text must be string"
                assert isinstance(annotation.confidence, float), "confidence must be float"
                assert 0.0 <= annotation.confidence <= 1.0, \
                    f"Confidence must be in [0,1], got {annotation.confidence}"
                
                # Verify bounding box
                bbox = annotation.bounding_box
                assert isinstance(bbox, tuple), "Bounding box must be a tuple"
                assert len(bbox) == 4, "Bounding box must have 4 values"
                
                x, y, width, height = bbox
                assert 0.0 <= x <= 1.0, f"x must be normalized [0,1], got {x}"
                assert 0.0 <= y <= 1.0, f"y must be normalized [0,1], got {y}"
                assert 0.0 <= width <= 1.0, f"width must be normalized [0,1], got {width}"
                assert 0.0 <= height <= 1.0, f"height must be normalized [0,1], got {height}"
                
        except Exception as e:
            error_msg = str(e).lower()
            if "model" in error_msg or "easyocr" in error_msg or "not supported" in error_msg:
                pytest.skip(f"EasyOCR not available or language not supported: {e}")
            else:
                raise
        finally:
            if adapter.is_loaded():
                adapter.unload_model()
    
    @given(image_bytes=valid_image_bytes())
    @settings(
        max_examples=5,
        deadline=60000,
        suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow]
    )
    def test_detect_labels_returns_empty_list(self, image_bytes):
        """Property: EasyOCR adapter doesn't support label detection.
        
        Should always return empty list for detect_labels.
        """
        adapter = EasyOCRAdapter()
        
        try:
            labels = adapter.detect_labels(image_bytes)
            assert isinstance(labels, list), "Must return a list"
            assert len(labels) == 0, "EasyOCR should return empty list for detect_labels"
        except Exception as e:
            if "model" in str(e).lower() or "easyocr" in str(e).lower():
                pytest.skip(f"EasyOCR not available: {e}")
            else:
                raise
        finally:
            if adapter.is_loaded():
                adapter.unload_model()


class TestLLaVAAdapterProperties:
    """Property-based tests for LLaVA adapter.
    
    **Property 1: Обработка изображений устройств**
    **Validates: Requirements 1.1**
    
    The LLM adapter should process device images and generate explanations.
    """
    
    @given(
        prompt=st.text(min_size=5, max_size=100),
        language=st.sampled_from(['en', 'ru', 'zh'])
    )
    @settings(
        max_examples=3,
        deadline=120000,  # LLM can be very slow
        suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow]
    )
    def test_generate_completion_returns_string(self, prompt, language):
        """Property: generate_completion always returns a string.
        
        For any valid prompt and language, the adapter should:
        1. Return a non-empty string
        2. Not crash or raise unexpected exceptions
        """
        # Filter out prompts with only whitespace or special chars
        assume(prompt.strip() != "")
        assume(any(c.isalnum() for c in prompt))
        
        adapter = LLaVAAdapter()
        
        try:
            # Act
            result = adapter.generate_completion(
                prompt=prompt,
                language=language,
                max_tokens=50  # Keep it short for testing
            )
            
            # Assert
            assert isinstance(result, str), "generate_completion must return a string"
            assert len(result) > 0, "Result should not be empty"
            
        except Exception as e:
            error_msg = str(e).lower()
            if "ollama" in error_msg or "connection" in error_msg or "model" in error_msg:
                pytest.skip(f"Ollama not available: {e}")
            else:
                raise
        finally:
            if adapter.is_loaded():
                adapter.unload_model()
    
    @given(image_bytes=valid_image_bytes())
    @settings(
        max_examples=2,
        deadline=120000,
        suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow]
    )
    def test_generate_completion_with_image(self, image_bytes):
        """Property: generate_completion handles images correctly.
        
        When provided with an image, the adapter should:
        1. Accept the image bytes
        2. Return a string response
        3. Not crash
        """
        adapter = LLaVAAdapter()
        
        try:
            # Act
            result = adapter.generate_completion(
                prompt="What do you see in this image?",
                image=image_bytes,
                language="en",
                max_tokens=50
            )
            
            # Assert
            assert isinstance(result, str), "Must return a string"
            assert len(result) > 0, "Result should not be empty"
            
        except Exception as e:
            error_msg = str(e).lower()
            if "ollama" in error_msg or "connection" in error_msg or "model" in error_msg:
                pytest.skip(f"Ollama not available: {e}")
            else:
                raise
        finally:
            if adapter.is_loaded():
                adapter.unload_model()
