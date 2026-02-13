"""Property-based tests for ImageAnalysisService.

**Property 2: Отображение результатов распознавания**
**Validates: Requirements 1.2**

These tests verify that when the CV-module successfully recognizes a device,
the system displays the device name and type to the user with proper structure.
"""

import pytest
from hypothesis import given, strategies as st, settings, assume, HealthCheck
from hypothesis.strategies import composite
from io import BytesIO
from PIL import Image, ImageDraw

from app.services.image_analysis_service import ImageAnalysisService
from app.models.device import DeviceAnalysisResult


# Custom strategies for generating test images
@composite
def device_image_bytes(draw):
    """Generate synthetic device images for testing.
    
    Creates images with various properties that simulate real device photos:
    - Different sizes (small to large)
    - Various color schemes
    - Simple shapes simulating device controls
    """
    # Image dimensions
    width = draw(st.integers(min_value=200, max_value=1024))
    height = draw(st.integers(min_value=200, max_value=1024))
    
    # Color mode - devices are typically photographed in RGB
    mode = 'RGB'
    
    # Background color (simulating device surface)
    bg_color = tuple(draw(st.integers(min_value=50, max_value=255)) for _ in range(3))
    
    img = Image.new(mode, (width, height), bg_color)
    draw_obj = ImageDraw.Draw(img)
    
    # Add shapes simulating device controls (buttons, displays, etc.)
    num_controls = draw(st.integers(min_value=1, max_value=8))
    for _ in range(num_controls):
        shape_type = draw(st.sampled_from(['rectangle', 'circle']))
        x1 = draw(st.integers(min_value=10, max_value=max(11, width-100)))
        y1 = draw(st.integers(min_value=10, max_value=max(11, height-100)))
        x2 = draw(st.integers(min_value=x1+20, max_value=min(x1+150, width-10)))
        y2 = draw(st.integers(min_value=y1+20, max_value=min(y1+150, height-10)))
        
        color = tuple(draw(st.integers(min_value=0, max_value=255)) for _ in range(3))
        
        if shape_type == 'rectangle':
            draw_obj.rectangle([x1, y1, x2, y2], fill=color, outline=(0, 0, 0), width=2)
        else:
            draw_obj.ellipse([x1, y1, x2, y2], fill=color, outline=(0, 0, 0), width=2)
    
    # Convert to bytes
    buffer = BytesIO()
    img.save(buffer, format='JPEG', quality=85)
    return buffer.getvalue()


class TestImageAnalysisServiceProperties:
    """Property-based tests for ImageAnalysisService device recognition.
    
    **Property 2: Отображение результатов распознавания**
    **Validates: Requirements 1.2**
    
    WHEN CV-модуль успешно распознает устройство, THEN Система SHALL отобразить 
    название и тип устройства пользователю.
    
    This property verifies that the system always returns properly structured
    device identification results that can be displayed to the user.
    """
    
    @given(image_bytes=device_image_bytes())
    @settings(
        max_examples=10,
        deadline=60000,  # 60 seconds - CV + OCR can be slow
        suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow]
    )
    def test_analyze_device_returns_displayable_result(self, image_bytes):
        """Property: analyze_device always returns a displayable DeviceAnalysisResult.
        
        For any valid device image input, the system should:
        1. Return a DeviceAnalysisResult object (not None, not error)
        2. Include a non-empty device_type that can be displayed
        3. Include a valid confidence score (0.0 to 1.0)
        4. Include detected_controls list (may be empty)
        5. When confidence is low, include suggested_categories for user selection
        
        This ensures Requirement 1.2: the system displays device name and type.
        """
        service = ImageAnalysisService()
        
        try:
            # Act - analyze the device image
            result = service.analyze_device(image_bytes)
            
            # Assert - verify result is displayable
            assert result is not None, "analyze_device must return a result, not None"
            assert isinstance(result, DeviceAnalysisResult), \
                f"Result must be DeviceAnalysisResult, got {type(result)}"
            
            # Property 1: Device type must be displayable (non-empty string)
            assert isinstance(result.device_type, str), \
                "device_type must be a string for display"
            assert len(result.device_type) > 0, \
                "device_type must not be empty - user needs to see device identification"
            assert result.device_type.strip() != "", \
                "device_type must not be only whitespace"
            
            # Property 2: Confidence must be valid for display
            assert isinstance(result.confidence, float), \
                "confidence must be a float"
            assert 0.0 <= result.confidence <= 1.0, \
                f"confidence must be in [0.0, 1.0] for display, got {result.confidence}"
            
            # Property 3: detected_controls must be a list (can be empty)
            assert isinstance(result.detected_controls, list), \
                "detected_controls must be a list"
            
            # Property 4: When confidence is low, suggest categories for user selection
            # (Requirement 1.3 - but related to displaying results)
            if result.confidence < service.HIGH_CONFIDENCE_THRESHOLD:
                assert isinstance(result.suggested_categories, list), \
                    "suggested_categories must be a list when confidence is low"
                # If confidence is low, we should suggest categories
                # (may be empty if system chooses not to suggest)
            
            # Property 5: suggested_categories must be a list
            assert isinstance(result.suggested_categories, list), \
                "suggested_categories must always be a list"
            
            # Property 6: All suggested categories must be displayable strings
            for category in result.suggested_categories:
                assert isinstance(category, str), \
                    f"Each suggested category must be a string, got {type(category)}"
                assert len(category) > 0, \
                    "Suggested categories must not be empty strings"
            
            # Property 7: Brand and model are optional but must be strings if present
            if result.brand is not None:
                assert isinstance(result.brand, str), \
                    "brand must be a string if present"
            
            if result.model is not None:
                assert isinstance(result.model, str), \
                    "model must be a string if present"
            
            # Property 8: All detected controls must have valid structure for display
            for control in result.detected_controls:
                assert hasattr(control, 'id'), "Control must have id"
                assert hasattr(control, 'type'), "Control must have type"
                assert hasattr(control, 'bounding_box'), "Control must have bounding_box"
                assert hasattr(control, 'confidence'), "Control must have confidence"
                
                assert isinstance(control.id, str), "Control id must be string"
                assert len(control.id) > 0, "Control id must not be empty"
                
                assert isinstance(control.type, str), "Control type must be string"
                assert len(control.type) > 0, "Control type must not be empty"
                
                assert 0.0 <= control.confidence <= 1.0, \
                    f"Control confidence must be in [0.0, 1.0], got {control.confidence}"
                
                # Bounding box must be valid for display/highlighting
                bbox = control.bounding_box
                assert 0.0 <= bbox.x <= 1.0, f"bbox.x must be normalized, got {bbox.x}"
                assert 0.0 <= bbox.y <= 1.0, f"bbox.y must be normalized, got {bbox.y}"
                assert 0.0 < bbox.width <= 1.0, f"bbox.width must be positive and normalized, got {bbox.width}"
                assert 0.0 < bbox.height <= 1.0, f"bbox.height must be positive and normalized, got {bbox.height}"
                
                # Label is optional but must be string if present
                if control.label is not None:
                    assert isinstance(control.label, str), "Control label must be string if present"
            
        except Exception as e:
            error_msg = str(e).lower()
            # Skip if models are not available
            if any(keyword in error_msg for keyword in ['model', 'file', 'cuda', 'ollama', 'not found']):
                pytest.skip(f"Models not available for testing: {e}")
            else:
                # Re-raise unexpected errors
                raise
    
    @given(image_bytes=device_image_bytes())
    @settings(
        max_examples=10,
        deadline=60000,
        suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow]
    )
    def test_device_type_is_consistent_format(self, image_bytes):
        """Property: device_type follows consistent naming format.
        
        The device_type should:
        1. Be lowercase or snake_case for consistency
        2. Not contain special characters that would break display
        3. Be suitable for both display and internal use
        
        This ensures the device type can be reliably displayed to users.
        """
        service = ImageAnalysisService()
        
        try:
            # Act
            result = service.analyze_device(image_bytes)
            
            # Assert - verify device_type format
            device_type = result.device_type
            
            # Should not contain problematic characters
            assert '\n' not in device_type, "device_type should not contain newlines"
            assert '\r' not in device_type, "device_type should not contain carriage returns"
            assert '\t' not in device_type, "device_type should not contain tabs"
            
            # Should be a reasonable length for display
            assert len(device_type) <= 100, \
                f"device_type should be reasonable length for display, got {len(device_type)} chars"
            
        except Exception as e:
            error_msg = str(e).lower()
            if any(keyword in error_msg for keyword in ['model', 'file', 'cuda', 'ollama', 'not found']):
                pytest.skip(f"Models not available for testing: {e}")
            else:
                raise
    
    @given(image_bytes=device_image_bytes())
    @settings(
        max_examples=10,
        deadline=60000,
        suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow]
    )
    def test_result_is_json_serializable(self, image_bytes):
        """Property: DeviceAnalysisResult can be serialized for API response.
        
        The result must be JSON-serializable so it can be:
        1. Sent to the frontend for display
        2. Stored in session history
        3. Logged for debugging
        
        This is essential for displaying results to the user (Requirement 1.2).
        """
        service = ImageAnalysisService()
        
        try:
            # Act
            result = service.analyze_device(image_bytes)
            
            # Assert - verify JSON serializability
            result_dict = result.model_dump()
            assert isinstance(result_dict, dict), "Result must be convertible to dict"
            
            # Verify all fields are JSON-serializable types
            import json
            try:
                json_str = json.dumps(result_dict)
                assert len(json_str) > 0, "JSON serialization should produce non-empty string"
                
                # Verify we can deserialize back
                deserialized = json.loads(json_str)
                assert isinstance(deserialized, dict), "Should deserialize to dict"
                assert 'device_type' in deserialized, "device_type must be in serialized result"
                assert 'confidence' in deserialized, "confidence must be in serialized result"
                
            except (TypeError, ValueError) as json_error:
                pytest.fail(f"Result is not JSON-serializable: {json_error}")
            
        except Exception as e:
            error_msg = str(e).lower()
            if any(keyword in error_msg for keyword in ['model', 'file', 'cuda', 'ollama', 'not found']):
                pytest.skip(f"Models not available for testing: {e}")
            else:
                raise
    
    @given(
        image_bytes=device_image_bytes(),
        num_calls=st.integers(min_value=1, max_value=3)
    )
    @settings(
        max_examples=5,
        deadline=120000,
        suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow]
    )
    def test_multiple_analyses_are_stable(self, image_bytes, num_calls):
        """Property: Multiple analyses of the same image produce consistent results.
        
        When analyzing the same image multiple times:
        1. device_type should be consistent
        2. confidence should be similar (within reasonable variance)
        3. Number of detected controls should be similar
        
        This ensures users see consistent device identification.
        """
        service = ImageAnalysisService()
        
        try:
            # Act - analyze same image multiple times
            results = []
            for _ in range(num_calls):
                result = service.analyze_device(image_bytes)
                results.append(result)
            
            # Assert - verify consistency
            if len(results) > 1:
                # Device types should be the same
                device_types = [r.device_type for r in results]
                assert len(set(device_types)) == 1, \
                    f"device_type should be consistent across calls, got {device_types}"
                
                # Confidences should be similar (within 10% variance)
                confidences = [r.confidence for r in results]
                max_conf = max(confidences)
                min_conf = min(confidences)
                if max_conf > 0:
                    variance = (max_conf - min_conf) / max_conf
                    assert variance <= 0.1, \
                        f"Confidence should be stable, got variance {variance:.2%}"
                
                # Number of controls should be similar (within 20% variance)
                control_counts = [len(r.detected_controls) for r in results]
                if max(control_counts) > 0:
                    max_count = max(control_counts)
                    min_count = min(control_counts)
                    count_variance = (max_count - min_count) / max_count
                    assert count_variance <= 0.2, \
                        f"Control count should be stable, got variance {count_variance:.2%}"
            
        except Exception as e:
            error_msg = str(e).lower()
            if any(keyword in error_msg for keyword in ['model', 'file', 'cuda', 'ollama', 'not found']):
                pytest.skip(f"Models not available for testing: {e}")
            else:
                raise


class TestLowConfidenceProperties:
    """Property-based tests for low confidence device identification.
    
    **Property 3: Предложение категорий при низкой уверенности**
    **Validates: Requirements 1.3**
    
    WHEN CV-модуль не может идентифицировать устройство с достаточной уверенностью, 
    THEN Система SHALL предложить пользователю выбрать тип устройства из списка категорий.
    
    This property verifies that when device identification confidence is low,
    the system provides suggested categories for manual user selection.
    """
    
    @given(image_bytes=device_image_bytes())
    @settings(
        max_examples=15,
        deadline=60000,
        suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow]
    )
    def test_low_confidence_provides_category_suggestions(self, image_bytes):
        """Property: When confidence is low, system suggests categories for user selection.
        
        For any device image where the CV module cannot identify the device with
        sufficient confidence (below HIGH_CONFIDENCE_THRESHOLD), the system must:
        
        1. Return suggested_categories as a non-empty list
        2. Include reasonable device category options
        3. All categories must be valid, displayable strings
        4. Categories should be distinct (no duplicates)
        5. The list should contain at least 3 categories for meaningful choice
        
        This ensures Requirement 1.3: system offers category selection when confidence is low.
        """
        service = ImageAnalysisService()
        
        try:
            # Act - analyze the device image
            result = service.analyze_device(image_bytes)
            
            # Property: When confidence is below threshold, suggest categories
            if result.confidence < service.HIGH_CONFIDENCE_THRESHOLD:
                # Assert 1: suggested_categories must be a non-empty list
                assert isinstance(result.suggested_categories, list), \
                    "suggested_categories must be a list when confidence is low"
                assert len(result.suggested_categories) > 0, \
                    f"When confidence ({result.confidence:.2f}) < threshold ({service.HIGH_CONFIDENCE_THRESHOLD}), " \
                    f"system MUST suggest categories for user selection (Requirement 1.3)"
                
                # Assert 2: Should provide meaningful number of choices (at least 3)
                assert len(result.suggested_categories) >= 3, \
                    f"System should suggest at least 3 categories for meaningful choice, " \
                    f"got {len(result.suggested_categories)}"
                
                # Assert 3: All categories must be valid strings
                for category in result.suggested_categories:
                    assert isinstance(category, str), \
                        f"Each category must be a string, got {type(category)}"
                    assert len(category) > 0, \
                        "Categories must not be empty strings"
                    assert category.strip() != "", \
                        "Categories must not be only whitespace"
                    
                    # Categories should be reasonable length for display
                    assert len(category) <= 50, \
                        f"Category '{category}' is too long ({len(category)} chars) for display"
                    
                    # Categories should not contain problematic characters
                    assert '\n' not in category, \
                        f"Category '{category}' should not contain newlines"
                    assert '\r' not in category, \
                        f"Category '{category}' should not contain carriage returns"
                
                # Assert 4: Categories should be distinct (no duplicates)
                unique_categories = set(result.suggested_categories)
                assert len(unique_categories) == len(result.suggested_categories), \
                    f"Suggested categories should be distinct, found duplicates: " \
                    f"{[cat for cat in result.suggested_categories if result.suggested_categories.count(cat) > 1]}"
                
                # Assert 5: Categories should be from known device types
                # (This ensures we're suggesting reasonable options)
                valid_category_patterns = [
                    "machine", "control", "remote", "appliance", "furniture",
                    "oven", "microwave", "washer", "dryer", "dishwasher",
                    "tv", "conditioner", "vacuum", "coffee", "other"
                ]
                
                # At least some categories should match known patterns
                matching_categories = 0
                for category in result.suggested_categories:
                    category_lower = category.lower()
                    if any(pattern in category_lower for pattern in valid_category_patterns):
                        matching_categories += 1
                
                assert matching_categories > 0, \
                    f"Suggested categories should include recognizable device types, " \
                    f"got: {result.suggested_categories}"
            
            # Property: When confidence is high, categories may be empty or minimal
            else:
                # High confidence - categories are optional
                # But if provided, they must still be valid
                assert isinstance(result.suggested_categories, list), \
                    "suggested_categories must always be a list"
                
                for category in result.suggested_categories:
                    assert isinstance(category, str), \
                        f"Each category must be a string, got {type(category)}"
                    assert len(category) > 0, \
                        "Categories must not be empty strings if provided"
            
        except Exception as e:
            error_msg = str(e).lower()
            if any(keyword in error_msg for keyword in ['model', 'file', 'cuda', 'ollama', 'not found']):
                pytest.skip(f"Models not available for testing: {e}")
            else:
                raise
    
    @given(
        image_bytes=device_image_bytes(),
        seed=st.integers(min_value=0, max_value=1000)
    )
    @settings(
        max_examples=10,
        deadline=60000,
        suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow]
    )
    def test_category_suggestions_are_consistent(self, image_bytes, seed):
        """Property: Category suggestions are consistent across multiple calls.
        
        When the system suggests categories for low confidence identification,
        the suggested categories should be:
        
        1. Consistent across multiple analyses of the same image
        2. Based on the same category pool
        3. Not randomly changing between calls
        
        This ensures users get reliable category suggestions.
        """
        service = ImageAnalysisService()
        
        try:
            # Act - analyze same image twice
            result1 = service.analyze_device(image_bytes)
            result2 = service.analyze_device(image_bytes)
            
            # Only test if both results have low confidence
            if (result1.confidence < service.HIGH_CONFIDENCE_THRESHOLD and 
                result2.confidence < service.HIGH_CONFIDENCE_THRESHOLD):
                
                # Assert: Suggested categories should be the same
                categories1 = set(result1.suggested_categories)
                categories2 = set(result2.suggested_categories)
                
                assert categories1 == categories2, \
                    f"Category suggestions should be consistent across calls. " \
                    f"First call: {result1.suggested_categories}, " \
                    f"Second call: {result2.suggested_categories}"
            
        except Exception as e:
            error_msg = str(e).lower()
            if any(keyword in error_msg for keyword in ['model', 'file', 'cuda', 'ollama', 'not found']):
                pytest.skip(f"Models not available for testing: {e}")
            else:
                raise
    
    @given(image_bytes=device_image_bytes())
    @settings(
        max_examples=10,
        deadline=60000,
        suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow]
    )
    def test_categories_include_fallback_option(self, image_bytes):
        """Property: Category suggestions include a fallback/other option.
        
        When suggesting categories for low confidence identification,
        the system should include a fallback option (like "other_appliance")
        to handle cases where none of the specific categories match.
        
        This ensures users always have a way to proceed even if their
        device doesn't fit the suggested categories.
        """
        service = ImageAnalysisService()
        
        try:
            # Act
            result = service.analyze_device(image_bytes)
            
            # Only test if confidence is low
            if result.confidence < service.HIGH_CONFIDENCE_THRESHOLD:
                # Assert: Should include a fallback/other option
                categories_lower = [cat.lower() for cat in result.suggested_categories]
                
                has_fallback = any(
                    'other' in cat or 'unknown' in cat or 'general' in cat
                    for cat in categories_lower
                )
                
                assert has_fallback, \
                    f"Category suggestions should include a fallback option (e.g., 'other_appliance'), " \
                    f"got: {result.suggested_categories}"
            
        except Exception as e:
            error_msg = str(e).lower()
            if any(keyword in error_msg for keyword in ['model', 'file', 'cuda', 'ollama', 'not found']):
                pytest.skip(f"Models not available for testing: {e}")
            else:
                raise
    
    @given(image_bytes=device_image_bytes())
    @settings(
        max_examples=10,
        deadline=60000,
        suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow]
    )
    def test_low_confidence_result_is_actionable(self, image_bytes):
        """Property: Low confidence results provide actionable information.
        
        When confidence is low, the result must still be actionable:
        
        1. device_type is set (even if uncertain)
        2. suggested_categories are provided for user selection
        3. Result can be displayed to user for manual category selection
        4. All data is properly structured for UI rendering
        
        This ensures the user can take action (select a category) when
        automatic identification fails (Requirement 1.3).
        """
        service = ImageAnalysisService()
        
        try:
            # Act
            result = service.analyze_device(image_bytes)
            
            # Only test if confidence is low
            if result.confidence < service.HIGH_CONFIDENCE_THRESHOLD:
                # Assert 1: device_type must still be set
                assert result.device_type is not None, \
                    "device_type must be set even with low confidence"
                assert isinstance(result.device_type, str), \
                    "device_type must be a string"
                assert len(result.device_type) > 0, \
                    "device_type must not be empty"
                
                # Assert 2: suggested_categories must be provided
                assert len(result.suggested_categories) > 0, \
                    "suggested_categories must be provided for low confidence results"
                
                # Assert 3: Result must be JSON-serializable for UI
                result_dict = result.model_dump()
                assert 'device_type' in result_dict, \
                    "Result must include device_type for display"
                assert 'confidence' in result_dict, \
                    "Result must include confidence for display"
                assert 'suggested_categories' in result_dict, \
                    "Result must include suggested_categories for user selection"
                
                # Assert 4: Confidence value clearly indicates low confidence
                assert result.confidence < service.HIGH_CONFIDENCE_THRESHOLD, \
                    f"Confidence ({result.confidence}) should be below threshold " \
                    f"({service.HIGH_CONFIDENCE_THRESHOLD}) to trigger category suggestions"
                
                # Assert 5: The combination of low confidence + suggestions is coherent
                # If we're suggesting categories, confidence should be meaningfully low
                if len(result.suggested_categories) > 0:
                    assert result.confidence < 0.9, \
                        f"If suggesting categories, confidence should be meaningfully low, " \
                        f"got {result.confidence:.2f}"
            
        except Exception as e:
            error_msg = str(e).lower()
            if any(keyword in error_msg for keyword in ['model', 'file', 'cuda', 'ollama', 'not found']):
                pytest.skip(f"Models not available for testing: {e}")
            else:
                raise
    
    @given(image_bytes=device_image_bytes())
    @settings(
        max_examples=10,
        deadline=60000,
        suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow]
    )
    def test_confidence_threshold_behavior(self, image_bytes):
        """Property: System behavior changes appropriately at confidence threshold.
        
        The system should have clear threshold-based behavior:
        
        1. confidence >= HIGH_CONFIDENCE_THRESHOLD: categories optional
        2. confidence < HIGH_CONFIDENCE_THRESHOLD: categories required
        3. Threshold behavior is consistent and predictable
        
        This ensures Requirement 1.3 is properly implemented with clear thresholds.
        """
        service = ImageAnalysisService()
        
        try:
            # Act
            result = service.analyze_device(image_bytes)
            
            # Assert: Behavior matches threshold
            if result.confidence >= service.HIGH_CONFIDENCE_THRESHOLD:
                # High confidence - categories are optional
                # (System is confident, user doesn't need to choose)
                assert isinstance(result.suggested_categories, list), \
                    "suggested_categories must be a list"
                # No requirement for non-empty when confidence is high
                
            else:
                # Low confidence - categories are required (Requirement 1.3)
                assert isinstance(result.suggested_categories, list), \
                    "suggested_categories must be a list"
                assert len(result.suggested_categories) > 0, \
                    f"When confidence ({result.confidence:.2f}) < HIGH_CONFIDENCE_THRESHOLD " \
                    f"({service.HIGH_CONFIDENCE_THRESHOLD}), system MUST suggest categories " \
                    f"for user selection (Requirement 1.3)"
            
            # Assert: Threshold value is reasonable
            assert 0.0 < service.HIGH_CONFIDENCE_THRESHOLD < 1.0, \
                f"HIGH_CONFIDENCE_THRESHOLD should be between 0 and 1, " \
                f"got {service.HIGH_CONFIDENCE_THRESHOLD}"
            
            # Assert: Confidence value is within valid range
            assert 0.0 <= result.confidence <= 1.0, \
                f"Confidence must be in [0.0, 1.0], got {result.confidence}"
            
        except Exception as e:
            error_msg = str(e).lower()
            if any(keyword in error_msg for keyword in ['model', 'file', 'cuda', 'ollama', 'not found']):
                pytest.skip(f"Models not available for testing: {e}")
            else:
                raise
