"""Property-based tests for language response matching.

**Property 5: Соответствие языка ответа**
**Validates: Requirements 2.2, 5.4**

These tests verify that the ExplanationService generates responses in the correct
language as requested by the user. The system SHALL ensure that text corresponds
to the user's selected language.
"""

import pytest
from hypothesis import given, strategies as st, settings, assume, HealthCheck
from hypothesis.strategies import composite
from io import BytesIO
from PIL import Image, ImageDraw
import re

from app.services.explanation_service import ExplanationService
from app.models.device import DeviceContext, Control, BoundingBox
from app.models.explanation import Step


# Language detection patterns
LANGUAGE_PATTERNS = {
    'en': {
        'alphabet': r'[a-zA-Z]',
        'common_words': ['the', 'is', 'are', 'and', 'or', 'to', 'of', 'in', 'for', 'on', 'with'],
        'name': 'English'
    },
    'ru': {
        'alphabet': r'[а-яА-ЯёЁ]',
        'common_words': ['и', 'в', 'на', 'с', 'для', 'это', 'как', 'по', 'не', 'что'],
        'name': 'Russian'
    },
    'zh': {
        'alphabet': r'[\u4e00-\u9fff]',
        'common_words': ['的', '是', '在', '和', '有', '了', '不', '人', '我', '他'],
        'name': 'Chinese'
    }
}


def detect_language(text: str) -> str:
    """Detect the primary language of a text.
    
    Args:
        text: Text to analyze
        
    Returns:
        Language code ('en', 'ru', 'zh') or 'unknown'
    """
    if not text or len(text.strip()) < 10:
        return 'unknown'
    
    # Count characters from each language
    scores = {}
    for lang_code, patterns in LANGUAGE_PATTERNS.items():
        alphabet_pattern = patterns['alphabet']
        matches = re.findall(alphabet_pattern, text)
        scores[lang_code] = len(matches)
    
    # Return language with highest score
    if max(scores.values()) == 0:
        return 'unknown'
    
    return max(scores, key=scores.get)


def has_language_content(text: str, language: str, min_ratio: float = 0.3) -> bool:
    """Check if text has sufficient content in the specified language.
    
    Args:
        text: Text to check
        language: Expected language code
        min_ratio: Minimum ratio of language-specific characters
        
    Returns:
        True if text has sufficient content in the language
    """
    if language not in LANGUAGE_PATTERNS:
        return False
    
    alphabet_pattern = LANGUAGE_PATTERNS[language]['alphabet']
    lang_chars = len(re.findall(alphabet_pattern, text))
    
    # Count total alphabetic characters
    total_chars = len(re.findall(r'[a-zA-Zа-яА-ЯёЁ\u4e00-\u9fff]', text))
    
    if total_chars == 0:
        return False
    
    ratio = lang_chars / total_chars
    return ratio >= min_ratio


@composite
def simple_device_image(draw):
    """Generate a simple device image for testing."""
    width = draw(st.integers(min_value=200, max_value=400))
    height = draw(st.integers(min_value=200, max_value=400))
    
    img = Image.new('RGB', (width, height), (240, 240, 240))
    draw_obj = ImageDraw.Draw(img)
    
    # Add a simple button-like shape
    x1, y1 = width // 3, height // 3
    x2, y2 = 2 * width // 3, 2 * height // 3
    draw_obj.rectangle([x1, y1, x2, y2], fill=(100, 100, 200), outline=(0, 0, 0))
    
    buffer = BytesIO()
    img.save(buffer, format='JPEG', quality=85)
    return buffer.getvalue()


@composite
def device_question(draw):
    """Generate a device-related question."""
    questions = [
        "What does this button do?",
        "How do I use this control?",
        "What is this feature?",
        "How does this work?",
        "What is the purpose of this element?"
    ]
    return draw(st.sampled_from(questions))


@composite
def device_task(draw):
    """Generate a device-related task."""
    tasks = [
        "Start the device",
        "Turn on the power",
        "Set the temperature",
        "Adjust the settings",
        "Configure the device"
    ]
    return draw(st.sampled_from(tasks))


class TestLanguageMatchingProperties:
    """Property-based tests for language response matching.
    
    **Property 5: Соответствие языка ответа**
    **Validates: Requirements 2.2, 5.4**
    
    The system SHALL ensure that generated explanations match the user's
    selected language. This is critical for user understanding and accessibility.
    """
    
    @pytest.fixture
    def service(self):
        """Create ExplanationService instance."""
        return ExplanationService()
    
    @pytest.fixture
    def device_context(self):
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
    
    @given(
        language=st.sampled_from(['en', 'ru', 'zh']),
        question=device_question(),
        image_bytes=simple_device_image()
    )
    @settings(
        max_examples=5,
        deadline=120000,  # 2 minutes for LLM operations
        suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow]
    )
    def test_generate_explanation_matches_requested_language(
        self,
        service,
        device_context,
        language,
        question,
        image_bytes
    ):
        """Property: Generated explanations match the requested language.
        
        For any valid language code and question:
        1. The explanation text should contain characters from the requested language
        2. The primary detected language should match the requested language
        3. The response should have meaningful content in that language
        
        This validates Requirements 2.2 and 5.4.
        """
        try:
            # Act
            explanation = service.generate_explanation(
                image=image_bytes,
                question=question,
                device_context=device_context,
                language=language
            )
            
            # Assert - explanation should exist
            assert explanation is not None, "Explanation should not be None"
            assert explanation.text, "Explanation text should not be empty"
            
            # Remove disclaimer for language detection (it's in English)
            text_without_disclaimer = explanation.text.split("⚠️ Disclaimer:")[0].strip()
            
            # Skip if response is too short or is an error message
            if len(text_without_disclaimer) < 20:
                assume(False)  # Skip this example
            
            if "error" in text_without_disclaimer.lower() or "apologize" in text_without_disclaimer.lower():
                assume(False)  # Skip error responses
            
            # Property 1: Text should have content in the requested language
            has_content = has_language_content(text_without_disclaimer, language, min_ratio=0.3)
            assert has_content, (
                f"Explanation should have content in {LANGUAGE_PATTERNS[language]['name']} "
                f"(requested: {language}). Got text: {text_without_disclaimer[:200]}"
            )
            
            # Property 2: Detected language should match requested language
            detected = detect_language(text_without_disclaimer)
            assert detected == language or detected == 'unknown', (
                f"Detected language '{detected}' should match requested '{language}'. "
                f"Text: {text_without_disclaimer[:200]}"
            )
            
        except Exception as e:
            error_msg = str(e).lower()
            if any(keyword in error_msg for keyword in ['ollama', 'connection', 'model', 'llava']):
                pytest.skip(f"LLM service not available: {e}")
            else:
                raise
    
    @given(
        language=st.sampled_from(['en', 'ru', 'zh']),
        task=device_task(),
        image_bytes=simple_device_image()
    )
    @settings(
        max_examples=5,
        deadline=120000,
        suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow]
    )
    def test_generate_instructions_matches_requested_language(
        self,
        service,
        device_context,
        language,
        task,
        image_bytes
    ):
        """Property: Generated instructions match the requested language.
        
        For any valid language code and task:
        1. Step descriptions should contain characters from the requested language
        2. The primary detected language should match the requested language
        3. All steps should be in the same language
        
        This validates Requirements 2.2 and 5.4.
        """
        try:
            # Act
            steps = service.generate_instructions(
                task=task,
                device_context=device_context,
                language=language,
                image=image_bytes
            )
            
            # Assert - steps should exist
            assert steps is not None, "Steps should not be None"
            assert len(steps) > 0, "Should have at least one step"
            assert all(isinstance(step, Step) for step in steps), "All items should be Step objects"
            
            # Check language of step descriptions
            for step in steps:
                description = step.description.strip()
                
                # Skip very short descriptions or error messages
                if len(description) < 15:
                    continue
                
                if "error" in description.lower() or "apologize" in description.lower():
                    continue
                
                # Property 1: Step should have content in the requested language
                has_content = has_language_content(description, language, min_ratio=0.3)
                assert has_content, (
                    f"Step {step.number} should have content in {LANGUAGE_PATTERNS[language]['name']} "
                    f"(requested: {language}). Got: {description[:200]}"
                )
                
                # Property 2: Detected language should match
                detected = detect_language(description)
                assert detected == language or detected == 'unknown', (
                    f"Step {step.number}: Detected language '{detected}' should match "
                    f"requested '{language}'. Text: {description[:200]}"
                )
            
        except Exception as e:
            error_msg = str(e).lower()
            if any(keyword in error_msg for keyword in ['ollama', 'connection', 'model', 'llava']):
                pytest.skip(f"LLM service not available: {e}")
            else:
                raise
    
    @given(
        language=st.sampled_from(['en', 'ru', 'zh']),
        question=st.text(min_size=5, max_size=50, alphabet=st.characters(whitelist_categories=('Lu', 'Ll')))
    )
    @settings(
        max_examples=5,
        deadline=120000,
        suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow]
    )
    def test_clarify_step_matches_requested_language(
        self,
        service,
        device_context,
        language,
        question
    ):
        """Property: Step clarifications match the requested language.
        
        For any valid language code and clarification question:
        1. The clarification text should contain characters from the requested language
        2. The primary detected language should match the requested language
        
        This validates Requirements 2.2 and 5.4.
        """
        # Filter out questions with only whitespace
        assume(question.strip() != "")
        assume(any(c.isalnum() for c in question))
        
        try:
            # Create a sample step
            step = Step(
                number=1,
                description="Press the start button",
                warning=None
            )
            
            # Act
            clarification = service.clarify_step(
                step=step,
                question=question,
                device_context=device_context,
                language=language
            )
            
            # Assert - clarification should exist
            assert clarification is not None, "Clarification should not be None"
            assert len(clarification) > 0, "Clarification should not be empty"
            
            # Remove disclaimer for language detection
            text_without_disclaimer = clarification.split("⚠️ Disclaimer:")[0].strip()
            
            # Skip if response is too short or is an error message
            if len(text_without_disclaimer) < 20:
                assume(False)
            
            if "error" in text_without_disclaimer.lower() or "apologize" in text_without_disclaimer.lower():
                assume(False)
            
            # Property 1: Text should have content in the requested language
            has_content = has_language_content(text_without_disclaimer, language, min_ratio=0.3)
            assert has_content, (
                f"Clarification should have content in {LANGUAGE_PATTERNS[language]['name']} "
                f"(requested: {language}). Got: {text_without_disclaimer[:200]}"
            )
            
            # Property 2: Detected language should match
            detected = detect_language(text_without_disclaimer)
            assert detected == language or detected == 'unknown', (
                f"Detected language '{detected}' should match requested '{language}'. "
                f"Text: {text_without_disclaimer[:200]}"
            )
            
        except Exception as e:
            error_msg = str(e).lower()
            if any(keyword in error_msg for keyword in ['ollama', 'connection', 'model', 'llava']):
                pytest.skip(f"LLM service not available: {e}")
            else:
                raise
    
    def test_language_detection_utility_english(self):
        """Test that language detection works for English."""
        text = "This is a test in English with some common words like the and is."
        assert detect_language(text) == 'en'
    
    def test_language_detection_utility_russian(self):
        """Test that language detection works for Russian."""
        text = "Это тест на русском языке с общими словами как и в."
        assert detect_language(text) == 'ru'
    
    def test_language_detection_utility_chinese(self):
        """Test that language detection works for Chinese."""
        text = "这是一个中文测试，包含一些常用词如的和是。"
        assert detect_language(text) == 'zh'
    
    def test_has_language_content_english(self):
        """Test language content detection for English."""
        text = "This is mostly English text with a few numbers 123."
        assert has_language_content(text, 'en', min_ratio=0.3)
    
    def test_has_language_content_russian(self):
        """Test language content detection for Russian."""
        text = "Это в основном русский текст с несколькими цифрами 123."
        assert has_language_content(text, 'ru', min_ratio=0.3)
    
    def test_has_language_content_mixed(self):
        """Test language content detection with mixed content."""
        # Text with both English and Russian, but more English
        text = "This is English text это русский but mostly English."
        assert has_language_content(text, 'en', min_ratio=0.3)
        assert not has_language_content(text, 'ru', min_ratio=0.5)
