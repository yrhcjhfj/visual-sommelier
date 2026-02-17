"""Property-based tests for safety warnings in explanations.

**Property 23: Предупреждения о безопасности для опасных операций**
**Property 24: Предупреждения для электрических устройств**
**Property 27: Наличие дисклеймера**

Validates: Requirements 8.1, 8.2, 8.4, 8.5
"""

import pytest
from hypothesis import given, strategies as st, settings, HealthCheck
import re

from app.services.explanation_service import ExplanationService
from app.models.device import DeviceContext, BoundingBox

class TestExplanationSafetyProperties:
    @pytest.fixture
    def service(self):
        return ExplanationService()

    @given(
        device_type=st.sampled_from([
            "washing_machine", "microwave", "oven", "dishwasher",
            "coffee_machine", "toaster"
        ]),
        question=st.text(min_size=1)
    )
    @settings(max_examples=20, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_property_24_electrical_device_warning(self, service, device_type, question):
        """Property 24: Электрические устройства должны иметь предупреждение."""
        context = DeviceContext(
            device_type=device_type,
            detected_controls=[],
            safety_warnings=[]
        )

        # Проверяем внутренний метод детекции
        warnings = service._detect_safety_concerns(context, question)

        assert any("ELECTRICAL" in w for w in warnings), \
            f"Device {device_type} must trigger an electrical safety warning"

    @given(
        question=st.sampled_from([
            "How do I repair this?",
            "Can I disassemble the back panel?",
            "How to open the internal wiring?",
            "Is it safe to remove the heater?",
            "How to fix the motor myself?"
        ])
    )
    @settings(max_examples=10, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_property_23_dangerous_operation_warning(self, service, question):
        """Property 23: Опасные операции должны вызывать предупреждение."""
        context = DeviceContext(
            device_type="remote_control", # Обычно не опасен, но операция опасна
            detected_controls=[],
            safety_warnings=[]
        )

        warnings = service._detect_safety_concerns(context, question)

        assert any("CAUTION" in w or "professional" in w.lower() for w in warnings), \
            f"Dangerous question '{question}' must trigger a caution warning"

    @given(
        device_type=st.sampled_from(["washing_machine", "remote_control", "kettle"]),
        question=st.text(min_size=5, max_size=50),
        language=st.sampled_from(["en", "ru", "zh"])
    )
    @settings(max_examples=10, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_property_27_disclaimer_presence(self, service, device_type, question, language):
        """Property 27: Дисклеймер должен быть в любом ответе."""
        context = DeviceContext(
            device_type=device_type,
            detected_controls=[],
            safety_warnings=[]
        )

        # Используем mock для LLM, чтобы не тратить ресурсы
        with pytest.MonkeyPatch.context() as mp:
            mock_adapter = type('MockAdapter', (), {
                'generate_completion': lambda *args, **kwargs: "Test response content"
            })
            mp.setattr(service, "_get_llm_adapter", lambda: mock_adapter)

            explanation = service.generate_explanation(
                image=b"fake",
                question=question,
                device_context=context,
                language=language
            )

            assert "Disclaimer" in explanation.text or "⚠️" in explanation.text, \
                "Every explanation must contain a disclaimer"
