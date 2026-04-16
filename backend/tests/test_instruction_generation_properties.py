"""Property-based tests for instruction generation in ExplanationService.

**Property 9: Генерация структурированных инструкций**

Validates: Requirements 3.1, 3.2
"""

import pytest
from hypothesis import given, strategies as st, settings, HealthCheck
import re

from app.services.explanation_service import ExplanationService
from app.models.device import DeviceContext
from app.models.explanation import Step

class TestInstructionGenerationProperties:
    @pytest.fixture
    def service(self):
        return ExplanationService()

    @given(
        task=st.text(min_size=10, max_size=100),
        llm_response=st.sampled_from([
            "Step 1: Plug it in.\nStep 2: Press start.",
            "1. Open the lid\n2. Fill with water\n3. Close lid",
            "First, do this. Second, do that.", # Non-standard but should be handled
        ]),
        device_type=st.sampled_from(["washing_machine", "microwave", "remote_control"]),
        language=st.sampled_from(["en", "ru", "zh"])
    )
    @settings(max_examples=10, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_property_9_structured_instructions(self, service, task, llm_response, device_type, language):
        """Property 9: Генерация структурированных инструкций."""
        context = DeviceContext(
            device_type=device_type,
            detected_controls=[],
            safety_warnings=[]
        )

        # Mock LLM adapter
        with pytest.MonkeyPatch.context() as mp:
            mock_adapter = type('MockAdapter', (), {
                'generate_completion': lambda *args, **kwargs: llm_response
            })
            mp.setattr(service, "_get_llm_adapter", lambda: mock_adapter)

            steps = service.generate_instructions(
                task=task,
                device_context=context,
                language=language
            )

            assert isinstance(steps, list)
            assert len(steps) > 0
            assert all(isinstance(s, Step) for s in steps)

            # Check step numbering
            for i, step in enumerate(steps):
                assert step.number > 0
                assert step.description
                assert not step.completed # Default state

    def test_safety_warnings_in_steps(self, service):
        """Verify that steps with dangerous keywords get specific warnings."""
        context = DeviceContext(
            device_type="microwave",
            detected_controls=[],
            safety_warnings=[]
        )

        # Text with mixed dangerous and safe steps
        llm_response = (
            "1. Unplug the device from the socket\n"
            "2. Open the back panel to repair the motor\n"
            "3. Clean the surface with a cloth"
        )

        with pytest.MonkeyPatch.context() as mp:
            mock_adapter = type('MockAdapter', (), {
                'generate_completion': lambda *args, **kwargs: llm_response
            })
            mp.setattr(service, "_get_llm_adapter", lambda: mock_adapter)

            steps = service.generate_instructions(
                task="Repair microwave",
                device_context=context
            )

            assert len(steps) == 3
            # Step 1 has 'unplug' -> Electrical warning
            assert "ELECTRICAL" in steps[0].warning or "unplug" in steps[0].warning.lower()
            # Step 2 has 'repair' -> Caution warning
            assert "CAUTION" in steps[1].warning or "care" in steps[1].warning.lower()
            # Step 3 is safe
            assert steps[2].warning is None
