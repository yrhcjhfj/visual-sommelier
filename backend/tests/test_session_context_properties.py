"""Property-based tests for session context preservation.

**Property 8: Сохранение контекста сессии**
**Validates: Requirements 2.5**

These tests verify that the SessionService correctly maintains conversation
context within a session. The system SHALL preserve all messages in a session
and allow retrieval of the conversation history.
"""

import pytest
from hypothesis import given, strategies as st, settings, assume, HealthCheck
from hypothesis.strategies import composite

from backend.app.services.session_service import SessionService
from backend.app.models.session import Message, MessageRole
from backend.app.models.device import DeviceContext, Control, BoundingBox


@composite
def user_ids(draw):
    """Generate a user ID string."""
    return draw(st.text(min_size=1, max_size=20, alphabet=st.characters(whitelist_categories=('Lu', 'Ll'))))


@composite
def device_types(draw):
    """Generate a device type string."""
    types = ["washing_machine", "microwave", "oven", "dishwasher", "tv", "coffee_machine"]
    return draw(st.sampled_from(types))


@composite
def message_contents(draw):
    """Generate message content strings."""
    contents = [
        "What does this button do?",
        "How do I start the device?",
        "Please explain the control panel.",
        "Set the temperature to 60 degrees.",
        "Turn on the power.",
    ]
    return draw(st.sampled_from(contents))


class TestSessionContextProperties:
    """Property-based tests for session context preservation.

    **Property 8: Сохранение контекста сессии**
    **Validates: Requirements 2.5**

    The system SHALL preserve all messages within a session and allow
    retrieval of the conversation history in the correct order.
    """

    @pytest.fixture
    def service(self):
        """Create SessionService instance."""
        return SessionService()

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
                    confidence=0.9,
                )
            ],
            safety_warnings=[],
        )

    @given(
        num_messages=st.integers(min_value=1, max_value=10),
        user_id=user_ids(),
        device_type=device_types(),
    )
    @settings(
        max_examples=10,
        deadline=5000,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    def test_messages_preserved_in_order(self, service, device_context, num_messages, user_id, device_type):
        """Property: All messages are preserved in the correct order.

        For any session with N messages:
        1. All N messages should be retrievable
        2. Messages should be in the same order as they were added
        3. Message content should match exactly
        """
        assume(user_id.strip() != "")
        assume(device_type.strip() != "")

        # Create session
        session = service.create_session(
            user_id=user_id,
            device_type=device_type,
            device_image_url="/test/image.jpg",
            device_context=device_context,
        )

        # Add messages alternating between user and assistant
        added_contents = []
        for i in range(num_messages):
            role = MessageRole.USER if i % 2 == 0 else MessageRole.ASSISTANT
            content = f"Message {i + 1} from {role.value}"
            service.add_message(
                session_id=session.id,
                role=role,
                content=content,
            )
            added_contents.append((role, content))

        # Retrieve context
        context = service.get_session_context(session.id, last_n=num_messages + 5)

        # Property 1: All messages should be preserved
        assert len(context) == num_messages, (
            f"Expected {num_messages} messages, got {len(context)}"
        )

        # Property 2: Messages should be in the same order
        for i, message in enumerate(context):
            expected_role, expected_content = added_contents[i]
            assert message.role == expected_role, (
                f"Message {i}: expected role {expected_role}, got {message.role}"
            )
            assert message.content == expected_content, (
                f"Message {i}: expected content '{expected_content}', got '{message.content}'"
            )

    @given(
        messages_to_add=st.integers(min_value=5, max_value=15),
        messages_to_retrieve=st.integers(min_value=1, max_value=10),
    )
    @settings(
        max_examples=10,
        deadline=5000,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    def test_context_returns_last_n_messages(self, service, device_context, messages_to_add, messages_to_retrieve):
        """Property: get_session_context returns exactly the last N messages.

        For any session with M messages where M > N:
        1. get_session_context(last_n=N) should return exactly N messages
        2. Returned messages should be the most recent N messages
        """
        assume(messages_to_retrieve <= messages_to_add)

        session = service.create_session(
            user_id="test_user",
            device_type="washing_machine",
            device_image_url="/test/image.jpg",
            device_context=device_context,
        )

        # Add messages
        for i in range(messages_to_add):
            role = MessageRole.USER if i % 2 == 0 else MessageRole.ASSISTANT
            service.add_message(
                session_id=session.id,
                role=role,
                content=f"Content for message {i + 1}",
            )

        # Retrieve last N messages
        context = service.get_session_context(session.id, last_n=messages_to_retrieve)

        # Property: Should return exactly N messages (or all if N > total)
        expected_count = min(messages_to_retrieve, messages_to_add)
        assert len(context) == expected_count, (
            f"Expected {expected_count} messages, got {len(context)}"
        )

        # Property: Returned messages should be the most recent ones
        if context:
            last_message_index = messages_to_add - 1
            first_message_index = messages_to_add - len(context)
            for i, message in enumerate(context):
                expected_content = f"Content for message {first_message_index + i + 1}"
                assert message.content == expected_content, (
                    f"Message {i}: expected '{expected_content}', got '{message.content}'"
                )

    @given(
        num_sessions=st.integers(min_value=1, max_value=5),
        user_id=user_ids(),
    )
    @settings(
        max_examples=10,
        deadline=5000,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    def test_sessions_are_isolated(self, service, device_context, num_sessions, user_id):
        """Property: Different sessions maintain separate contexts.

        For any number of sessions:
        1. Each session should have its own message history
        2. Adding messages to one session should not affect others
        3. Each session should have a unique ID
        """
        assume(user_id.strip() != "")

        sessions = []
        for i in range(num_sessions):
            session = service.create_session(
                user_id=user_id,
                device_type="washing_machine",
                device_image_url=f"/test/image_{i}.jpg",
                device_context=device_context,
            )
            sessions.append(session)

            # Add unique messages to each session
            for j in range(i + 1):
                service.add_message(
                    session_id=session.id,
                    role=MessageRole.USER,
                    content=f"Session {i} message {j}",
                )

        # Property 1: Each session should have unique ID
        session_ids = [s.id for s in sessions]
        assert len(set(session_ids)) == num_sessions, "All session IDs should be unique"

        # Property 2: Each session should have correct number of messages
        for i, session in enumerate(sessions):
            context = service.get_session_context(session.id, last_n=100)
            expected_count = i + 1
            assert len(context) == expected_count, (
                f"Session {i} should have {expected_count} messages, got {len(context)}"
            )

    def test_create_session_sets_timestamps(self, service, device_context):
        """Property: Created sessions have valid timestamps.

        A new session should have:
        1. created_at timestamp set
        2. updated_at timestamp set
        3. created_at <= updated_at
        """
        session = service.create_session(
            user_id="test_user",
            device_type="washing_machine",
            device_image_url="/test/image.jpg",
            device_context=device_context,
        )

        assert session.created_at is not None, "created_at should be set"
        assert session.updated_at is not None, "updated_at should be set"
        assert session.created_at <= session.updated_at, "created_at should be <= updated_at"

    def test_add_message_updates_timestamp(self, service, device_context):
        """Property: Adding a message updates the session's updated_at timestamp.

        After adding a message:
        1. updated_at should be greater than or equal to the previous updated_at
        """
        session = service.create_session(
            user_id="test_user",
            device_type="washing_machine",
            device_image_url="/test/image.jpg",
            device_context=device_context,
        )

        initial_updated = session.updated_at

        service.add_message(
            session_id=session.id,
            role=MessageRole.USER,
            content="Test message",
        )

        updated_session = service.get_session(session.id)
        assert updated_session.updated_at >= initial_updated, (
            "updated_at should be >= after adding message"
        )

    def test_get_nonexistent_session_returns_none(self, service):
        """Property: Retrieving a non-existent session returns None."""
        result = service.get_session("nonexistent-id")
        assert result is None, "Non-existent session should return None"

    def test_add_message_to_nonexistent_session_raises(self, service):
        """Property: Adding message to non-existent session raises ValueError."""
        with pytest.raises(ValueError, match="not found"):
            service.add_message(
                session_id="nonexistent-id",
                role=MessageRole.USER,
                content="Test message",
            )

    def test_delete_session_removes_it(self, service, device_context):
        """Property: Deleting a session removes it from storage."""
        session = service.create_session(
            user_id="test_user",
            device_type="washing_machine",
            device_image_url="/test/image.jpg",
            device_context=device_context,
        )

        deleted = service.delete_session(session.id)
        assert deleted is True, "delete_session should return True"

        result = service.get_session(session.id)
        assert result is None, "Deleted session should not be retrievable"

    def test_delete_nonexistent_session_returns_false(self, service):
        """Property: Deleting a non-existent session returns False."""
        deleted = service.delete_session("nonexistent-id")
        assert deleted is False, "Deleting non-existent session should return False"
