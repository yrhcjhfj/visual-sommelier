"""Session service for managing user sessions and conversation history."""

import logging
import uuid
from datetime import datetime
from typing import Dict, List, Optional

from ..models.session import Message, MessageRole, Session
from ..models.device import DeviceContext

logger = logging.getLogger(__name__)


class SessionService:
    """Service for managing user sessions and conversation context.
    
    Implements Requirements 2.5, 6.1.
    """

    def __init__(self) -> None:
        """Initialize the session service with in-memory storage."""
        self._sessions: Dict[str, Session] = {}
        self._max_history_length: int = 50

    def create_session(
        self,
        user_id: str,
        device_type: str,
        device_image_url: str,
        device_context: Optional[DeviceContext] = None,
    ) -> Session:
        """Create a new session for a user and device.
        
        Args:
            user_id: User identifier
            device_type: Type of device being analyzed
            device_image_url: URL or path to the device image
            device_context: Optional device context information
            
        Returns:
            Newly created Session object
        """
        session_id = str(uuid.uuid4())
        now = datetime.utcnow()

        session = Session(
            id=session_id,
            user_id=user_id,
            device_type=device_type,
            device_image_url=device_image_url,
            messages=[],
            created_at=now,
            updated_at=now,
            device_context=device_context,
        )

        self._sessions[session_id] = session
        logger.info(f"Created session {session_id} for user {user_id}")

        return session

    def get_session(self, session_id: str) -> Optional[Session]:
        """Retrieve a session by its ID.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Session object or None if not found
        """
        return self._sessions.get(session_id)

    def add_message(
        self,
        session_id: str,
        role: MessageRole,
        content: str,
        image_ref: Optional[str] = None,
    ) -> Message:
        """Add a message to a session's conversation history.
        
        Args:
            session_id: Session identifier
            role: Role of the message sender (user, assistant, system)
            content: Message content
            image_ref: Optional reference to an attached image
            
        Returns:
            The added Message object
            
        Raises:
            ValueError: If session not found
        """
        session = self._sessions.get(session_id)
        if session is None:
            raise ValueError(f"Session {session_id} not found")

        message = Message(
            role=role,
            content=content,
            timestamp=datetime.utcnow(),
            image_ref=image_ref,
        )

        session.messages.append(message)
        session.updated_at = datetime.utcnow()

        # Trim history if exceeds max length
        if len(session.messages) > self._max_history_length:
            session.messages = session.messages[-self._max_history_length :]

        logger.info(f"Added {role.value} message to session {session_id}")

        return message

    def get_session_context(self, session_id: str, last_n: int = 10) -> List[Message]:
        """Get recent conversation context for a session.
        
        Args:
            session_id: Session identifier
            last_n: Number of recent messages to return
            
        Returns:
            List of recent messages
            
        Raises:
            ValueError: If session not found
        """
        session = self._sessions.get(session_id)
        if session is None:
            raise ValueError(f"Session {session_id} not found")

        return session.messages[-last_n:]

    def get_user_sessions(self, user_id: str) -> List[Session]:
        """Get all sessions for a specific user.
        
        Args:
            user_id: User identifier
            
        Returns:
            List of sessions for the user, sorted by most recent first
        """
        user_sessions = [
            session for session in self._sessions.values() if session.user_id == user_id
        ]
        return sorted(user_sessions, key=lambda s: s.updated_at, reverse=True)

    def delete_session(self, session_id: str) -> bool:
        """Delete a session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            True if deleted, False if session not found
        """
        if session_id in self._sessions:
            del self._sessions[session_id]
            logger.info(f"Deleted session {session_id}")
            return True
        return False

    def update_device_context(
        self, session_id: str, device_context: DeviceContext
    ) -> Session:
        """Update the device context for a session.
        
        Args:
            session_id: Session identifier
            device_context: New device context
            
        Returns:
            Updated Session object
            
        Raises:
            ValueError: If session not found
        """
        session = self._sessions.get(session_id)
        if session is None:
            raise ValueError(f"Session {session_id} not found")

        session.device_context = device_context
        session.updated_at = datetime.utcnow()

        logger.info(f"Updated device context for session {session_id}")

        return session
