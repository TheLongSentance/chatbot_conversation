"""Test fixtures for chatbot conversation tests."""

from typing import Generator
from unittest.mock import Mock, patch

import pytest

# ...existing code...

@pytest.fixture
def mock_conversation_manager() -> Generator[Mock, None, None]:
    """Create a mock for the ConversationManager class.

    This fixture provides a mock object for testing the conversation manager
    initialization and execution without actual file operations.

    Yields:
        Generator[Mock, None, None]: A mock object representing the ConversationManager class.

    Example:
        def test_example(mock_conversation_manager):
            manager_instance = mock_conversation_manager.return_value
            # Use manager_instance in tests
    """
    with patch('chatbot_conversation.main.ConversationManager') as mock:
        yield mock


@pytest.fixture
def mock_api_config() -> Generator[Mock, None, None]:
    """Create a mock for the APIConfig class.

    This fixture provides a mock object for testing API configuration
    without actual environment variable operations.

    Yields:
        Generator[Mock, None, None]: A mock object representing the APIConfig class.

    Example:
        def test_example(mock_api_config):
            mock_api_config.setup_env.assert_called_once()
    """
    with patch('chatbot_conversation.main.APIConfig') as mock:
        yield mock
