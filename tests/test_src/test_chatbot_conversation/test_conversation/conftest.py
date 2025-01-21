import os
from unittest.mock import Mock

import pytest

from chatbot_conversation.models import (
    ChatbotBase,
    ConversationMessage,
)


@pytest.fixture
def test_config_path() -> str:
    return os.path.join(os.path.dirname(__file__), "../../../config/test_config.json")


@pytest.fixture
def test_config_empty_path() -> str:
    return os.path.join(
        os.path.dirname(__file__), "../../../config/test_config_empty.json"
    )


@pytest.fixture
def invalid_config_path() -> str:
    return "nonexistent_config.json"


@pytest.fixture
def mock_bot() -> ChatbotBase:
    """Create a mock chatbot for testing."""
    mock = Mock(spec=ChatbotBase)
    mock.name = "TestBot"
    mock.bot_index = 1
    mock.system_prompt = "Initial prompt"
    return mock


@pytest.fixture
def sample_conversation_data() -> list[ConversationMessage]:
    """Return sample conversation data for testing."""
    return [
        {"bot_index": 0, "content": "Test seed message"},
        {"bot_index": 1, "content": "Bot 1 response"},
        {"bot_index": 2, "content": "Bot 2 response"},
    ]
