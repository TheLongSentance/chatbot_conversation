"""
This module contains pytest fixtures for creating instances of various chatbot models.

Fixtures:
    dummy_chatbot: Creates an instance of dummy Chatbot as proxy for ChatbotBase.
    bot_registry: Creates an instance of BotRegistry and registers bot types.
    chatbot_factory: Creates an instance of ChatbotFactory.
"""

from typing import List

import pytest

from chatbot_conversation.models import ConversationMessage


@pytest.fixture
def basic_conversation() -> List[ConversationMessage]:
    """Basic conversation fixture for testing."""
    return [
        {"bot_index": 0, "content": "Hello"},
        {"bot_index": 1, "content": "Hi there"},
        {"bot_index": 0, "content": "How are you?"},
    ]
