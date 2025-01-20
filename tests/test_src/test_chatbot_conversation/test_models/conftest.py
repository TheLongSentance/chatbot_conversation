"""
This module contains pytest fixtures for creating instances of various chatbot models.

Fixtures:
    dummy_chatbot: Creates an instance of dummy Chatbot as proxy for ChatbotBase.
    bot_registry: Creates an instance of BotRegistry and registers bot types.
    chatbot_factory: Creates an instance of ChatbotFactory.
"""

import pytest

from chatbot_conversation.models import (  # BotRegistry,; ChatbotFactory,; ChatbotBase,
    ChatbotConfig,
    ChatbotModel,
    ChatbotParamsOpt,
)
from chatbot_conversation.models.bots.dummy_bot import DummyChatbot


@pytest.fixture
def dummy_chatbot() -> DummyChatbot:
    """Fixture to create an instance of GeminiChatbot."""
    config = ChatbotConfig(
        name="DummyTestBot",
        system_prompt="You are a helpful assistant.",
        model=ChatbotModel(
            type="DUMMY",
            version="None",
            params_opt=ChatbotParamsOpt(temperature=0.7, max_tokens=100),
        ),
    )
    return DummyChatbot(config)
