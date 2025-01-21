"""
This module contains pytest fixtures for creating instances of various chatbot models.

Fixtures:
    dummy_chatbot: Creates an instance of dummy Chatbot as proxy for ChatbotBase.
    bot_registry: Creates an instance of BotRegistry and registers bot types.
    chatbot_factory: Creates an instance of ChatbotFactory.
"""

from typing import List, Type

import pytest

from chatbot_conversation.models import (
    ConversationMessage, 
    ChatbotConfig, 
    ChatbotModel, 
    ChatbotParamsOpt, 
    ChatbotTimeout, 
    ChatbotBase,
    BotRegistry,
    ChatbotFactory,
)


@pytest.fixture
def basic_conversation() -> List[ConversationMessage]:
    """Basic conversation fixture for testing."""
    return [
        {"bot_index": 0, "content": "Hello"},
        {"bot_index": 1, "content": "Hi there"},
        {"bot_index": 0, "content": "How are you?"},
    ]

@pytest.fixture
def dummy_config() -> ChatbotConfig:
    """Create a dummy ChatbotConfig for testing."""
    return ChatbotConfig(
        name="test_bot",
        system_prompt="You are a test bot",
        model=ChatbotModel(
            type="DUMMY",  # This should match your dummy bot implementation
            version="None",
            params_opt=ChatbotParamsOpt(temperature=0.7, max_tokens=100),
        ),
        timeout=ChatbotTimeout(),
    )

@pytest.fixture
def bot_registry() -> BotRegistry:
    """Create a BotRegistry instance for testing."""
    return BotRegistry()

@pytest.fixture
def chatbot_factory(bot_registry: BotRegistry) -> ChatbotFactory:
    """Create a ChatbotFactory instance for testing."""
    return ChatbotFactory(bot_registry)

@pytest.fixture
def dummy_bot_class() -> Type[ChatbotBase]:
    """Create a dummy ChatbotBase implementation for testing."""
    
    class DummyBot(ChatbotBase):
        """Minimal ChatbotBase implementation for testing."""
        
        @classmethod
        def _get_class_model_type(cls) -> str:
            return "DUMMY"

        def _should_retry_on_exception(self, exception: Exception) -> bool:
            return False

        def _generate_response(self, conversation: List[ConversationMessage]) -> str:
            return "dummy response"

        @property
        def _default_temperature(self) -> float:
            return 0.7

    return DummyBot
