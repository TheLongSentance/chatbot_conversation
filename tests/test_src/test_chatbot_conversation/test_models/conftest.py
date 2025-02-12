"""
This module contains pytest fixtures for creating instances of various chatbot models.

Fixtures:
    dummy_chatbot: Creates an instance of dummy Chatbot as proxy for ChatbotBase.
    bot_registry: Creates an instance of BotRegistry and registers bot types.
    chatbot_factory: Creates an instance of ChatbotFactory.
"""

from typing import List, Optional, Type

import pytest

from chatbot_conversation.models import (
    BotRegistry,
    ChatbotBase,
    ChatbotConfig,
    ChatbotFactory,
    ChatbotModel,
    ChatbotParamsOpt,
    ChatbotTimeout,
    ConversationMessage,
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
            version="tpg-o4-mini",
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
        def available_versions(cls) -> Optional[List[str]]:
            """Return list of valid versions for testing."""
            return ["1.0", "2.0", "3.0"]

        @classmethod
        def _get_class_model_type(cls) -> str:
            return "DUMMY"

        @classmethod
        def _should_retry_on_exception(cls, exception: BaseException) -> bool:
            return False

        def _generate_response(self, conversation: List[ConversationMessage]) -> str:
            return "dummy response"

        @property
        def _default_temperature(self) -> float:
            return 0.7

    return DummyBot
