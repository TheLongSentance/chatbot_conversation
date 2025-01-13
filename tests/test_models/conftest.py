"""
This module contains pytest fixtures for creating instances of various chatbot models.

Fixtures:
    openai_chatbot: Creates an instance of OpenAIChatbot.
    claude_chatbot: Creates an instance of ClaudeChatbot.
    ollama_chatbot: Creates an instance of OllamaChatbot.
    gemini_chatbot: Creates an instance of GeminiChatbot.
    bot_registry: Creates an instance of BotRegistry and registers bot types.
    chatbot_factory: Creates an instance of ChatbotFactory.
"""

import pytest

from chatbot_conversation.models import BotRegistry, ChatbotFactory
from chatbot_conversation.models.bots.claude_bot import ClaudeChatbot
from chatbot_conversation.models.bots.gemini_bot import GeminiChatbot
from chatbot_conversation.models.bots.ollama_bot import OllamaChatbot
from chatbot_conversation.models.bots.openai_bot import OpenAIChatbot


@pytest.fixture
def openai_chatbot() -> OpenAIChatbot:
    """Fixture to create an instance of OpenAIChatbot."""
    return OpenAIChatbot(
        bot_name="OpenAITestBot1",
        bot_system_prompt="You are a helpful assistant.",
        bot_model_version="gpt-4o-mini",
        bot_temp=0.7,
    )


@pytest.fixture
def claude_chatbot() -> ClaudeChatbot:
    """Fixture to create an instance of ClaudeChatbot."""
    return ClaudeChatbot(
        bot_name="ClaudeTestBot1",
        bot_system_prompt="You are a helpful assistant.",
        bot_model_version="claude-3-haiku-20240307",
        bot_temp=0.7,
    )


@pytest.fixture
def ollama_chatbot() -> OllamaChatbot:
    """Fixture to create an instance of OllamaChatbot."""
    return OllamaChatbot(
        bot_name="OllamaTestBot1",
        bot_system_prompt="You are a helpful assistant.",
        bot_model_version="llama3.2",
        bot_temp=0.7,
    )


@pytest.fixture
def gemini_chatbot() -> GeminiChatbot:
    """Fixture to create an instance of GeminiChatbot."""
    return GeminiChatbot(
        bot_name="GeminiTestBot1",
        bot_model_version="gemini-1.5-flash",
        bot_system_prompt="You are a helpful assistant.",
        bot_temp=0.7,
    )


@pytest.fixture
def bot_registry() -> BotRegistry:
    """Fixture for creating a BotRegistry instance and registering bot types."""
    bot_registry = BotRegistry()

    return bot_registry


@pytest.fixture
def chatbot_factory(bot_registry: BotRegistry) -> ChatbotFactory:
    """Fixture to create an instance of ChatbotFactory."""
    return ChatbotFactory(bot_registry)
