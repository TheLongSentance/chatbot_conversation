"""
This module contains pytest fixtures for creating instances of various chatbot models.

Fixtures:
    gpt_chatbot: Creates an instance of GPTChatbot.
    claude_chatbot: Creates an instance of ClaudeChatbot.
    ollama_chatbot: Creates an instance of OllamaChatbot.
    gemini_chatbot: Creates an instance of GeminiChatbot.
    bot_registry: Creates an instance of BotRegistry and registers bot types.
    chatbot_factory: Creates an instance of ChatbotFactory.
"""

import pytest

from chatbot_conversation.models import (
    BotRegistry,
    ChatbotConfig,
    ChatbotFactory,
    ChatbotModel,
    ChatbotParamsOpt,
)
from chatbot_conversation.models.bots.claude_bot import ClaudeChatbot
from chatbot_conversation.models.bots.gemini_bot import GeminiChatbot
from chatbot_conversation.models.bots.gpt_bot import GPTChatbot
from chatbot_conversation.models.bots.ollama_bot import OllamaChatbot


@pytest.fixture
def gpt_chatbot() -> GPTChatbot:
    """Fixture to create an instance of GPTChatbot."""
    config = ChatbotConfig(
        name="GPTTestBot1",
        system_prompt="You are a helpful assistant.",
        model=ChatbotModel(
            type="GPT",
            version="gpt-4o-mini",
            params_opt=ChatbotParamsOpt(temperature=0.7),
        ),
    )
    return GPTChatbot(config)


@pytest.fixture
def claude_chatbot() -> ClaudeChatbot:
    """Fixture to create an instance of ClaudeChatbot."""
    config = ChatbotConfig(
        name="ClaudeTestBot1",
        system_prompt="You are a helpful assistant.",
        model=ChatbotModel(
            type="CLAUDE",
            version="claude-3-haiku-20240307",
            params_opt=ChatbotParamsOpt(temperature=0.7),
        ),
    )
    return ClaudeChatbot(config)


@pytest.fixture
def ollama_chatbot() -> OllamaChatbot:
    """Fixture to create an instance of OllamaChatbot."""
    config = ChatbotConfig(
        name="OllamaTestBot1",
        system_prompt="You are a helpful assistant.",
        model=ChatbotModel(
            type="OLLAMA",
            version="llama3.2",
            params_opt=ChatbotParamsOpt(temperature=0.7),
        ),
    )
    return OllamaChatbot(config)


@pytest.fixture
def gemini_chatbot() -> GeminiChatbot:
    """Fixture to create an instance of GeminiChatbot."""
    config = ChatbotConfig(
        name="GeminiTestBot1",
        system_prompt="You are a helpful assistant.",
        model=ChatbotModel(
            type="GEMINI",
            version="gemini-1.5-flash",
            params_opt=ChatbotParamsOpt(temperature=0.7),
        ),
    )
    return GeminiChatbot(config)


@pytest.fixture
def bot_registry() -> BotRegistry:
    """Fixture for creating a BotRegistry instance and registering bot types."""
    bot_registry = BotRegistry()

    return bot_registry


@pytest.fixture
def chatbot_factory(bot_registry: BotRegistry) -> ChatbotFactory:
    """Fixture to create an instance of ChatbotFactory."""
    return ChatbotFactory(bot_registry)
