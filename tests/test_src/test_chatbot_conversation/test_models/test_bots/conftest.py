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

from chatbot_conversation.models import (  # BotRegistry,; ChatbotFactory,
    ChatbotBase,
    ChatbotConfig,
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
def gpt_config_for_tests() -> ChatbotConfig:
    """Basic config fixture for GPT-specific tests"""
    return ChatbotConfig(
        name="TestGPTBot",
        system_prompt="You are a test assistant.",
        model=ChatbotModel(type="GPT", version="gpt-4o-mini"),
    )


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
def claude_config_for_tests() -> ChatbotConfig:
    """Basic config fixture for Claude-specific tests"""
    return ChatbotConfig(
        name="TestClaudeBot",
        system_prompt="You are a test assistant.",
        model=ChatbotModel(type="CLAUDE", version="claude-3-haiku-20240307"),
    )


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
def ollama_config_for_tests() -> ChatbotConfig:
    """Basic config fixture for Ollama-specific tests"""
    return ChatbotConfig(
        name="TestOllamaBot",
        system_prompt="You are a test assistant.",
        model=ChatbotModel(type="OLLAMA", version="llama2"),
    )


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
def gemini_config_for_tests() -> ChatbotConfig:
    """Basic config fixture for Gemini-specific tests"""
    return ChatbotConfig(
        name="TestGeminiBot",
        system_prompt="You are a test assistant.",
        model=ChatbotModel(type="GEMINI", version="gemini-1.5-flash"),
    )


@pytest.fixture
def real_bot_classes() -> list[type[ChatbotBase]]:
    """Fixture providing all concrete bot classes for testing."""
    return [GPTChatbot, ClaudeChatbot, OllamaChatbot, GeminiChatbot]


@pytest.fixture
def mock_api_error() -> Exception:
    """Generic API error for testing retries."""
    return Exception("API Error")
