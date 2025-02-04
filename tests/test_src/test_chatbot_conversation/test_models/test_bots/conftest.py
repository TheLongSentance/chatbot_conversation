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
)
from chatbot_conversation.models.bots.claude_bot import ClaudeChatbot
from chatbot_conversation.models.bots.dummy_bot import DummyChatbot
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
    # Get first available version from API
    config = ChatbotConfig(
        name="OllamaTestBot1",
        system_prompt="You are a helpful assistant.",
        model=ChatbotModel(
            type="OLLAMA",
            version="llama3.2",
        ),
    )
    return OllamaChatbot(config)


@pytest.fixture
def ollama_config_for_tests() -> ChatbotConfig:
    """Basic config fixture for Ollama-specific tests"""
    # Get first available version from API
    return ChatbotConfig(
        name="TestOllamaBot",
        system_prompt="You are a test assistant.",
        model=ChatbotModel(type="OLLAMA", version="llama3.2"),
    )


@pytest.fixture
def gemini_chatbot() -> GeminiChatbot:
    """Fixture to create an instance of GeminiChatbot."""
    # Get first available version from API
    config = ChatbotConfig(
        name="GeminiTestBot1",
        system_prompt="You are a helpful assistant.",
        model=ChatbotModel(
            type="GEMINI",
            version="gemini-1.5-flash",
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
def dummy_chatbot() -> DummyChatbot:
    """Fixture to create an instance of GeminiChatbot."""
    config = ChatbotConfig(
        name="DummyTestBot1",
        system_prompt="You are a helpful assistant.",
        model=ChatbotModel(
            type="DUMMY",
            version="None",
        ),
    )
    return DummyChatbot(config)


@pytest.fixture
def dummy_config_for_tests() -> ChatbotConfig:
    """Basic config fixture for Gemini-specific tests"""
    return ChatbotConfig(
        name="TestDummyBot",
        system_prompt="You are a test assistant.",
        model=ChatbotModel(type="DUMMY", version="None"),
    )


@pytest.fixture
def real_bot_classes() -> list[type[ChatbotBase]]:
    """Fixture providing all concrete bot classes for testing."""
    return [GPTChatbot, ClaudeChatbot, OllamaChatbot, GeminiChatbot]


@pytest.fixture
def mock_api_error() -> Exception:
    """Generic API error for testing retries."""
    return Exception("API Error")


@pytest.fixture(autouse=True)
def clear_version_cache():
    """Clear the version cache before each test"""
    ClaudeChatbot._available_versions_cache = None # pyright: ignore[reportPrivateUsage]
    GPTChatbot._available_versions_cache = None # pyright: ignore[reportPrivateUsage]
    yield
    ClaudeChatbot._available_versions_cache = None # pyright: ignore[reportPrivateUsage]
    GPTChatbot._available_versions_cache = None # pyright: ignore[reportPrivateUsage]
