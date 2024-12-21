"""
This module contains pytest fixtures for creating instances of various chatbot models.

Fixtures:
    openai_chatbot: Creates an instance of OpenAIChatbot.
    claude_chatbot: Creates an instance of ClaudeChatbot.
    ollama_chatbot: Creates an instance of OllamaChatbot.
    gemini_chatbot: Creates an instance of GeminiChatbot.
    chatbot_factory: Creates an instance of ChatbotFactory.
"""

import pytest

from src.models import ClaudeChatbot, GeminiChatbot, OllamaChatbot, OpenAIChatbot
from src.models.factory import ChatbotFactory


@pytest.fixture
def openai_chatbot() -> OpenAIChatbot:
    """Fixture to create an instance of OpenAIChatbot."""
    return OpenAIChatbot(
        bot_model_version="gpt-4o-mini",
        bot_specific_system_prompt="You are a helpful assistant.",
        bot_name="OpenAITestBot1",
        shared_system_prompt_prefix="You are in a test program and you are called {bot_name} - ",
    )


@pytest.fixture
def claude_chatbot() -> ClaudeChatbot:
    """Fixture to create an instance of ClaudeChatbot."""
    return ClaudeChatbot(
        bot_model_version="claude-3-haiku-20240307",
        bot_specific_system_prompt="You are a helpful assistant.",
        bot_name="ClaudeTestBot1",
        shared_system_prompt_prefix="You are in a test program and you are called {bot_name} - ",
    )


@pytest.fixture
def ollama_chatbot() -> OllamaChatbot:
    """Fixture to create an instance of OllamaChatbot."""
    return OllamaChatbot(
        bot_model_version="llama3.2",
        bot_specific_system_prompt="You are a helpful assistant.",
        bot_name="OllamaTestBot1",
        shared_system_prompt_prefix="You are in a test program and you are called {bot_name} - ",
    )


@pytest.fixture
def gemini_chatbot() -> GeminiChatbot:
    """Fixture to create an instance of GeminiChatbot."""
    return GeminiChatbot(
        bot_model_version="gemini-1.5-flash",
        bot_specific_system_prompt="You are a helpful assistant.",
        bot_name="GeminiTestBot1",
        shared_system_prompt_prefix="You are in a test program and you are called {bot_name} - ",
    )


@pytest.fixture
def chatbot_factory() -> ChatbotFactory:
    """Fixture to create an instance of ChatbotFactory."""
    return ChatbotFactory()
