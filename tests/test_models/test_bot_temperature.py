"""Unit tests for temperature parameter handling in different chatbot implementations.

Tests cover:
- Default temperature values
- Valid temperature ranges
- Invalid temperature handling
- Temperature bounds for each bot type
"""

from typing import Type

import pytest

from chatbot_conversation.models.base import ChatbotBase
from chatbot_conversation.models.bots.claude_bot import ClaudeChatbot
from chatbot_conversation.models.bots.gemini_bot import GeminiChatbot
from chatbot_conversation.models.bots.ollama_bot import OllamaChatbot
from chatbot_conversation.models.bots.openai_bot import OpenAIChatbot

ASSISTANT_TEST_PROMPT = "You are a helpful assistant."


@pytest.fixture(autouse=True)
def reset_bot_count() -> None:
    """Reset the bot counter before each test."""
    from chatbot_conversation.models.base import ChatbotBase

    ChatbotBase.reset_total_count()


def test_default_temperatures() -> None:
    """Verify default temperature values for each chatbot implementation.

    Tests:
        - Claude default temp = 1.0
        - Gemini default temp = 1.0
        - Ollama default temp = 0.8
        - OpenAI default temp = 1.0
    """
    bots = {
        "claude": ClaudeChatbot("test_claude", ASSISTANT_TEST_PROMPT, "claude-3-opus"),
        "gemini": GeminiChatbot("test_gemini", ASSISTANT_TEST_PROMPT, "gemini-1.5-pro"),
        "ollama": OllamaChatbot("test_ollama", ASSISTANT_TEST_PROMPT, "llama2"),
        "openai": OpenAIChatbot("test_openai", ASSISTANT_TEST_PROMPT, "gpt-4"),
    }

    expected_defaults = {
        "claude": 1.0,
        "gemini": 1.0,
        "ollama": 0.8,
        "openai": 1.0,
    }

    for bot_name, bot in bots.items():
        assert (
            bot.temp == expected_defaults[bot_name]
        ), f"Default temperature mismatch for {bot_name}"


@pytest.mark.parametrize(
    "bot_class,temp,model_version",
    [
        (ClaudeChatbot, 0.5, "claude-3-opus"),
        (ClaudeChatbot, 1.5, "claude-3-opus"),
        (GeminiChatbot, 0.5, "gemini-1.5-pro"),
        (GeminiChatbot, 1.5, "gemini-1.5-pro"),
        (OllamaChatbot, 0.25, "llama2"),
        (OllamaChatbot, 0.75, "llama2"),
        (OpenAIChatbot, 0.5, "gpt-4"),
        (OpenAIChatbot, 1.5, "gpt-4"),
    ],
)
def test_valid_temperature_values(
    bot_class: Type[ChatbotBase], temp: float, model_version: str
) -> None:
    """Verify that valid temperature values are properly set for each bot type.

    Args:
        bot_class: The chatbot class to test
        temp: Temperature value within valid range
        model_version: Model identifier string

    Tests that the temperature property matches the input value when within valid range.
    """
    bot = bot_class("test_bot", ASSISTANT_TEST_PROMPT, model_version, bot_temp=temp)
    assert bot.temp == temp, f"Temperature not set correctly for {bot_class.__name__}"


@pytest.mark.parametrize(
    "bot_class,temp,model_version",
    [
        (ClaudeChatbot, -0.1, "claude-3-opus"),
        (GeminiChatbot, 2.1, "gemini-1.5-pro"),
        (OllamaChatbot, 1.1, "llama2"),
        (OpenAIChatbot, -1.0, "gpt-4"),
    ],
)
def test_invalid_temperature_values(
    bot_class: Type[ChatbotBase], temp: float, model_version: str
) -> None:
    """Verify that invalid temperature values raise appropriate exceptions.

    Args:
        bot_class: The chatbot class to test
        temp: Temperature value outside valid range
        model_version: Model identifier string

    Tests that ValueError is raised when temperature is outside the valid range.
    """
    with pytest.raises(ValueError):
        _ = bot_class("test_bot", ASSISTANT_TEST_PROMPT, model_version, bot_temp=temp)


@pytest.mark.parametrize(
    "bot_class,model_version,min_temp,max_temp",
    [
        (ClaudeChatbot, "claude-3-opus", 0.0, 2.0),
        (GeminiChatbot, "gemini-1.5-pro", 0.0, 2.0),
        (OllamaChatbot, "llama2", 0.0, 1.0),
        (OpenAIChatbot, "gpt-4", 0.0, 2.0),
    ],
)
def test_temperature_bounds(
    bot_class: Type[ChatbotBase], model_version: str, min_temp: float, max_temp: float
) -> None:
    """Verify that temperature bounds are correctly enforced for each bot type.

    Args:
        bot_class: The chatbot class to test
        model_version: Model identifier string
        min_temp: Minimum allowed temperature value
        max_temp: Maximum allowed temperature value

    Tests:
        - Minimum temperature value is accepted
        - Maximum temperature value is accepted
    """
    # Test minimum temperature
    bot = bot_class("test_bot", ASSISTANT_TEST_PROMPT, model_version, bot_temp=min_temp)
    assert bot.temp == min_temp

    # Test maximum temperature
    bot = bot_class("test_bot", ASSISTANT_TEST_PROMPT, model_version, bot_temp=max_temp)
    assert bot.temp == max_temp
