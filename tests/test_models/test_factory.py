"""
This module contains tests for the ChatbotFactory class, ensuring that it correctly creates instances
of various chatbot types and handles unknown bot types appropriately.
"""

from chatbot_conversation.models import (
    BotType,
    BotConfig,
    ChatbotFactory,
    ClaudeChatbot,
    GeminiChatbot,
    OllamaChatbot,
    OpenAIChatbot,
)
from chatbot_conversation.models.factory import BotConfig


def test_create_openai_chatbot(chatbot_factory: ChatbotFactory) -> None:
    """
    Test the creation of an OpenAIChatbot instance.
    """
    config = BotConfig(
        bot_type=BotType.GPT,
        bot_model_version="gpt-4o-mini",
        bot_specific_system_prompt="You are a helpful assistant.",
        bot_name="OpenAITestBot1",
        bot_shared_system_prompt_prefix="You are in a test program and you are called {bot_name} - ",
    )
    bot = chatbot_factory.create_bot(config)
    assert isinstance(bot, OpenAIChatbot)
    assert bot.name == "OpenAITestBot1"


def test_create_claude_chatbot(chatbot_factory: ChatbotFactory) -> None:
    """
    Test the creation of a ClaudeChatbot instance.
    """
    config = BotConfig(
        bot_type=BotType.CLAUDE,
        bot_model_version="claude-3-haiku-20240307",
        bot_specific_system_prompt="You are a helpful assistant.",
        bot_name="ClaudeTestBot1",
        bot_shared_system_prompt_prefix="You are in a test program and you are called {bot_name} - ",
    )
    bot = chatbot_factory.create_bot(config)
    assert isinstance(bot, ClaudeChatbot)
    assert bot.name == "ClaudeTestBot1"


def test_create_gemini_chatbot(chatbot_factory: ChatbotFactory) -> None:
    """
    Test the creation of a GeminiChatbot instance.
    """
    config = BotConfig(
        bot_type=BotType.GEMINI,
        bot_model_version="gemini-1.5-flash",
        bot_specific_system_prompt="You are a helpful assistant.",
        bot_name="GeminiTestBot1",
        bot_shared_system_prompt_prefix="You are in a test program and you are called {bot_name} - ",
    )
    bot = chatbot_factory.create_bot(config)
    assert isinstance(bot, GeminiChatbot)
    assert bot.name == "GeminiTestBot1"


def test_create_ollama_chatbot(chatbot_factory: ChatbotFactory) -> None:
    """
    Test the creation of an OllamaChatbot instance.
    """
    config = BotConfig(
        bot_type=BotType.OLLAMA,
        bot_model_version="llama3.2",
        bot_specific_system_prompt="You are a helpful assistant.",
        bot_name="OllamaTestBot1",
        bot_shared_system_prompt_prefix="You are in a test program and you are called {bot_name} - ",
    )
    bot = chatbot_factory.create_bot(config)
    assert isinstance(bot, OllamaChatbot)
    assert bot.name == "OllamaTestBot1"
