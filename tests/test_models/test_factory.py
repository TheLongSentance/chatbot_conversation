"""
This module contains tests for the ChatbotFactory class, ensuring that it correctly creates instances
of various chatbot types and handles unknown bot types appropriately.
"""

from chatbot_conversation.models import BotConfig, ChatbotFactory
from chatbot_conversation.models.claude_bot import ClaudeChatbot
from chatbot_conversation.models.gemini_bot import GeminiChatbot
from chatbot_conversation.models.ollama_bot import OllamaChatbot
from chatbot_conversation.models.openai_bot import OpenAIChatbot


def test_create_openai_chatbot(chatbot_factory: ChatbotFactory) -> None:
    """
    Test the creation of an OpenAIChatbot instance.
    """
    config = BotConfig(
        bot_type="GPT",
        bot_model_version="gpt-4o-mini",
        bot_system_prompt="You are a helpful assistant.",
        bot_name="OpenAITestBot1",
    )
    bot = chatbot_factory.create_bot(config)
    assert isinstance(bot, OpenAIChatbot)
    assert bot.name == "OpenAITestBot1"


def test_create_claude_chatbot(chatbot_factory: ChatbotFactory) -> None:
    """
    Test the creation of a ClaudeChatbot instance.
    """
    config = BotConfig(
        bot_type="CLAUDE",
        bot_model_version="claude-3-haiku-20240307",
        bot_system_prompt="You are a helpful assistant.",
        bot_name="ClaudeTestBot1",
    )
    bot = chatbot_factory.create_bot(config)
    assert isinstance(bot, ClaudeChatbot)
    assert bot.name == "ClaudeTestBot1"


def test_create_gemini_chatbot(chatbot_factory: ChatbotFactory) -> None:
    """
    Test the creation of a GeminiChatbot instance.
    """
    config = BotConfig(
        bot_type="GEMINI",
        bot_model_version="gemini-1.5-flash",
        bot_system_prompt="You are a helpful assistant.",
        bot_name="GeminiTestBot1",
    )
    bot = chatbot_factory.create_bot(config)
    assert isinstance(bot, GeminiChatbot)
    assert bot.name == "GeminiTestBot1"


def test_create_ollama_chatbot(chatbot_factory: ChatbotFactory) -> None:
    """
    Test the creation of an OllamaChatbot instance.
    """
    config = BotConfig(
        bot_type="OLLAMA",
        bot_model_version="llama3.2",
        bot_system_prompt="You are a helpful assistant.",
        bot_name="OllamaTestBot1",
    )
    bot = chatbot_factory.create_bot(config)
    assert isinstance(bot, OllamaChatbot)
    assert bot.name == "OllamaTestBot1"
