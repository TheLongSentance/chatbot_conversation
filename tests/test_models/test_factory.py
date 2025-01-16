"""
This module contains tests for the ChatbotFactory class, ensuring that it correctly creates instances
of various chatbot types and handles unknown bot types appropriately.
"""

from chatbot_conversation.models import (
    ChatbotConfig,
    ChatbotFactory,
    ChatbotModel,
    ChatbotParamsOpt,
)
from chatbot_conversation.models.bots.claude_bot import ClaudeChatbot
from chatbot_conversation.models.bots.gemini_bot import GeminiChatbot
from chatbot_conversation.models.bots.gpt_bot import GPTChatbot
from chatbot_conversation.models.bots.ollama_bot import OllamaChatbot


def test_create_gpt_chatbot(chatbot_factory: ChatbotFactory) -> None:
    """Test the creation of an GPTChatbot instance."""
    config: ChatbotConfig = ChatbotConfig(
        name="GPTTestBot1",
        system_prompt="You are a helpful assistant.",
        model=ChatbotModel(
            type="GPT",
            version="gpt-4o-mini",
            params_opt=ChatbotParamsOpt(temperature=0.7),
        ),
    )
    bot = chatbot_factory.create_bot(config)
    assert isinstance(bot, GPTChatbot)
    assert bot.name == "GPTTestBot1"


def test_create_claude_chatbot(chatbot_factory: ChatbotFactory) -> None:
    """Test the creation of a ClaudeChatbot instance."""
    config = ChatbotConfig(
        name="ClaudeTestBot1",
        system_prompt="You are a helpful assistant.",
        model=ChatbotModel(
            type="CLAUDE",
            version="claude-3-haiku-20240307",
            params_opt=ChatbotParamsOpt(temperature=0.7),
        ),
    )
    bot = chatbot_factory.create_bot(config)
    assert isinstance(bot, ClaudeChatbot)
    assert bot.name == "ClaudeTestBot1"


def test_create_gemini_chatbot(chatbot_factory: ChatbotFactory) -> None:
    """Test the creation of a GeminiChatbot instance."""
    config = ChatbotConfig(
        name="GeminiTestBot1",
        system_prompt="You are a helpful assistant.",
        model=ChatbotModel(
            type="GEMINI",
            version="gemini-1.5-flash",
            params_opt=ChatbotParamsOpt(temperature=0.7),
        ),
    )
    bot = chatbot_factory.create_bot(config)
    assert isinstance(bot, GeminiChatbot)
    assert bot.name == "GeminiTestBot1"


def test_create_ollama_chatbot(chatbot_factory: ChatbotFactory) -> None:
    """Test the creation of a OllamaChatbot instance."""
    config = ChatbotConfig(
        name="OllamaTestBot1",
        system_prompt="You are a helpful assistant.",
        model=ChatbotModel(
            type="OLLAMA",
            version="llama3.2",
            params_opt=ChatbotParamsOpt(temperature=0.7),
        ),
    )
    bot = chatbot_factory.create_bot(config)
    assert isinstance(bot, OllamaChatbot)
    assert bot.name == "OllamaTestBot1"
