"""
This package contains various chatbot models and a factory for creating them.

Modules:
    base: Contains base classes and types for chatbots.
    claude_bot: Contains the ClaudeChatbot class.
    factory: Contains the ChatbotFactory class for creating chatbot instances.
    gemini_bot: Contains the GeminiChatbot class.
    ollama_bot: Contains the OllamaChatbot class.
    openai_bot: Contains the OpenAIChatbot class.

Classes:
    ChatbotBase: Base class for all chatbots.
    ConversationMessage: Represents a conversation message.
    OpenAIChatbot: Chatbot class for OpenAI models.
    ClaudeChatbot: Chatbot class for Claude models.
    GeminiChatbot: Chatbot class for Gemini models.
    OllamaChatbot: Chatbot class for Ollama models.
    ChatbotFactory: Factory class for creating chatbot instances.
    BotType: Enum for different types of chatbots.
"""

from .base import BotType, ChatbotBase, ConversationMessage
from .claude_bot import ClaudeChatbot
from .factory import ChatbotFactory
from .gemini_bot import GeminiChatbot
from .ollama_bot import OllamaChatbot
from .openai_bot import OpenAIChatbot

__all__ = [
    "ChatbotBase",
    "ConversationMessage",
    "OpenAIChatbot",
    "ClaudeChatbot",
    "GeminiChatbot",
    "OllamaChatbot",
    "ChatbotFactory",
    "BotType",
]
