"""
This module contains the ChatbotFactory class for creating different types of chatbots.
"""

from typing import Any
from .base import BotType, ChatbotBase
from .openai_bot import OpenAIChatbot
from .claude_bot import ClaudeChatbot
from .gemini_bot import GeminiChatbot
from .ollama_bot import OllamaChatbot


class ChatbotFactory:
    """Factory for creating different types of chatbots."""

    def create_bot(
        self,
        bot_type: BotType,
        bot_model_version: str,
        bot_specific_system_prompt: str,
        bot_name: str,
        bot_shared_system_prompt_prefix: str,
    ) -> ChatbotBase[Any]:
        """Create a new chatbot instance based on type.

        Args:
            bot_type: Type of bot to create (GPT, CLAUDE, etc.)
            bot_model_version: Model version to use
            bot_specific_system_prompt: System instruction for bot behavior
            bot_name: Name of the bot
            bot_shared_system_prompt_prefix: Shared system prompt prefix for the bot

        Returns:
            ChatbotBase: Initialized chatbot instance

        Raises:
            ValueError: If bot_type is not recognized
        """
        if bot_type == BotType.GPT:
            return OpenAIChatbot(
                bot_model_version,
                bot_specific_system_prompt,
                bot_name,
                bot_shared_system_prompt_prefix,
            )
        elif bot_type == BotType.CLAUDE:
            return ClaudeChatbot(
                bot_model_version,
                bot_specific_system_prompt,
                bot_name,
                bot_shared_system_prompt_prefix,
            )
        elif bot_type == BotType.GEMINI:
            return GeminiChatbot(
                bot_model_version,
                bot_specific_system_prompt,
                bot_name,
                bot_shared_system_prompt_prefix,
            )
        elif bot_type == BotType.OLLAMA:
            return OllamaChatbot(
                bot_model_version,
                bot_specific_system_prompt,
                bot_name,
                bot_shared_system_prompt_prefix,
            )
        else:
            raise ValueError(f"Unknown bot type: {bot_type}")
