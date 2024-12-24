"""
This module contains the ChatbotFactory class for creating different types of chatbots.
"""

from chatbot_conversation.models.base import BotType, ChatbotBase
from chatbot_conversation.models.bot_registry import BotRegistry


class ChatbotFactory:
    """Factory for creating different types of chatbots using dependency injection."""

    def __init__(self, bot_registry: BotRegistry):
        self._bot_registry = bot_registry

    def create_bot(
        self,
        bot_type: BotType,
        bot_model_version: str,
        bot_specific_system_prompt: str,
        bot_name: str,
        bot_shared_system_prompt_prefix: str,
    ) -> ChatbotBase:
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
        bot_class = self._bot_registry.get_bot_class(bot_type)
        return bot_class(
            bot_model_version,
            bot_specific_system_prompt,
            bot_name,
            bot_shared_system_prompt_prefix,
        )
