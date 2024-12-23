"""
This module contains the ChatbotFactory class for creating different types of chatbots.
"""

from chatbot_conversation.models.base import BotConfig, ChatbotBase
from chatbot_conversation.models.bot_registry import BotRegistry


class ChatbotFactory:
    """Factory for creating different types of chatbots using dependency injection."""

    def __init__(self, bot_registry: BotRegistry):
        self._bot_registry = bot_registry

    def create_bot(self, config: BotConfig) -> ChatbotBase:
        """Create a new chatbot instance based on configuration.

        Args:
            config: Bot configuration parameters

        Returns:
            ChatbotBase: Initialized chatbot instance

        Raises:
            ValueError: If bot_type is not recognized
        """
        bot_class = self._bot_registry.get_bot_class(config.bot_type)
        return bot_class(
            config.bot_model_version,
            config.bot_specific_system_prompt,
            config.bot_name,
            config.bot_shared_system_prompt_prefix,
        )
