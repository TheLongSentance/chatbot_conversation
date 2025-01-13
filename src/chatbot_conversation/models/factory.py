"""
This module contains the ChatbotFactory class for creating different types of chatbots.

Classes:
    ChatbotFactory: Factory for creating different types of chatbots using dependency injection.
"""

from typing import List

from chatbot_conversation.models.base import BotConfig, ChatbotBase
from chatbot_conversation.models.bot_registry import BotRegistry


class ChatbotFactory:
    """Factory for creating different types of chatbots using dependency injection."""

    def __init__(self, bot_registry: BotRegistry):
        """
        Initialize the factory with a bot registry.

        Args:
            bot_registry (BotRegistry): The registry containing bot classes.
        """
        self._bot_registry = bot_registry

    def create_bot(self, config: BotConfig) -> ChatbotBase:
        """
        Create a new chatbot instance based on configuration.

        Args:
            config (BotConfig): Bot configuration parameters.

        Returns:
            ChatbotBase: Initialized chatbot instance.

        Raises:
            ValueError: If bot_type is not recognized.
        """
        bot_class = self._bot_registry.get_bot_class(config.bot_type)
        return bot_class(
            bot_name = config.bot_name,
            bot_system_prompt = config.bot_system_prompt,
            bot_model_version = config.bot_version,
            bot_temp = config.bot_temp,
        )

    def list_available_bots(self) -> List[str]:
        """
        List all available bot types in the registry.

        Returns:
            list: A list of available bot type names.
        """
        return self._bot_registry.list_registered_bots()

    def is_bot_registered(self, bot_type_name: str) -> bool:
        """
        Check if a bot type is registered in the registry.

        Args:
            bot_type_name (str): The name of the bot type.

        Returns:
            bool: True if the bot type is registered, False otherwise.
        """
        return self._bot_registry.is_bot_registered(bot_type_name)
