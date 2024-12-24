"""
This module contains the BotRegistry class for managing the registration of different chatbot types.
"""

from typing import Type, Dict

from chatbot_conversation.models.base import BotType, ChatbotBase


class BotRegistry:
    """Registry for managing chatbot types and their corresponding classes."""

    def __init__(self) -> None:
        self._bot_classes: Dict[BotType, Type[ChatbotBase]] = {}

    def register_bot(self, bot_type: BotType, bot_class: Type[ChatbotBase]) -> None:
        """Register a new bot type with its corresponding class.

        Args:
            bot_type (BotType): The type of the bot.
            bot_class (Type[ChatbotBase]): The class of the bot.
        """
        self._bot_classes[bot_type] = bot_class

    def get_bot_class(self, bot_type: BotType) -> Type[ChatbotBase]:
        """Get the class corresponding to the bot type.

        Args:
            bot_type (BotType): The type of the bot.

        Returns:
            Type[ChatbotBase]: The class of the bot.

        Raises:
            ValueError: If bot_type is not recognized
        """
        bot_class = self._bot_classes.get(bot_type)
        if not bot_class:
            raise ValueError(f"Unknown bot type: {bot_type}")
        return bot_class
