"""
This module contains the BotRegistry class for managing the registration of different chatbot types.
"""

import importlib
from typing import Callable, Dict, List, Type

from chatbot_conversation.models.base import ChatbotBase
from chatbot_conversation.models.bot_config import BOT_MODULES
from chatbot_conversation.utils import get_logger

logger = get_logger("models")


class BotRegistry:
    """Registry for managing chatbot types and their corresponding classes.

    This class implements a singleton pattern to ensure that only one instance
    of the registry exists. It allows for the registration and retrieval of
    chatbot classes based on their type names.
    """

    _instance = None

    def __new__(cls) -> "BotRegistry":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        if not hasattr(self, "_initialized"):
            self._bot_classes: Dict[str, Type[ChatbotBase]] = {}
            self._initialized = True
            self.import_bot_modules()  # Import bot modules during initialization

    def register_bot(self, bot_type_name: str, bot_class: Type[ChatbotBase]) -> None:
        """Register a new bot type with its corresponding class.

        Args:
            bot_type_name (str): The name of the bot type.
            bot_class (Type[ChatbotBase]): The class implementing the bot type.
        """
        self._bot_classes[bot_type_name.upper()] = bot_class

    def get_bot_class(self, bot_type_name: str) -> Type[ChatbotBase]:
        """Get the class corresponding to the bot type.

        Args:
            bot_type_name (str): The name of the bot type.

        Returns:
            Type[ChatbotBase]: The class of the bot.

        Raises:
            ValueError: If the bot type is not recognized.
        """
        bot_class = self._bot_classes.get(bot_type_name.upper())
        if not bot_class:
            raise ValueError(f"Unknown bot type: {bot_type_name}")
        return bot_class

    def import_bot_modules(self) -> None:
        """Import bot modules specified in the BOT_MODULES configuration.

        This method dynamically imports the modules listed in the BOT_MODULES
        configuration to ensure that all bot types are registered.
        """
        for module_name in BOT_MODULES:
            logger.debug("Registering module: %s", module_name)
            importlib.import_module(module_name)

    def is_bot_registered(self, bot_type_name: str) -> bool:
        """
        Check if a bot type is registered in the registry.

        Args:
            bot_type_name (str): The name of the bot type.

        Returns:
            bool: True if the bot type is registered, False otherwise.
        """
        return bot_type_name.upper() in self._bot_classes

    def list_registered_bots(self) -> List[str]:
        """
        List all registered bot types in the registry.

        Returns:
            list: A list of registered bot type names.
        """
        return list(self._bot_classes.keys())


def register_bot(
    bot_type_name: str,
) -> Callable[[Type[ChatbotBase]], Type[ChatbotBase]]:
    """Decorator to register a chatbot class with the singleton bot registry.

    Args:
        bot_type_name (str): The name of the bot type.

    Returns:
        Callable: A decorator function that registers the class.
    """

    def decorator(cls: Type[ChatbotBase]) -> Type[ChatbotBase]:
        BotRegistry().register_bot(bot_type_name.upper(), cls)
        return cls

    return decorator
