"""
Singleton registry for managing chatbot type registration and instantiation.

This module implements a singleton registry that maintains mappings between
chatbot type names and their implementing classes. It provides automatic
discovery and registration of bot implementations through file system scanning.

Classes:
    BotRegistry: Singleton registry managing chatbot class registration and lookup.

Functions:
    register_bot: Decorator for registering chatbot implementations with the registry.
"""

import importlib
import os
from typing import Callable, Dict, List, Type

from chatbot_conversation.models.base import ChatbotBase
from chatbot_conversation.utils import (
    LOGNAME_MODELS,
    ErrorSeverity,
    ValidationException,
    get_logger,
)

# pylint: disable=duplicate-code

logger = get_logger(LOGNAME_MODELS)


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
            error_msg = f"Unknown bot type: {bot_type_name}"
            raise ValidationException(
                message=error_msg,
                user_message=f"{error_msg}, please check conversation configuration file",
                severity=ErrorSeverity.ERROR,
                original_error=None,
            )
        return bot_class

    def import_bot_modules(self) -> None:
        """Import all bot modules in the 'chatbot_conversation/models/bots' directory.

        This method dynamically imports all .py files in the 'chatbot_conversation/models/bots'
        directory to ensure that all bot types are registered.
        """
        bots_dir = os.path.join(os.path.dirname(__file__), "bots")
        for filename in os.listdir(bots_dir):
            if filename.endswith(".py") and filename != "__init__.py":
                module_name = f"chatbot_conversation.models.bots.{filename[:-3]}"
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
