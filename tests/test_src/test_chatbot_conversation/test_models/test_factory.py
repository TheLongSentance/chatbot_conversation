"""
Test suite for ChatbotFactory class.

Tests the factory's ability to create chatbots, list available types,
and validate bot registration status.
"""

import pytest

from chatbot_conversation.models.base import ChatbotBase, ChatbotConfig
from chatbot_conversation.models.bot_registry import BotRegistry
from chatbot_conversation.models.factory import ChatbotFactory
from chatbot_conversation.utils import ValidationException


def test_factory_initialization(bot_registry: BotRegistry) -> None:
    """
    Test ChatbotFactory initialization with registry.

    Args:
        bot_registry: Pytest fixture providing BotRegistry instance.
    """
    factory = ChatbotFactory(bot_registry)
    assert isinstance(factory, ChatbotFactory)


def test_create_bot_with_valid_config(
    chatbot_factory: ChatbotFactory, dummy_config: ChatbotConfig
) -> None:
    """
    Test successful bot creation with valid configuration.

    Args:
        chatbot_factory: Pytest fixture providing ChatbotFactory instance.
        dummy_config: Pytest fixture providing valid ChatbotConfig.
    """
    bot = chatbot_factory.create_bot(dummy_config)
    assert isinstance(bot, ChatbotBase)
    assert bot.name == dummy_config.name
    assert bot.model_type == dummy_config.model.type
    assert bot.model_version == dummy_config.model.version


def test_create_bot_with_invalid_type(
    chatbot_factory: ChatbotFactory, dummy_config: ChatbotConfig
) -> None:
    """
    Test bot creation fails with invalid bot type.

    Args:
        chatbot_factory: Pytest fixture providing ChatbotFactory instance.
        dummy_config: Pytest fixture providing ChatbotConfig to modify.
    """
    dummy_config.model.type = "NONEXISTENT_BOT_TYPE"
    with pytest.raises(ValidationException, match="Unknown bot type"):
        chatbot_factory.create_bot(dummy_config)


def test_list_available_bots(chatbot_factory: ChatbotFactory) -> None:
    """
    Test listing of available bot types.

    Args:
        chatbot_factory: Pytest fixture providing ChatbotFactory instance.
    """
    available_bots = chatbot_factory.list_available_bots()
    assert isinstance(available_bots, list)
    assert len(available_bots) > 0
    assert all(isinstance(bot_type, str) for bot_type in available_bots)


def test_is_bot_registered_with_valid_type(
    chatbot_factory: ChatbotFactory, dummy_config: ChatbotConfig
) -> None:
    """
    Test validation of registered bot type.

    Args:
        chatbot_factory: Pytest fixture providing ChatbotFactory instance.
        dummy_config: Pytest fixture providing ChatbotConfig with valid type.
    """
    assert chatbot_factory.is_bot_registered(dummy_config.model.type)


def test_is_bot_registered_with_invalid_type(chatbot_factory: ChatbotFactory) -> None:
    """
    Test validation of unregistered bot type.

    Args:
        chatbot_factory: Pytest fixture providing ChatbotFactory instance.
    """
    assert not chatbot_factory.is_bot_registered("NONEXISTENT_BOT_TYPE")
