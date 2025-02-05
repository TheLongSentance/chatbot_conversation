"""
Test suite for BotRegistry class.

Tests singleton behavior, bot registration, class retrieval, and module importing functionality.
"""

from typing import Type

import pytest

from chatbot_conversation.models.base import ChatbotBase
from chatbot_conversation.models.bot_registry import BotRegistry, register_bot
from chatbot_conversation.utils import ValidationException


def test_singleton_behavior() -> None:
    """Test that BotRegistry maintains singleton behavior."""
    registry1 = BotRegistry()
    registry2 = BotRegistry()
    assert registry1 is registry2


def test_register_and_get_bot(dummy_bot_class: Type[ChatbotBase]) -> None:
    """
    Test registering and retrieving a bot class.

    Args:
        dummy_bot_class: Pytest fixture providing a dummy bot class.
    """
    registry = BotRegistry()
    registry.register_bot("TEST_BOT", dummy_bot_class)
    retrieved_class = registry.get_bot_class("TEST_BOT")
    assert retrieved_class is dummy_bot_class


def test_register_bot_decorator(dummy_bot_class: Type[ChatbotBase]) -> None:
    """
    Test the register_bot decorator functionality.

    Args:
        dummy_bot_class: Pytest fixture providing a dummy bot class.
    """
    decorated_class = register_bot("TEST_BOT")(dummy_bot_class)
    registry = BotRegistry()
    assert registry.get_bot_class("TEST_BOT") is decorated_class


def test_get_nonexistent_bot() -> None:
    """Test getting a non-registered bot class raises ValueError."""
    registry = BotRegistry()
    with pytest.raises(ValidationException, match="Unknown bot type: NONEXISTENT_BOT"):
        registry.get_bot_class("NONEXISTENT_BOT")


def test_list_registered_bots(dummy_bot_class: Type[ChatbotBase]) -> None:
    """
    Test listing registered bot types.

    Args:
        dummy_bot_class: Pytest fixture providing a dummy bot class.
    """
    registry = BotRegistry()
    registry.register_bot("TEST_BOT", dummy_bot_class)
    bot_list = registry.list_registered_bots()
    assert "TEST_BOT" in bot_list
    assert isinstance(bot_list, list)
    assert all(isinstance(bot_type, str) for bot_type in bot_list)


def test_is_bot_registered(dummy_bot_class: Type[ChatbotBase]) -> None:
    """
    Test checking if a bot type is registered.

    Args:
        dummy_bot_class: Pytest fixture providing a dummy bot class.
    """
    registry = BotRegistry()
    registry.register_bot("TEST_BOT", dummy_bot_class)
    assert registry.is_bot_registered("TEST_BOT")
    assert not registry.is_bot_registered("NONEXISTENT_BOT")


def test_case_insensitive_registration(dummy_bot_class: Type[ChatbotBase]) -> None:
    """
    Test that bot registration and retrieval is case insensitive.

    Args:
        dummy_bot_class: Pytest fixture providing a dummy bot class.
    """
    registry = BotRegistry()
    registry.register_bot("test_bot", dummy_bot_class)
    assert registry.get_bot_class("TEST_BOT") is dummy_bot_class
    assert registry.get_bot_class("test_bot") is dummy_bot_class
