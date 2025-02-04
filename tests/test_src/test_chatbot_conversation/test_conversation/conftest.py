"""Test fixtures for chatbot conversation testing.

This module provides pytest fixtures used across multiple test modules,
including configuration data, mock objects, and manager instances.
"""

import os
from typing import List
from unittest.mock import Mock

import pytest

from chatbot_conversation.conversation import (
    ChatbotConfigData,
    ChatbotParamsOptData,
    ConversationConfig,
    ModeratorMessage,
)
from chatbot_conversation.conversation.manager import ConversationManager
from chatbot_conversation.models import ChatbotBase, ConversationMessage


@pytest.fixture
def test_config_path() -> str:
    """Provide path to test configuration file.

    Returns:
        str: Path to test configuration file
    """
    return os.path.join(os.path.dirname(__file__), "../../../config/test_config.json")


@pytest.fixture
def test_config_empty_path() -> str:
    """Provide path to empty test configuration file.

    Returns:
        str: Path to empty test configuration file
    """
    return os.path.join(
        os.path.dirname(__file__), "../../../config/test_config_empty.json"
    )


@pytest.fixture
def invalid_config_path() -> str:
    """Provide path to nonexistent configuration file.

    Returns:
        str: Path to nonexistent configuration file
    """
    return "nonexistent_config.json"


@pytest.fixture
def mock_bot() -> ChatbotBase:
    """Create a mock chatbot for testing.

    Returns:
        ChatbotBase: Mock chatbot instance
    """
    mock = Mock(spec=ChatbotBase)
    mock.name = "TestBot"
    mock.bot_index = 1
    mock.system_prompt = "Initial prompt"
    mock.model_max_tokens = 100
    return mock


@pytest.fixture
def sample_conversation_data() -> List[ConversationMessage]:
    """Return sample conversation data for testing.

    Returns:
        List[ConversationMessage]: Sample conversation messages
    """
    return [
        {"bot_index": 0, "content": "Test seed message"},
        {"bot_index": 1, "content": "Bot1 response"},
        {"bot_index": 2, "content": "Bot2 response"},
    ]


@pytest.fixture
def sample_conversation_config() -> ConversationConfig:
    """Provide a valid ConversationConfig with sample data.

    Returns:
        ConversationConfig: Sample configuration for testing
    """
    return ConversationConfig(
        author="Test Author",
        conversation_seed="Test seed",
        rounds=2,
        core_prompt="You are {bot_name}. Respond within {max_tokens} tokens.",
        moderator_messages_opt=[
            ModeratorMessage(round_number=1, content="Welcome to round 1"),
            ModeratorMessage(round_number=2, content="Final round"),
        ],
        bots=[
            ChatbotConfigData(
                bot_name="Bot1",
                bot_prompt="You are Bot1, an example bot.",
                bot_type="DUMMY",
                bot_version="None",
                bot_params_opt=ChatbotParamsOptData(
                    temperature=0.7,
                    max_tokens=100,
                ),
            ),
            ChatbotConfigData(
                bot_name="Bot2",
                bot_prompt="You are Bot2, an example bot.",
                bot_type="DUMMY",
                bot_version="None",
                bot_params_opt=ChatbotParamsOptData(
                    temperature=0.9,
                    max_tokens=200,
                ),
            ),
        ],
    )


@pytest.fixture
def manager(test_config_path: str) -> ConversationManager:
    """Provide a ConversationManager instance for testing.

    Args:
        test_config_path: Path to test configuration file

    Returns:
        ConversationManager: Instance of ConversationManager
    """
    return ConversationManager(test_config_path)
