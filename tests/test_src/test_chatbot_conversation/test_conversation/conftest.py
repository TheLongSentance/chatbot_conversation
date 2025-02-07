"""Test fixtures for chatbot conversation testing.

This module provides pytest fixtures used across multiple test modules,
including configuration data, mock objects, and manager instances.
"""

import os
from pathlib import Path
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
        List[ConversationMessage]: Sample conversation messages representing a complete conversation
    """
    return [
        ConversationMessage(bot_index=0, content="Test seed message"),
        ConversationMessage(bot_index=1, content="Bot1 response"),
        ConversationMessage(bot_index=2, content="Bot2 response"),
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
            ModeratorMessage(round_number=2, content="Final round", display_opt=False),
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
def env_transcript_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Set up and return a temporary transcript directory via environment variable."""
    env_dir = tmp_path / "env_transcripts"
    monkeypatch.setenv("BOTCONV_TRANSCRIPT_DIR", str(env_dir))
    return env_dir


@pytest.fixture
def mock_project_root(tmp_path: Path) -> Path:
    """Create a mock project root with pyproject.toml."""
    root = tmp_path / "project_root"
    root.mkdir()
    (root / "pyproject.toml").touch()
    return root


@pytest.fixture
def manager(test_config_path: str) -> ConversationManager:
    """Provide a ConversationManager instance for testing.

    Args:
        test_config_path: Path to test configuration file

    Returns:
        ConversationManager: Instance of ConversationManager
    """
    return ConversationManager(Path(test_config_path))


@pytest.fixture
def sample_private_messages() -> List[ConversationMessage]:
    """Return sample messages with private content for testing.

    Returns:
        List[ConversationMessage]: Sample messages with mixed public and private content
    """
    return [
        {"bot_index": 1, "content": "Public message from bot 1"},
        {"bot_index": 1, "content": "Public part PR1V4T3: Private part for bot 1"},
        {"bot_index": 2, "content": "Another public message PR1V4T3: Secret bot 2 stuff"},
        {"bot_index": 3, "content": "Just public content"},
    ]
