import os
from typing import List
from unittest.mock import Mock

import pytest

from chatbot_conversation.conversation import (
    ChatbotConfigData,
    ChatbotParamsOptData,
    ConversationConfig,
)
from chatbot_conversation.conversation.prompt import SuffixManager
from chatbot_conversation.models import ChatbotBase, ConversationMessage


@pytest.fixture
def test_config_path() -> str:
    return os.path.join(os.path.dirname(__file__), "../../../config/test_config.json")


@pytest.fixture
def test_config_empty_path() -> str:
    return os.path.join(
        os.path.dirname(__file__), "../../../config/test_config_empty.json"
    )


@pytest.fixture
def invalid_config_path() -> str:
    return "nonexistent_config.json"


@pytest.fixture
def mock_bot() -> ChatbotBase:
    """Create a mock chatbot for testing."""
    mock = Mock(spec=ChatbotBase)
    mock.name = "TestBot"
    mock.bot_index = 1
    mock.system_prompt = "Initial prompt"
    mock.model_max_tokens = 100
    return mock


@pytest.fixture
def suffix_manager() -> SuffixManager:
    """Provide a SuffixManager instance for testing."""
    return SuffixManager()


@pytest.fixture
def sample_conversation_data() -> List[ConversationMessage]:
    """Return sample conversation data for testing."""
    return [
        {"bot_index": 0, "content": "Test seed message"},
        {"bot_index": 1, "content": "Bot1 response"},
        {"bot_index": 2, "content": "Bot2 response"},
    ]


@pytest.fixture
def sample_conversation_config() -> ConversationConfig:
    """
    Provide a valid ConversationConfig with dictionary-based bots.
    """
    return ConversationConfig(
        author="Test Author",
        conversation_seed="Test seed",
        rounds=2,
        shared_prefix="Shared prefix",
        first_round_postfix="First round postfix",
        last_round_postfix="Last round postfix",
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
