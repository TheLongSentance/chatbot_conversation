"""Tests specific to ClaudeChatbot implementation"""

from unittest.mock import MagicMock, patch

import pytest
from anthropic import APIConnectionError, APIError, RateLimitError

from chatbot_conversation.models.base import (
    ChatbotConfig,
    ChatbotModel,
    ConversationMessage,
)
from chatbot_conversation.models.bots.claude_bot import CLAUDE_MODEL_TYPE, ClaudeChatbot
from chatbot_conversation.utils import ModelException, ValidationException


class TestClaudeChatbot:
    """Test Claude-specific chatbot functionality"""

    def test_model_type(self, claude_config_for_tests: ChatbotConfig) -> None:
        """Test that Claude model type constant is correctly used"""
        bot = ClaudeChatbot(claude_config_for_tests)
        assert (
            bot._get_class_model_type()  # pyright: ignore[reportPrivateUsage]
            == CLAUDE_MODEL_TYPE
        )
        assert bot.model_type == CLAUDE_MODEL_TYPE

    @pytest.mark.parametrize(
        "exception,should_retry",
        [
            (APIError("test", request=MagicMock(), body=None), True),
            (APIConnectionError(request=MagicMock()), True),
            (RateLimitError("test", response=MagicMock(), body=None), True),
            (ValueError("test"), False),
            (Exception("test"), False),
        ],
    )
    def test_should_retry_on_exception(
        self,
        claude_config_for_tests: ChatbotConfig,
        exception: Exception,
        should_retry: bool,
    ) -> None:
        """Test retry logic for Claude-specific exceptions"""
        bot = ClaudeChatbot(claude_config_for_tests)
        assert (
            bot._should_retry_on_exception(  # pyright: ignore[reportPrivateUsage]
                exception
            )
            == should_retry
        )

    @patch("chatbot_conversation.models.bots.claude_bot.anthropic.Anthropic")
    def test_api_call_parameters(
        self, mock_anthropic: MagicMock, claude_config_for_tests: ChatbotConfig
    ) -> None:
        """Test Claude API call parameter formatting"""
        # Mock models list
        mock_model = MagicMock()
        mock_model.id = "claude-3-haiku-20240307"
        mock_anthropic.return_value.models.list.return_value = [mock_model]

        # Create a mock response
        mock_message = MagicMock()
        mock_message.content[0].text = "Test response"
        mock_anthropic.return_value.messages.create.return_value = mock_message

        # Create bot and test conversation
        bot = ClaudeChatbot(claude_config_for_tests)
        conversation: list[ConversationMessage] = [
            {"bot_index": 0, "content": "Explain quantum computing"}
        ]

        # Call the method that will use the mock chain to generate a response
        response = bot._generate_response(  # pyright: ignore[reportPrivateUsage]
            conversation
        )

        # Verify the response
        assert response == "Test response"

        # Verify create was called with correct parameters
        create_call = mock_anthropic.return_value.messages.create.call_args
        assert create_call is not None, "Create method was not called"

        call_kwargs = create_call[1]
        assert call_kwargs["model"] == claude_config_for_tests.model.version
        assert call_kwargs["temperature"] == bot.model_temperature
        assert call_kwargs["timeout"] == bot.model_timeout.api_timeout
        assert isinstance(call_kwargs["messages"], list)
        assert "system" in call_kwargs
        assert call_kwargs["system"] == bot.system_prompt

    @patch("chatbot_conversation.models.bots.claude_bot.anthropic.Anthropic")
    def test_empty_response_handling(
        self, mock_anthropic: MagicMock, claude_config_for_tests: ChatbotConfig
    ) -> None:
        """Test handling of empty responses from Claude API"""
        # Mock models list
        mock_model = MagicMock()
        mock_model.id = "claude-3-haiku-20240307"
        mock_anthropic.return_value.models.list.return_value = [mock_model]

        # Mock empty response
        mock_message = MagicMock()
        mock_message.content[0].text = ""
        mock_anthropic.return_value.messages.create.return_value = mock_message

        bot = ClaudeChatbot(claude_config_for_tests)
        conversation: list[ConversationMessage] = [{"bot_index": 1, "content": "Hello"}]

        with pytest.raises(
            ModelException, match=".*Model returned an empty string response.*"
        ):
            bot.generate_response(conversation)

    def test_available_versions_live(self) -> None:
        """Test retrieving available model versions using live API"""
        versions = ClaudeChatbot.available_versions()
        assert versions is not None
        assert len(versions) > 0
        # Verify some known Claude models are present
        assert any("claude" in version for version in versions)

    def test_valid_model_version_live(self) -> None:
        """Test initialization with valid model version using live API"""
        versions = ClaudeChatbot.available_versions()
        assert versions is not None
        valid_version = next(v for v in versions if "claude-3" in v)

        config = ChatbotConfig(
            name="TestBot",
            system_prompt="Test prompt",
            model=ChatbotModel(type="CLAUDE", version=valid_version),
        )

        # Should not raise any exceptions
        bot = ClaudeChatbot(config)
        assert bot.model_version == valid_version

    def test_invalid_model_version_live(self) -> None:
        """Test initialization fails with invalid model version using live API"""
        config = ChatbotConfig(
            name="TestBot",
            system_prompt="Test prompt",
            model=ChatbotModel(type="CLAUDE", version="invalid-version"),
        )

        with pytest.raises(ValidationException, match="Invalid model version"):
            ClaudeChatbot(config)
