"""Tests specific to GPTChatbot implementation"""

from unittest.mock import MagicMock, patch

import pytest
from openai import APIConnectionError, APIError, RateLimitError

from chatbot_conversation.models.base import ChatbotConfig, ConversationMessage
from chatbot_conversation.models.bots.gpt_bot import GPT_MODEL_TYPE, GPTChatbot
from chatbot_conversation.utils import ModelException, ValidationException


class TestGPTChatbot:
    """Test GPT-specific chatbot functionality"""

    def test_model_type(self, gpt_config_for_tests: ChatbotConfig) -> None:
        """Test that GPT model type constant is correctly used"""
        bot = GPTChatbot(gpt_config_for_tests)
        assert (
            bot._get_class_model_type()  # pyright: ignore[reportPrivateUsage]
            == GPT_MODEL_TYPE
        )
        assert bot.model_type == GPT_MODEL_TYPE

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
        gpt_config_for_tests: ChatbotConfig,
        exception: Exception,
        should_retry: bool,
    ) -> None:
        """Test retry logic for OpenAI-specific exceptions"""
        bot = GPTChatbot(gpt_config_for_tests)
        assert (
            bot._should_retry_on_exception(  # pyright: ignore[reportPrivateUsage]
                exception
            )
            == should_retry
        )

    @patch("chatbot_conversation.models.bots.gpt_bot.OpenAI")
    def test_api_call_parameters(
        self, mock_openai: MagicMock, gpt_config_for_tests: ChatbotConfig
    ) -> None:
        """Test OpenAI API call parameter formatting"""
        # Mock models list
        mock_model = MagicMock()
        mock_model.id = "gpt-4o-mini"
        mock_openai.return_value.models.list.return_value = [mock_model]

        # Create a mock response
        mock_completion = MagicMock()
        mock_completion.choices[0].message.content = "Test response"
        mock_openai.return_value.chat.completions.create.return_value = mock_completion

        # Create bot and test conversation
        bot = GPTChatbot(gpt_config_for_tests)
        conversation: list[ConversationMessage] = [
            {"bot_index": 0, "content": "Explain the meaning of life"}
        ]

        # Call the method that will use the mock chain to generate a response
        response = bot._generate_response(  # pyright: ignore[reportPrivateUsage]
            conversation
        )

        # Verify the response
        assert response == "Test response"

        # Verify create was called with correct parameters
        create_call = mock_openai.return_value.chat.completions.create.call_args
        assert create_call is not None, "Create method was not called"

        call_kwargs = create_call[1]
        assert call_kwargs["model"] == gpt_config_for_tests.model.version
        assert call_kwargs["temperature"] == bot.model_temperature
        assert call_kwargs["timeout"] == bot.model_timeout.api_timeout
        assert isinstance(call_kwargs["messages"], list)
        assert len(call_kwargs["messages"]) >= 2  # System prompt + user message
        assert call_kwargs["messages"][0]["role"] == "system"
        assert call_kwargs["messages"][1]["role"] == "user"

    @patch("chatbot_conversation.models.bots.gpt_bot.OpenAI")
    def test_empty_response_handling(
        self, mock_openai: MagicMock, gpt_config_for_tests: ChatbotConfig
    ) -> None:
        # Mock models list
        mock_model = MagicMock()
        mock_model.id = "gpt-4o-mini"
        mock_openai.return_value.models.list.return_value = [mock_model]

        """Test handling of empty responses from OpenAI API"""
        mock_completion = MagicMock()
        mock_completion.choices[0].message.content = ""
        mock_openai.return_value.chat.completions.create.return_value = mock_completion

        bot = GPTChatbot(gpt_config_for_tests)
        conversation: list[ConversationMessage] = [{"bot_index": 1, "content": "Hello"}]

        with pytest.raises(ModelException, match="Model returned an empty string response"):
            bot.generate_response(conversation)

    def test_available_versions_returns_valid_list(self) -> None:
        """Test that available_versions returns non-empty list of model versions"""
        versions = GPTChatbot.available_versions()
        assert versions is not None
        assert len(versions) > 0
        assert isinstance(versions, list)
        assert all(isinstance(v, str) for v in versions)
        # Common models that should be available
        assert any("gpt-4" in v for v in versions)
        assert any("gpt-3.5" in v for v in versions)

    def test_bot_creation_with_valid_version(
        self, gpt_config_for_tests: ChatbotConfig
    ) -> None:
        """Test that bot creation with valid version succeeds"""
        # Use first available version from API
        versions = GPTChatbot.available_versions()
        assert versions is not None
        gpt_config_for_tests.model.version = versions[0]

        bot = GPTChatbot(gpt_config_for_tests)
        assert bot.model_version == versions[0]

    def test_bot_creation_with_invalid_version(
        self, gpt_config_for_tests: ChatbotConfig
    ) -> None:
        """Test that bot creation with invalid version fails"""
        gpt_config_for_tests.model.version = "invalid-model-version"
        with pytest.raises(ValidationException, match="Invalid model version"):
            GPTChatbot(gpt_config_for_tests)

    def test_version_caching(self) -> None:
        """Test that available versions are cached"""
        # Clear cache first
        GPTChatbot._available_versions_cache = None  # pyright: ignore[reportPrivateUsage]

        # First call should hit API
        versions1 = GPTChatbot.available_versions()

        # Second call should use cache
        versions2 = GPTChatbot.available_versions()

        assert versions1 == versions2
        assert GPTChatbot._available_versions_cache == versions1  # pyright: ignore[reportPrivateUsage]
