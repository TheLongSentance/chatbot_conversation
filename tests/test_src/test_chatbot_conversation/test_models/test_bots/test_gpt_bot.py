"""Tests specific to GPTChatbot implementation"""

from unittest.mock import MagicMock, patch

import pytest
from openai import APIConnectionError, APIError, RateLimitError

from chatbot_conversation.models.base import ChatbotConfig, ConversationMessage
from chatbot_conversation.models.bots.gpt_bot import MODEL_TYPE, GPTChatbot


class TestGPTChatbot:
    """Test GPT-specific chatbot functionality"""

    def test_model_type(self, gpt_config_for_tests: ChatbotConfig) -> None:
        """Test that GPT model type constant is correctly used"""
        bot = GPTChatbot(gpt_config_for_tests)
        assert (
            bot._get_class_model_type()  # pyright: ignore[reportPrivateUsage]
            == MODEL_TYPE
        )
        assert bot.model_type == MODEL_TYPE

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
        """Test handling of empty responses from OpenAI API"""
        mock_completion = MagicMock()
        mock_completion.choices[0].message.content = ""
        mock_openai.return_value.chat.completions.create.return_value = mock_completion

        bot = GPTChatbot(gpt_config_for_tests)
        conversation: list[ConversationMessage] = [{"bot_index": 1, "content": "Hello"}]

        with pytest.raises(ValueError, match="Model returned an empty response"):
            bot.generate_response(conversation)
