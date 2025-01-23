"""Tests specific to GeminiChatbot implementation"""

from typing import List
from unittest.mock import MagicMock, patch

import google.api_core.exceptions
import pytest

from chatbot_conversation.models.base import ChatbotConfig, ConversationMessage
from chatbot_conversation.models.bots.gemini_bot import (
    _GeminiMessage,  # pyright: ignore[reportPrivateUsage]
)
from chatbot_conversation.models.bots.gemini_bot import MODEL_TYPE, GeminiChatbot


class TestGeminiChatbot:
    """Test Gemini-specific chatbot functionality"""

    def test_model_type(self, gemini_chatbot: GeminiChatbot) -> None:
        """Test that Gemini model type constant is correctly used"""
        assert (
            gemini_chatbot._get_class_model_type()  # pyright: ignore[reportPrivateUsage]
            == MODEL_TYPE
        )
        assert gemini_chatbot.model_type == MODEL_TYPE

    @pytest.mark.parametrize(
        "exception,should_retry",
        [
            (
                google.api_core.exceptions.DeadlineExceeded("test"),  # type: ignore
                True,
            ),
            (
                google.api_core.exceptions.ServiceUnavailable("test"), # type: ignore
                True,
            ),
            (ValueError("test"), False),
            (Exception("test"), False),
        ],
    )
    def test_should_retry_on_exception(
        self,
        gemini_chatbot: GeminiChatbot,
        exception: Exception,
        should_retry: bool,
    ) -> None:
        """Test retry logic for Gemini-specific exceptions"""
        assert (
            gemini_chatbot._should_retry_on_exception(  # pyright: ignore[reportPrivateUsage]
                exception
            )
            == should_retry
        )

    @patch("google.generativeai.GenerativeModel")  
    def test_api_call_parameters(
        self,
        mock_gemini_model: MagicMock,
        gemini_config_for_tests: ChatbotConfig,
    ) -> None:
        """Test Gemini API call parameter formatting"""
        # Create mock response
        mock_response = MagicMock()
        mock_response.text = "Test response"
        mock_gemini_model.return_value.generate_content.return_value = mock_response

        # Create bot and test conversation
        bot = GeminiChatbot(gemini_config_for_tests)
        conversation: list[ConversationMessage] = [
            {"bot_index": 0, "content": "Explain quantum computing"}
        ]

        # Call the method
        response = bot._generate_response(  # pyright: ignore[reportPrivateUsage]
            conversation
        )

        # Verify response
        assert response == "Test response"

        # Verify model initialization parameters
        model_init_call = mock_gemini_model.call_args
        assert model_init_call is not None
        kwargs = model_init_call[1]
        assert kwargs["model_name"] == gemini_config_for_tests.model.version
        assert kwargs["system_instruction"] == bot.system_prompt

        # Verify generation config
        gen_config = kwargs["generation_config"]
        assert gen_config.temperature == bot.model_temperature
        assert gen_config.max_output_tokens == bot.model_max_tokens

    @patch("google.generativeai.GenerativeModel")
    def test_system_prompt_reinit(
        self,
        mock_gemini_model: MagicMock,
        gemini_config_for_tests: ChatbotConfig,
    ) -> None:
        """Test API reinitialization when system prompt changes"""
        # Setup mock response
        mock_response = MagicMock()
        mock_response.text = "Test response"
        mock_gemini_model.return_value.generate_content.return_value = mock_response

        # Create bot
        bot = GeminiChatbot(gemini_config_for_tests)
        conversation: list[ConversationMessage] = [{"bot_index": 0, "content": "Hello"}]

        # First call should initialize model
        initial_prompt = bot.system_prompt
        bot._generate_response(conversation)  # pyright: ignore[reportPrivateUsage]
        assert mock_gemini_model.call_args[1]["system_instruction"] == initial_prompt

        # Change system prompt
        new_prompt = "You are now a different assistant"
        bot.system_prompt = new_prompt

        # Second call should reinitialize with new prompt
        bot._generate_response(conversation)  # pyright: ignore[reportPrivateUsage]
        assert mock_gemini_model.call_args[1]["system_instruction"] == new_prompt

    def test_format_conv_for_gemini_api(self, gemini_chatbot: GeminiChatbot) -> None:
        """Test conversation formatting for Gemini API"""
        # Create test conversation with mixed bot and user messages
        conversation: list[ConversationMessage] = [
            {"bot_index": 0, "content": "Hello"},  # User message (different bot index)
            {
                "bot_index": gemini_chatbot.bot_index,
                "content": "Hi there",
            },  # Bot response
            {"bot_index": 0, "content": "How are you?"},  # User message
        ]

        # Format the conversation
        formatted: List[_GeminiMessage] = (   # pyright: ignore[reportUnknownVariableType]  
            gemini_chatbot._format_conv_for_gemini_api(  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType]
                conversation
            )
        )

        # Verify message format and roles
        assert len(formatted) == 3  # pyright: ignore[reportUnknownArgumentType]
        assert formatted[0] == {"role": "user", "parts": conversation[0]["content"]}  
        assert formatted[1] == {"role": "model", "parts": conversation[1]["content"]}  
        assert formatted[2] == {"role": "user", "parts": conversation[2]["content"]}  
