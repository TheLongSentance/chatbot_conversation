"""Tests specific to GeminiChatbot implementation"""

from typing import List
from unittest.mock import MagicMock, patch

import google.api_core.exceptions
import pytest

from chatbot_conversation.models.base import ChatbotConfig, ConversationMessage
from chatbot_conversation.models.bots.gemini_bot import (
    _GeminiMessage,  # pyright: ignore[reportPrivateUsage]
)
from chatbot_conversation.models.bots.gemini_bot import GEMINI_MODEL_TYPE, GeminiChatbot


class TestGeminiChatbot:
    """Test Gemini-specific chatbot functionality"""

    def test_model_type(self, gemini_chatbot: GeminiChatbot) -> None:
        """Test that Gemini model type constant is correctly used"""
        assert (
            gemini_chatbot._get_class_model_type()  # pyright: ignore[reportPrivateUsage]
            == GEMINI_MODEL_TYPE
        )
        assert gemini_chatbot.model_type == GEMINI_MODEL_TYPE

    @pytest.mark.parametrize(
        "exception,should_retry",
        [
            (
                google.api_core.exceptions.DeadlineExceeded("test"),  # type: ignore
                True,
            ),
            (
                google.api_core.exceptions.ServiceUnavailable("test"),  # type: ignore
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

        formatted: List[_GeminiMessage] = (  # pyright: ignore[reportUnknownVariableType]
            gemini_chatbot._format_conv_for_gemini_api(  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType]
                conversation
            )
        )

        # Verify message format and roles
        assert len(formatted) == 3  # pyright: ignore[reportUnknownArgumentType]
        assert formatted[0] == {"role": "user", "parts": conversation[0]["content"]}
        assert formatted[1] == {"role": "model", "parts": conversation[1]["content"]}
        assert formatted[2] == {"role": "user", "parts": conversation[2]["content"]}

    def test_available_versions_returns_valid_list(self) -> None:
        """Test that available_versions returns non-empty list of model versions"""
        versions = GeminiChatbot.available_versions()
        assert versions is not None
        assert len(versions) > 0
        assert isinstance(versions, list)
        assert all(isinstance(v, str) for v in versions)

    def test_bot_creation_with_valid_version(
        self, gemini_config_for_tests: ChatbotConfig
    ) -> None:
        """Test that bot creation with valid version succeeds"""
        # Use the first available version from the API
        versions = GeminiChatbot.available_versions()
        assert versions is not None
        gemini_config_for_tests.model.version = versions[0]
        bot = GeminiChatbot(gemini_config_for_tests)
        assert bot.model_version == versions[0]

    def test_bot_creation_with_invalid_version(
        self, gemini_config_for_tests: ChatbotConfig
    ) -> None:
        """Test that bot creation with invalid version fails"""
        gemini_config_for_tests.model.version = "invalid-model-version"
        with pytest.raises(ValueError, match="Invalid model version"):
            GeminiChatbot(gemini_config_for_tests)

    def test_version_caching(self) -> None:
        """Test that available versions are cached"""
        # Clear cache first
        GeminiChatbot._available_versions_cache = None  # pyright: ignore[reportPrivateUsage]
        
        # First call should hit API
        versions1 = GeminiChatbot.available_versions()

        # Second call should use cache
        versions2 = GeminiChatbot.available_versions()

        assert versions1 == versions2
        assert GeminiChatbot._available_versions_cache == versions1  # pyright: ignore[reportPrivateUsage]

