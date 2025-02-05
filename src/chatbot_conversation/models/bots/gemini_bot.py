"""
Google Gemini API integration for chatbot functionality.

This module provides a Gemini-specific implementation of the ChatbotBase class,
enabling interaction with Google's Gemini language models through their API.

Features:
- Seamless integration with Google's Gemini API
- Support for multiple Gemini model variants
- Dynamic temperature control
- Streaming response capability
- Automatic API reinitialization on prompt changes
- Comprehensive error handling

Supported Models:
    - gemini-1.5-flash: Optimized for low-latency inference
    - gemini-1.5-pro: Latest generation pro model
    - gemini-1.0-pro-vision: Multimodal model with vision capabilities
    - gemini-1.0-pro-002: Enhanced reasoning capabilities
    - gemini-1.0-pro-001: Base professional model

Classes:
    GeminiChatbot: Main implementation class for Gemini-based chat functionality
    _GeminiMessage: Internal type for API message formatting
"""

import json
from typing import Any, Iterator, List, Optional, TypedDict

import google.api_core.exceptions

# no stub file from google.generativeai so ignore for pylance etc
import google.generativeai  # type: ignore

from chatbot_conversation.models.base import (
    ChatbotBase,
    ChatbotConfig,
    ConversationMessage,
)
from chatbot_conversation.models.bot_registry import register_bot
from chatbot_conversation.utils import APIException, ErrorSeverity

# Gemini 1.5 models default temperature (others may vary)
GEMINI_MINIMUM_TEMPERATURE = 0.0
GEMINI_MAXIMUM_TEMPERATURE = 2.0
GEMINI_DEFAULT_TEMPERATURE = 1.0

GEMINI_MODEL_TYPE = "GEMINI"


class _GeminiMessage(TypedDict):
    """
    Internal message format for Gemini API calls.

    Attributes:
        role (str): Message source ('model' or 'user')
        parts (str): Message content text
    """

    role: str
    parts: str


@register_bot(GEMINI_MODEL_TYPE)
class GeminiChatbot(ChatbotBase):
    """
    Gemini-specific chatbot implementation using Google's AI models.

    This class provides a concrete implementation of ChatbotBase specifically
    designed for Google's Gemini AI models. It handles all aspects of API
    communication, message formatting, and conversation management.

    Key Features:
    - Automatic API authentication and session management
    - Dynamic system prompt handling with auto-reinitialization
    - Configurable temperature settings for response randomness
    - Stream and non-stream response generation
    - Robust error handling with retry logic
    - Conversation state management

    Args:
        config (ChatbotConfig): Configuration containing:
            - name (str): Unique identifier for the bot instance
            - system_prompt (str): Initial system behavior instructions
            - model (str): Specific Gemini model identifier
            - api_key (str): Google API authentication key
            - temperature (float, optional): Response randomness (0.0-2.0)
            - timeout (int, optional): API request timeout in seconds

    Attributes:
        _model_api (google.generativeai.GenerativeModel): Active Gemini API client
        system_prompt (str): Current system instructions
        model_temperature (float): Current temperature setting

    Notes:
        Unlike other LLM APIs, Gemini requires system prompts to be set during
        model initialization. This means the API client must be reinitialized
        whenever the system prompt changes, which is handled automatically by
        this implementation.
    """

    @classmethod
    def available_versions(cls) -> Optional[List[str]]:
        """
        Get available model versions for this bot type.

        Returns:
            Optional[List[str]]: List of valid model versions, or None if
            versions are not applicable/available

        Raises:
            APIError: If API call to retrieve versions fails
        """
        if cls._available_versions_cache is None:
            try:
                google.generativeai.configure()  # pyright: ignore
                models = google.generativeai.list_models()  # pyright: ignore
                # extract version from model name "models/gemini-1.5-pro"
                cls._available_versions_cache = \
                    [model.name.split("/")[-1] for model in models]  # pyright: ignore
            except google.api_core.exceptions.GoogleAPIError as e:
                error_msg = f"Failed to retrieve model versions from Gemini API: {e}"
                raise APIException(
                    message=error_msg,
                    user_message="Failed to retrieve available model versions from Gemini API",
                    severity=ErrorSeverity.ERROR,
                    retry_allowed=False,
                    original_error=e,
                )
        return cls._available_versions_cache

    @classmethod
    def _get_class_model_type(cls) -> str:
        """
        Get the model type identifier for GPT models.

        Returns:
            str: "GPT" as the model type identifier
        """
        return GEMINI_MODEL_TYPE

    @classmethod
    def _get_model_min_temperature(cls) -> float:
        """Get the minimum allowed temperature value."""
        return GEMINI_MINIMUM_TEMPERATURE

    @classmethod
    def _get_model_max_temperature(cls) -> float:
        """Get the maximum allowed temperature value."""
        return GEMINI_MAXIMUM_TEMPERATURE

    @classmethod
    def _get_model_default_temperature(cls) -> float:
        """Get the default temperature value."""
        return GEMINI_DEFAULT_TEMPERATURE

    @classmethod
    def _should_retry_on_exception(cls, exception: Exception) -> bool:
        """
        Determine if an API call should be retried based on the exception type.

        Evaluates Gemini-specific API exceptions to decide if a retry attempt
        is appropriate based on the error condition.

        Args:
            exception (Exception): The caught exception from the API call

        Returns:
            bool: True if the error is transient and retry is recommended,
                 False for permanent failures

        Supported retry cases:
            - DeadlineExceeded: Timeout errors that may resolve
            - ServiceUnavailable: Temporary API availability issues
        """
        return isinstance(
            exception,
            (
                google.api_core.exceptions.DeadlineExceeded,
                google.api_core.exceptions.ServiceUnavailable,
            ),
        )

    def __init__(self, config: ChatbotConfig) -> None:
        """
        Initialize Gemini chatbot with specified configuration.

        Validates configuration and sets up Gemini API client with system prompt
        and temperature settings.

        Args:
            config (ChatbotConfig): Complete bot configuration
        """
        super().__init__(config)

        # no stub file from google.generativeai so ignore for pylance (-> pyright) etc
        google.generativeai.configure()  # pyright: ignore[reportUnknownMemberType]

        # initialise api here
        self._initialize_model_api()

    def _generate_response(self, conversation: List[ConversationMessage]) -> str:
        """
        Generate a response using the Gemini API based on conversation history.

        Processes the conversation history, handles system prompt updates,
        and manages API communication to generate appropriate responses.

        Args:
            conversation (List[ConversationMessage]): Complete conversation history
                including both user and bot messages

        Returns:
            str: Generated response text from the Gemini model

        Raises:
            google.api_core.exceptions.DeadlineExceeded: When request times out
            google.api_core.exceptions.ServiceUnavailable: When API is unavailable
            RuntimeError: For other API communication failures
        """
        formatted_messages = self._format_conv_for_gemini_api(conversation)

        message = self._model_api.generate_content(  # pyright: ignore[reportUnknownMemberType]
            formatted_messages
        )
        response: str = message.text
        return response

    def _format_conv_for_gemini_api(
        self, conversation: List[ConversationMessage]
    ) -> List[_GeminiMessage]:
        """
        Format conversation history for Gemini API submission.

        Converts internal message format to Gemini's expected structure with
        appropriate role assignments ('model' or 'user').

        Args:
            conversation: Complete conversation history to format

        Returns:
            List[_GeminiMessage]: Messages formatted for Gemini API submission
        """
        messages: List[_GeminiMessage] = []

        for contribution in conversation:
            role = "model" if contribution["bot_index"] == self.bot_index else "user"
            messages.append({"role": role, "parts": contribution["content"]})

        self._log_debug(json.dumps(messages, indent=2))

        return messages

    def _initialize_model_api(self) -> None:
        """
        Initialize or reinitialize the Gemini API client.

        Creates a new GenerativeModel instance with current configuration settings
        including system prompt, temperature, and token limits. Called on first
        initialization and when system prompt changes.

        Note:
            Updates model_system_prompt_updated flag after initialization
        """
        self._model_api = google.generativeai.GenerativeModel(
            model_name=self.model_version,
            system_instruction=self.system_prompt,
            generation_config=google.generativeai.GenerationConfig(
                temperature=self.model_temperature,
                max_output_tokens=self.model_max_tokens,
            ),
        )

    def _get_text_from_chunk(self, chunk: Any) -> str:
        """
        Extract text content from a streaming response chunk.

        Args:
            chunk (Any): Response chunk from Gemini streaming API

        Returns:
            str: Extracted text content from the chunk, or empty string if not found
        """
        return chunk.text or ""

    def _generate_stream(
        self, conversation: list[ConversationMessage]
    ) -> Iterator[Any]:
        """
        Generate streaming responses using the Gemini API.

        Reinitializes API client if system prompt has changed, then generates
        content in streaming mode.

        Args:
            conversation (list[ConversationMessage]): List of conversation messages

        Returns:
            Iterator[Any]: Iterator yielding response chunks from Gemini's streaming API
        """
        return self._model_api.generate_content(  # type: ignore
            self._format_conv_for_gemini_api(conversation),
            stream=True,
        )
