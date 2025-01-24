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
from typing import Any, Iterator, List, TypedDict

import google.api_core.exceptions

# no stub file from google.generativeai so ignore for pylance etc
import google.generativeai  # type: ignore

from chatbot_conversation.models.base import (
    ChatbotBase,
    ChatbotConfig,
    ConversationMessage,
)
from chatbot_conversation.models.bot_registry import register_bot
from chatbot_conversation.utils import get_logger

# Gemini 1.5 models default temperature (others may vary)
MINIMUM_TEMPERATURE = 0.0
MAXIMUM_TEMPERATURE = 2.0
DEFAULT_TEMPERATURE = 1.0

MODEL_TYPE = "GEMINI"


class _GeminiMessage(TypedDict):
    """
    Internal message format for Gemini API calls.

    Attributes:
        role (str): Message source ('model' or 'user')
        parts (str): Message content text
    """

    role: str
    parts: str


logger = get_logger("models")


@register_bot(MODEL_TYPE)
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
        model_api (google.generativeai.GenerativeModel): Active Gemini API client
        system_prompt (str): Current system instructions
        model_temperature (float): Current temperature setting

    Notes:
        Unlike other LLM APIs, Gemini requires system prompts to be set during
        model initialization. This means the API client must be reinitialized
        whenever the system prompt changes, which is handled automatically by
        this implementation.
    """

    @classmethod
    def _get_class_model_type(cls) -> str:
        """
        Get the model type identifier for GPT models.

        Returns:
            str: "GPT" as the model type identifier
        """
        return MODEL_TYPE

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

        # initialise api here, but will be updated in _generate_response
        # when system prompt is set or updated since it is not passed in
        # the generate_content call for Gemini as either a parameter or
        # part of the message history

        self._initialize_model_api()

    @property
    def model_min_temperature(self) -> float:
        """Get the minimum allowed temperature value."""
        return MINIMUM_TEMPERATURE

    @property
    def model_max_temperature(self) -> float:
        """Get the maximum allowed temperature value."""
        return MAXIMUM_TEMPERATURE

    @property
    def model_default_temperature(self) -> float:
        """Get the default temperature value."""
        return DEFAULT_TEMPERATURE

    def _should_retry_on_exception(self, exception: Exception) -> bool:
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

        # test if system prompt has changed and re-initialize API in order
        # to reset the system prompt for Gemini API. This is not typical
        # for other models as they include system prompt in either:
        # - as a parameter in the api call (e.g. Claude)
        # - or as part of the message history (e.g. OpenAI, Ollama)
        # for Gemini, this will happen when the system prompt is first set
        # and whenever it is updated (first round, after first round, before last)

        if self.model_system_prompt_needs_update:
            self._initialize_model_api()

        message = (
            self.model_api.generate_content(  # pyright: ignore[reportUnknownMemberType]
                formatted_messages
            )
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
        self.model_api = google.generativeai.GenerativeModel(
            model_name=self.model_version,
            system_instruction=self.system_prompt,
            generation_config=google.generativeai.GenerationConfig(
                temperature=self.model_temperature,
                max_output_tokens=self.model_max_tokens,
            ),
        )
        self.model_system_prompt_updated()

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
        if self.model_system_prompt_needs_update:
            self._initialize_model_api()

        return self.model_api.generate_content(  # type: ignore
            self._format_conv_for_gemini_api(conversation),
            stream=True,
        )
