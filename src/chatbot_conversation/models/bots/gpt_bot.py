"""
OpenAI GPT API integration for chatbot functionality.

A concrete implementation of ChatbotBase specifically designed for OpenAI's GPT models.
Handles API communication, message formatting, and conversation management with
configurable parameters and robust error handling.

Features:
    - Full OpenAI GPT API integration with authentication
    - Configurable model parameters (temperature, tokens, timeouts)
    - Streaming and non-streaming response generation
    - Automatic retry handling for API failures
    - Conversation history management
    - System prompt customization

Supported Models:
    - gpt-4: Latest GPT-4 model
    - gpt-4-turbo: Optimized GPT-4 for faster responses
    - gpt-3.5-turbo: Enhanced GPT-3.5 model
    - gpt-3.5: Base GPT-3.5 model

Dependencies:
    - openai: Official OpenAI Python client
    - ChatbotBase: Base class for chatbot implementations
"""

from typing import Any, Iterator, List, Optional, Type

import openai
from openai import OpenAI

from chatbot_conversation.models.base import (
    ChatbotBase,
    ChatbotConfig,
    ConversationMessage,
)
from chatbot_conversation.models.bot_registry import register_bot
from chatbot_conversation.utils import APIException, ErrorSeverity

# OpenAI default temperature for GPT models
GPT_MINIMUM_TEMPERATURE = 0.0
GPT_MAXIMUM_TEMPERATURE = 2.0
GPT_DEFAULT_TEMPERATURE = 1.0

GPT_MODEL_TYPE = "GPT"


@register_bot(GPT_MODEL_TYPE)
class GPTChatbot(ChatbotBase):
    """
    Chatbot implementation using OpenAI's GPT API service.

    Provides a concrete implementation of ChatbotBase for GPT models,
    extending core functionality with OpenAI-specific API integration.

    Features:
    - OpenAI API authentication and communication
    - Message formatting for GPT models
    - Temperature-controlled response generation
    - Automatic retry handling for API failures
    - Support for multiple GPT model versions
    - Configurable system prompts and timeouts

    Args:
        config (ChatbotConfig): Configuration object containing:
            - name: Bot instance identifier
            - system_prompt: Initial system behavior instructions
            - model: Model type, version and parameters
            - timeout: API communication settings

    Attributes:
        Inherits all attributes from ChatbotBase plus:
        _model_api (openai.OpenAI): Authenticated OpenAI API client

    Notes:
        Requires OpenAI API key to be set in environment variables
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
                api = OpenAI()
                models = api.models.list()
                cls._available_versions_cache = [model.id for model in models]
            except (
                openai.APIConnectionError,
                openai.APIError,
            ) as e:
                error_msg = f"Failed to retrieve model versions: {e}"
                raise APIException(
                    message=error_msg,
                    user_message="Failed to retrieve available model versions from GPT API",
                    severity=ErrorSeverity.ERROR,
                    original_error=e,
                ) from e

        return cls._available_versions_cache

    @classmethod
    def _get_class_model_type(cls) -> str:
        """
        Get the model type identifier for GPT models.

        Returns:
            str: "GPT" as the model type identifier
        """
        return GPT_MODEL_TYPE

    @classmethod
    def _get_model_min_temperature(cls) -> float:
        """Get the minimum allowed temperature value."""
        return GPT_MINIMUM_TEMPERATURE

    @classmethod
    def _get_model_max_temperature(cls) -> float:
        """Get the maximum allowed temperature value."""
        return GPT_MAXIMUM_TEMPERATURE

    @classmethod
    def _get_model_default_temperature(cls) -> float:
        """Get the default temperature value."""
        return GPT_DEFAULT_TEMPERATURE

    @classmethod
    def _retryable_exceptions(cls) -> tuple[Type[Exception], ...]:
        """
        Returns tuple of OpenAI-specific retryable exception types.

        Returns:
            tuple: Exception types that warrant retry attempts
        """
        retryable_types = (
            openai.APIError,
            APIException,
            openai.APIConnectionError,
            openai.RateLimitError,
            ConnectionError,
            TimeoutError,
        )
        return retryable_types

    def __init__(self, config: ChatbotConfig) -> None:
        """
        Initialize OpenAI chatbot with specified configuration.

        Validates configuration and sets up OpenAI API client.
        API key must be available in environment variables.

        Args:
            config (ChatbotConfig): Complete bot configuration
        """
        super().__init__(config)

        self._model_api = OpenAI()

    def _generate_response(self, conversation: List[ConversationMessage]) -> str:
        """
        Generate a response using the OpenAI API.

        Makes a synchronous API call to generate a response based on the
        conversation history. Applies all configured parameters including
        temperature, token limits, and timeouts.

        Args:
            conversation (List[ConversationMessage]): Complete conversation history
                including system prompts and user messages

        Returns:
            str: The generated response text from the GPT model

        Raises:
            APIError: For general API communication failures
            APIConnectionError: When network connectivity fails
            RateLimitError: When API rate/quota limits are exceeded
            TimeoutError: When the API call exceeds configured timeout
        """
        response_content: str = ""
        completion = self._model_api.chat.completions.create(
            model=self.model_version,
            messages=self._format_conv_for_api_util(conversation),
            stream=False,
            timeout=self.model_timeout.api_timeout,
            temperature=self.model_temperature,
            max_tokens=self.model_max_tokens,
        )
        response_content = completion.choices[0].message.content
        return response_content

    def _get_text_from_chunk(self, chunk: Any) -> str:
        """
        Extract text content from an OpenAI API streaming response chunk.

        Processes a single chunk from the streaming API response to extract
        the generated text content. Handles empty chunks gracefully.

        Args:
            chunk (Any): Raw chunk from OpenAI's streaming API response

        Returns:
            str: Extracted text content, or empty string if chunk contains no content

        Notes:
            - Chunk structure is specific to OpenAI's streaming format
            - Empty chunks may occur during streaming and are handled safely
        """
        return chunk.choices[0].delta.content or ""

    def _generate_stream(self, conversation: list[ConversationMessage]) -> Iterator[Any]:
        """
        Generate a streaming response using the OpenAI API.

        Creates a streaming API connection that yields response chunks in real-time
        as they're generated. Enables progressive response display and potentially
        faster first-token response times.

        Args:
            conversation (list[ConversationMessage]): Complete conversation history
                including system prompts and user messages

        Returns:
            Iterator[Any]: Stream of response chunks from the OpenAI API

        Notes:
            - Streaming mode reduces time to first token
            - All configured parameters (temperature, tokens, etc.) are applied
            - Each chunk contains partial response text
            - Use _get_text_from_chunk() to process individual chunks
        """
        return self._model_api.chat.completions.create(  # type: ignore
            model=self.model_version,
            messages=self._format_conv_for_api_util(conversation),
            stream=True,
            timeout=self.model_timeout.api_timeout,
            temperature=self.model_temperature,
            max_tokens=self.model_max_tokens,
        )
