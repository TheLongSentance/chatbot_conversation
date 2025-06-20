"""
Ollama API integration for chatbot conversations.

This module provides a concrete implementation of ChatbotBase specifically designed
for Ollama's local LLM service. It handles all aspects of API communication,
message formatting, and conversation management with configurable parameters.

Key Features:
- Local LLM integration via Ollama API
- Stateful conversation management
- Configurable temperature and token limits
- Support for both streaming and non-streaming responses
- Automatic retry mechanisms for network failures

Note:
    Ollama uses a normalized temperature range (0.0-1.0) which differs from
    other implementations that typically use 0.0-2.0.
"""

from typing import Any, Iterator, List, Optional, Type

import httpx
import ollama
from ollama import ChatResponse

from chatbot_conversation.models.base import ChatbotBase, ConversationMessage
from chatbot_conversation.models.bot_registry import register_bot
from chatbot_conversation.utils import APIException, ErrorSeverity

# Model temperature range specifically for Ollama API
OLLAMA_MINIMUM_TEMPERATURE = 0.0
OLLAMA_MAXIMUM_TEMPERATURE = 1.0
OLLAMA_DEFAULT_TEMPERATURE = 0.8

OLLAMA_MODEL_TYPE = "OLLAMA"


@register_bot(OLLAMA_MODEL_TYPE)
class OllamaChatbot(ChatbotBase):
    """
    Chatbot implementation using Ollama's local API service.

    This class provides a concrete implementation of ChatbotBase for Ollama models,
    handling all API interactions and conversation management. It supports both
    streaming and non-streaming responses with configurable parameters.

    Features:
    - Local LLM integration via HTTP API (default: http://localhost:11434)
    - Normalized temperature range (0.0-1.0)
    - Automatic retry mechanism for network failures
    - Configurable system prompts and model parameters
    - Stateful conversation history management
    - Token limit enforcement
    - Streaming response support

    Args:
        config (ChatbotConfig): Configuration containing:
            - name (str): Bot instance identifier
            - system_prompt (str): Initial system behavior instructions
            - model (str): Model identifier and version
            - temperature (float, optional): Response randomness (0.0-1.0)
            - max_tokens (int, optional): Maximum response length
            - timeout (float, optional): API request timeout in seconds

    Raises:
        ValueError: If configuration parameters are invalid
        ConnectionError: If initial API connection fails
    """

    # no __init__() method needed, OllamaChatbot uses the base class __init__()
    # which is automatically called when creating an instance of this class

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
                response = ollama.list()
                models = getattr(response, "models", [])
                # strip off colon so for example: "llama3.2:latest" becomes "llama3.2"
                cls._available_versions_cache = [
                    str(model.model).split(":", maxsplit=1)[0] for model in models
                ]
            except Exception as e:
                error_msg = f"Failed to retrieve installed model versions for Ollama: {e}"
                raise APIException(
                    message=error_msg,
                    user_message="Failed to retrieve installed model versions from Ollama API",
                    severity=ErrorSeverity.ERROR,
                    original_error=e,
                ) from e
        return cls._available_versions_cache

    @classmethod
    def _get_class_model_type(cls) -> str:
        """
        Get the model type identifier for Ollama models.

        Returns:
            str: "OLLAMA" as the model type identifier
        """
        return OLLAMA_MODEL_TYPE

    @classmethod
    def _get_model_min_temperature(cls) -> float:
        """Get the minimum allowed temperature value."""
        return OLLAMA_MINIMUM_TEMPERATURE

    @classmethod
    def _get_model_max_temperature(cls) -> float:
        """Get the maximum allowed temperature value."""
        return OLLAMA_MAXIMUM_TEMPERATURE

    @classmethod
    def _get_model_default_temperature(cls) -> float:
        """Get the default temperature value."""
        return OLLAMA_DEFAULT_TEMPERATURE

    @classmethod
    def _retryable_exceptions(cls) -> tuple[Type[Exception], ...]:
        """
        Returns tuple of Ollama-specific retryable exception types.

        Returns:
            tuple: Exception types that warrant retry attempts
        """
        retryable_types = (
            APIException,
            ConnectionError,
            httpx.TimeoutException,
            httpx.NetworkError,
            httpx.HTTPStatusError,
            TimeoutError,
        )
        return retryable_types

    def _generate_response(self, conversation: List[ConversationMessage]) -> str:
        """
        Generate a response using the Ollama API.

        Formats conversation history and makes API call with configured parameters.
        Handles message structure requirements specific to Ollama's API.

        Args:
            conversation: Sequential list of prior conversation messages

        Returns:
            str: Generated response text from Ollama

        Raises:
            httpx.TimeoutException: On connection/read timeout
            httpx.NetworkError: On network connectivity issues
            httpx.HTTPStatusError: On HTTP error responses
        """
        response_content: str = ""
        response: ChatResponse = ollama.chat(
            model=self.model_version,
            messages=self._format_conv_for_api_util(conversation),
            stream=False,
            options={
                "temperature": self.model_temperature,
                "num_predict": self.model_max_tokens,
            },
        )
        response_content = response["message"]["content"]
        return response_content

    def _get_text_from_chunk(self, chunk: Any) -> str:
        """
        Extract text content from a streaming response chunk.

        Args:
            chunk (Any): Response chunk from Ollama streaming API

        Returns:
            str: Extracted text content from the chunk, or empty string if not found
        """
        return chunk.get("message", {}).get("content", "")  # type: ignore

    def _generate_stream(self, conversation: list[ConversationMessage]) -> Iterator[Any]:
        """
        Generate streaming responses using the Ollama API.

        Implements streaming response generation by making API calls with
        stream=True, allowing for real-time token generation and processing.

        Args:
            conversation: Sequential list of conversation messages

        Returns:
            Iterator[Any]: Stream of response chunks from Ollama API

        Raises:
            httpx.TimeoutException: On connection/read timeout
            httpx.NetworkError: On network connectivity issues
            httpx.HTTPStatusError: On HTTP error responses
        """
        return ollama.chat(
            model=self.model_version,
            messages=self._format_conv_for_api_util(conversation),
            stream=True,
            options={
                "temperature": self.model_temperature,
                "num_predict": self.model_max_tokens,
            },
        )
