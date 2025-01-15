"""
Ollama API integration for chatbot functionality.

Provides a concrete implementation of ChatbotBase for Ollama's local LLM service,
handling API communication, message formatting, and conversation management
with configurable parameters.

Major Classes:
    OllamaChatbot: Ollama-specific chatbot implementation

Notes:
    Ollama uses a modified temperature range (0.0-1.0) compared to other
    implementations which typically use 0.0-2.0.
"""

from typing import List

import httpx
import ollama
from ollama import ChatResponse

from chatbot_conversation.models.base import ChatbotBase, ConversationMessage
from chatbot_conversation.models.bot_registry import register_bot

# Model temperature range specifically for Ollama API
# Overrides the base class range of 0.0-2.0
OLLAMA_MIN_MODEL_TEMP = 0.0
OLLAMA_MAX_MODEL_TEMP = 1.0
OLLAMA_DEFAULT_TEMP = 0.8

MODEL_TYPE = "OLLAMA"


@register_bot(MODEL_TYPE)
class OllamaChatbot(ChatbotBase):
    """
    Chatbot implementation using Ollama's API service.

    Provides a concrete implementation of ChatbotBase for Ollama models,
    extending core functionality with Ollama-specific API integration.

    Features:
    - Local LLM deployment support
    - Ollama-specific temperature range (0.0-1.0)
    - Network resilience with automatic retries
    - HTTP error recovery
    - Configurable system prompts
    - Stateful conversation handling

    Args:
        config (ChatbotConfig): Configuration object containing:
            - name: Bot instance identifier
            - system_prompt: Initial system behavior instructions
            - model: Model type, version and parameters
            - timeout: API communication settings

    Attributes:
        Inherits all attributes from ChatbotBase
        No additional attributes required for Ollama implementation

    Notes:
        Uses local API communication (default: http://localhost:11434)
    """

    # no __init__() method needed, OllamaChatbot uses the base class __init__()
    # which is automatically called when creating an instance of this class

    def _get_model_type(self) -> str:
        """
        Get the model type identifier for Ollama models.

        Returns:
            str: "OLLAMA" as the model type identifier
        """
        return MODEL_TYPE

    def _get_default_temperature(self) -> float:
        """
        Get the default temperature setting for Ollama models.

        Returns:
            float: Default temperature value (0.8) for Ollama response generation

        Note:
            Ollama uses a 0.0-1.0 temperature range unlike most other APIs
        """
        return OLLAMA_DEFAULT_TEMP

    def _should_retry_on_exception(self, exception: Exception) -> bool:
        """
        Determine if an API call should be retried based on Ollama-specific exceptions.

        Handles common Ollama API errors that warrant retry attempts:
        - TimeoutException: Connection or read timeout
        - NetworkError: General network connectivity issues
        - HTTPStatusError: Server errors (5xx) or rate limits

        Args:
            exception: The caught exception

        Returns:
            bool: True if retry is recommended, False otherwise
        """
        return isinstance(
            exception,
            (httpx.TimeoutException, httpx.NetworkError, httpx.HTTPStatusError),
        )

    @ChatbotBase.temp.setter  # type: ignore
    def temp(self, value: float) -> None:
        """
        Set the temperature value for response generation.

        Validates and sets the temperature within Ollama's supported range (0.0-1.0).
        Higher values increase response randomness, lower values make responses
        more deterministic.

        Args:
            value (float): Temperature value between 0.0 and 1.0

        Raises:
            ValueError: If temperature is outside Ollama's valid range
        """
        if not OLLAMA_MIN_MODEL_TEMP <= value <= OLLAMA_MAX_MODEL_TEMP:
            raise ValueError(f"Ollama temperature {value} must be between 0.0 and 1.0")
        self._temp = value

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
        formatted_messages = self._format_conv_for_api_util(conversation)
        response: ChatResponse = (
            ollama.chat(  # pyright: ignore[reportUnknownMemberType]
                model=self.model_version,
                messages=formatted_messages,
                options={"temperature": self.temp},
            )
        )
        response_content = response["message"]["content"]
        return response_content
