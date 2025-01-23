"""
OpenAI GPT API integration for chatbot functionality.

Provides a concrete implementation of ChatbotBase for OpenAI's GPT models,
handling API communication, message formatting, and conversation management
with configurable parameters.

Major Classes:
    GPTChatbot: GPT-specific chatbot implementation

Supported Models:
    - gpt-4: Latest GPT-4 model
    - gpt-4-turbo: Optimized GPT-4 for faster responses
    - gpt-3.5-turbo: Enhanced GPT-3.5 model
    - gpt-3.5: Base GPT-3.5 model
"""

from typing import Any, Iterator, List

from openai import APIConnectionError, APIError, OpenAI, RateLimitError

from chatbot_conversation.models.base import (
    ChatbotBase,
    ChatbotConfig,
    ConversationMessage,
)
from chatbot_conversation.models.bot_registry import register_bot

# OpenAI default temperature for GPT models
# Inherits range from 0.0 to 2.0 from the base class
# For other temps specify in the config file for a specific model
OPENAI_DEFAULT_TEMP = 1.0

MODEL_TYPE = "GPT"


@register_bot("GPT")
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
        model_api (openai.OpenAI): Authenticated OpenAI API client

    Notes:
        Requires OpenAI API key to be set in environment variables
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
        Initialize OpenAI chatbot with specified configuration.

        Validates configuration and sets up OpenAI API client.
        API key must be available in environment variables.

        Args:
            config (ChatbotConfig): Complete bot configuration
        """
        super().__init__(config)

        self.model_api = OpenAI()

    @property
    def _default_temperature(self) -> float:
        """Default temperature override"""
        return OPENAI_DEFAULT_TEMP

    def _should_retry_on_exception(self, exception: Exception) -> bool:
        """
        Determine if an API call should be retried based on OpenAI-specific exceptions.

        Handles common OpenAI API errors that warrant retry attempts:
        - APIError: General API communication failures
        - APIConnectionError: Network connectivity issues
        - RateLimitError: API quota/throughput limits

        Args:
            exception: The caught exception

        Returns:
            bool: True if retry is recommended, False otherwise
        """
        return isinstance(exception, (APIError, APIConnectionError, RateLimitError))

    def _generate_response(self, conversation: List[ConversationMessage]) -> str:
        """
        Generate a response using the OpenAI API.

        Formats conversation history and makes API call with configured parameters.
        Handles message structure requirements specific to GPT chat models.

        Args:
            conversation: Sequential list of prior conversation messages

        Returns:
            str: Generated response text from GPT

        Raises:
            APIError: On API communication errors
            APIConnectionError: On network connectivity issues
            RateLimitError: When API rate limits are exceeded
        """
        response_content: str = ""
        completion = self.model_api.chat.completions.create(
            model=self.model_version,
            messages=self._format_conv_for_api_util(conversation),
            stream=False,
            timeout=self.model_timeout.api_timeout,
            temperature=self.model_temperature,
            max_tokens=self.model_max_tokens,
        )
        response_content = completion.choices[0].message.content
        return response_content

    def _get_text_from_chunk(
        self, chunk: Any
    ) -> str:  # pyright: ignore[reportUnknownParameterType]
        """
        Extract text content from an OpenAI API streaming chunk.

        Args:
            chunk (Any): A chunk of streaming response from OpenAI API

        Returns:
            str: Extracted text content from the chunk, or empty string if no content
        """
        return chunk.choices[0].delta.content or ""

    def _generate_stream(
        self, conversation: list[ConversationMessage]
    ) -> Iterator[Any]:
        """
        Generate a streaming response using the OpenAI API.

        Creates a streaming completion request with configured parameters,
        allowing for real-time response generation.

        Args:
            conversation (list[ConversationMessage]): Sequential list of prior conversation messages

        Returns:
            Iterator[Any]: Stream of response chunks from the OpenAI API

        Note:
            Uses streaming mode for real-time token generation
            Temperature and model settings are applied as configured
        """
        return self.model_api.chat.completions.create(  #type: ignore
            model=self.model_version,
            messages=self._format_conv_for_api_util(conversation),
            stream=True,
            timeout=self.model_timeout.api_timeout,
            temperature=self.model_temperature,
            max_tokens=self.model_max_tokens,
        )
