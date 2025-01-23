"""
Anthropic Claude API integration for chatbot functionality.

Provides a concrete implementation of ChatbotBase for Claude AI models,
handling API communication, message formatting, and conversation management
with configurable parameters.

Major Classes:
    ClaudeChatbot: Claude-specific chatbot implementation
"""

from typing import List, Any, Iterator

import anthropic
from anthropic import APIConnectionError, APIError, RateLimitError

from chatbot_conversation.models.base import (
    ChatbotBase,
    ChatbotConfig,
    ConversationMessage,
)
from chatbot_conversation.models.bot_registry import register_bot

# Default temperature for Claude models
# Inherits range from 0.0 to 2.0 from the base class
# Other specify in the config file for a specific model
CLAUDE_DEFAULT_TEMP = 1.0

MODEL_TYPE = "CLAUDE"


@register_bot(MODEL_TYPE)
class ClaudeChatbot(ChatbotBase):
    """
    Chatbot implementation using Anthropic's Claude API.

    Provides a concrete implementation of ChatbotBase for Claude AI models,
    extending core functionality with Claude-specific API integration.

    Features:
    - Claude API authentication and communication
    - Message formatting for Claude's API requirements
    - Temperature-controlled response generation
    - Automatic retry handling for API failures
    - Configurable system prompts and timeouts

    Args:
        config (ChatbotConfig): Configuration object containing:
            - name: Bot instance identifier
            - system_prompt: Initial system behavior instructions
            - model: Model type, version and parameters
            - timeout: API communication settings

    Attributes:
        Inherits all attributes from ChatbotBase plus:
        model_api (anthropic.Anthropic): Authenticated Claude API client
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
        Initialize Claude chatbot with specified configuration.

        Validates configuration and sets up Claude API client.

        Args:
            config (ChatbotConfig): Complete bot configuration
        """
        super().__init__(config)

        # Initialise Claude API
        self.model_api = anthropic.Anthropic()

    @property
    def _default_temperature(self) -> float:
        """Default temperature override"""
        return CLAUDE_DEFAULT_TEMP

    def _should_retry_on_exception(self, exception: Exception) -> bool:
        """
        Determine if an API call should be retried based on Claude-specific exceptions.

        Handles common Claude API errors that warrant retry attempts:
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
        Generate a response using the Claude API.

        Formats conversation history and makes API call with configured parameters.
        Handles message structure requirements specific to Claude's API.

        Args:
            conversation: Sequential list of prior conversation messages

        Returns:
            str: Generated response text from Claude

        Raises:
            APIError: On API communication errors
            RateLimitError: When API rate limits are exceeded
            TimeoutError: When API call exceeds timeout
        """
        message = self.model_api.messages.create(
            model=self.model_version,
            system=self.system_prompt,
            messages=self._format_conv_for_api_util(
                conversation, add_system_prompt=False
            ),
            timeout=self.model_timeout.api_timeout,
            max_tokens=self.model_max_tokens,
            temperature=self.model_temperature,
        )
        response_content = message.content[0].text
        return response_content

    def _get_text_from_chunk(self, chunk: Any) -> str:
        """
        Extract text content from a streaming response chunk.

        Args:
            chunk (Any): Response chunk from Claude streaming API

        Returns:
            str: Extracted text content from the chunk, or empty string if chunk is None
        """
        return chunk or ""

    def _generate_stream(self, conversation: list[ConversationMessage]) -> Iterator[Any]:
        """
        Generate streaming responses using the Claude API.

        Args:
            conversation (list[ConversationMessage]): List of conversation messages

        Returns:
            Iterator[Any]: Iterator yielding response chunks from Claude's text stream

        Note:
            Uses Claude's streaming mode for real-time response generation
        """
        stream_manager = self.model_api.messages.stream(
            model=self.model_version,
            system=self.system_prompt,
            messages=self._format_conv_for_api_util(
                conversation, add_system_prompt=False
            ),
            timeout=self.model_timeout.api_timeout,
            max_tokens=self.model_max_tokens,
            temperature=self.model_temperature,
        )
        with stream_manager as stream:
            # Consume stream inside context manager
            for chunk in stream.text_stream:
                yield chunk
