"""
Anthropic Claude API integration for chatbot functionality.

This module provides the ClaudeChatbot class, implementing ChatbotBase for Claude AI.
It handles API communication, message formatting, and conversation management with
configurable parameters for temperature and system prompts.

Classes:
    ClaudeChatbot: Claude API chatbot implementation
"""

from typing import List, Optional

import anthropic
from anthropic import APIConnectionError, APIError, RateLimitError

from chatbot_conversation.models.base import ChatbotBase, ConversationMessage
from chatbot_conversation.models.bot_registry import register_bot

# Default temperature for Claude models
# Inherits range from 0.0 to 2.0 from the base class
# Other specify in the config file for a specific model
CLAUDE_DEFAULT_TEMP = 1.0


@register_bot("CLAUDE")
class ClaudeChatbot(ChatbotBase):
    """
    Chatbot implementation using Anthropic's Claude API.

    Provides conversation capabilities through Claude's language models with
    configurable temperature and system prompts. Handles API communication,
    retries, and message formatting.

    Args:
        bot_name: Identifier for the bot instance
        bot_system_prompt: Instructions controlling bot behavior
        bot_model_version: Claude model version to use
        bot_temp: Temperature setting (0.0-2.0, default: 1.0)

    Attributes:
        api: Authenticated Claude API client
        model_version: Active model identifier
        system_prompt: System behavior instructions
        temp: Response randomness parameter
    """

    def __init__(
        self,
        bot_name: str,
        bot_system_prompt: str,
        bot_model_version: str,
        bot_temp: Optional[float] = None,
    ) -> None:
        """
        Initialize Claude chatbot with specified configuration.

        Args:
            bot_name: Identifier for this bot instance
            bot_system_prompt: Instructions for bot behavior
            bot_model_version: Claude model version to use
            bot_temp: Temperature parameter (0.0-2.0)
        """
        super().__init__(  # pylint: disable=duplicate-code
            bot_name=bot_name,
            bot_system_prompt=bot_system_prompt,
            bot_model_version=bot_model_version,
            bot_temp=bot_temp,
        )

        # Initialise Claude API
        self.api = anthropic.Anthropic()

    def _get_default_temperature(self) -> float:
        """
        Return the default temperature setting for Claude models.

        Returns:
            float: Default temperature value (1.0) for Claude response generation
        """
        return CLAUDE_DEFAULT_TEMP

    def _should_retry_on_exception(self, exception: Exception) -> bool:
        """
        Determine if an API call should be retried based on the exception type.

        Args:
            exception: The caught exception

        Returns:
            True if retry is recommended, False otherwise
        """
        return isinstance(exception, (APIError, APIConnectionError, RateLimitError))

    def _generate_response(self, conversation: List[ConversationMessage]) -> str:
        """
        Generate a response using the Claude API.

        Args:
            conversation: List of previous messages in the conversation

        Returns:
            Generated response text from Claude

        Raises:
            APIError: On API communication errors
            RateLimitError: When API rate limits are exceeded
            TimeoutError: When API call exceeds timeout
        """
        response_content: str = ""
        formatted_messages = self._format_conv_for_api_util(
            conversation, add_system_prompt=False
        )
        message = self.api.messages.create(
            model=self.model_version,
            system=self.system_prompt,
            messages=formatted_messages,
            max_tokens=500,
            timeout=self.timeout.api_timeout,
            temperature=self.temp,
        )
        response_content = message.content[0].text
        return response_content
