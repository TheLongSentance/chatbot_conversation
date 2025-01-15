"""
A concrete implementation of the ChatbotBase class that integrates with OpenAI's GPT API service.

This module provides the OpenAIChatbot class which handles all interactions with OpenAI's API,
including message formatting, API calls, and error handling. It supports various GPT models
and configurable parameters for response generation.

Key Features:
- OpenAI API client management
- Message formatting for GPT models
- Configurable temperature and timeout settings
- Error handling and retry logic
- Support for system prompts and conversation history

Classes:
    OpenAIChatbot: A chatbot implementation using OpenAI's GPT models.
"""

from typing import List

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
class OpenAIChatbot(ChatbotBase):
    """
    A chatbot implementation that uses OpenAI's GPT models for generating responses.

    This class manages the interaction with OpenAI's API, handling authentication,
    message formatting, and response generation. It supports customizable parameters
    like model version and temperature for fine-tuning the bot's behavior.

    Attributes:
        api (OpenAI): Initialized OpenAI client instance.
        model_version (str): The GPT model version to use (e.g., "gpt-4", "gpt-3.5-turbo").
        system_prompt (str): Initial instruction that defines the bot's behavior and role.
        temp (float): Temperature parameter for response generation (0.0-2.0).
        timeout (TimeoutConfig): Configuration for API timeouts.
    """

    def __init__(
        self,
        config: ChatbotConfig,
    ) -> None:
        """
        Initialize a new OpenAIChatbot instance.

        Args:
            bot_name (str): Identifier name for the chatbot instance.
            bot_system_prompt (str): Initial instruction defining the bot's behavior and role.
            bot_model_version (str): GPT model version to use (e.g., "gpt-4").
            bot_temp (float, optional): Temperature for response generation. Defaults to 1.0.
                - 0.0: Focused, deterministic responses
                - 1.0: Balanced creativity and coherence
                - 2.0: Maximum creativity and variation

        Note:
            The OpenAI API key should be set in the environment variables.
        """
        super().__init__(config)

        self.api = OpenAI()

    def _get_model_type(self) -> str:
        """
        Get the model type identifier for the chatbot.

        Returns:
            str: The model type identifier for the chatbot.
        """
        return MODEL_TYPE

    def _get_default_temperature(self) -> float:
        """
        Return the default temperature setting for OpenAI GPT models.

        Returns:
            float: Default temperature value (1.0) for OpenAI GPT response generation
        """
        return OPENAI_DEFAULT_TEMP

    def _should_retry_on_exception(self, exception: Exception) -> bool:
        """
        Determine if an API call should be retried based on the exception type.

        Evaluates whether the encountered exception is temporary (like network issues
        or rate limits) and thus suitable for a retry attempt.

        Args:
            exception (Exception): The caught exception to evaluate.

        Returns:
            bool: True if the operation should be retried, False otherwise.

        Note:
            Currently handles APIError, APIConnectionError, and RateLimitError as
            retry-able exceptions.
        """
        return isinstance(exception, (APIError, APIConnectionError, RateLimitError))

    def _generate_response(self, conversation: List[ConversationMessage]) -> str:
        """
        Generate a response using OpenAI's chat completion API.

        Formats the conversation history and sends it to OpenAI's API for processing.
        Handles the API interaction and extracts the generated response.

        Args:
            conversation (List[ConversationMessage]): The conversation history,
                including system prompt, user messages, and assistant responses.

        Returns:
            str: The generated response text from the GPT model.

        Note:
            Uses the configured model version, temperature, and timeout settings
            when making the API call.
        """
        response_content: str = ""
        formatted_messages = self._format_conv_for_api_util(conversation)
        completion = self.model_api.chat.completions.create(
            model=self.model_version,
            messages=formatted_messages,
            timeout=self.model_timeout.api_timeout,
            temperature=self.model_temperature,
        )
        response_content = completion.choices[0].message.content
        return response_content
