"""
Ollama chatbot integration module.

Provides a ChatbotBase implementation for interacting with Ollama's API service.
Handles model interactions, conversation management, and response generation with
configurable temperature settings. Includes automatic retry logic for network
issues and proper message formatting for the Ollama API.
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
    Ollama API chatbot implementation.

    Manages interactions with Ollama language models through their API service.
    Provides conversation handling, response generation, and error recovery.
    Supports temperature-based response variation within Ollama's 0.0-1.0 range.

    Inherits from ChatbotBase to maintain consistent interface across bot implementations.
    """

    # no __init__() method needed, OllamaChatbot uses the base class __init__()
    # which is automatically called when creating an instance of this class

    def _get_model_type(self) -> str:
        """
        Get the model type identifier for the chatbot.

        Returns:
            str: The model type identifier for the chatbot.
        """
        return MODEL_TYPE

    def _get_default_temperature(self) -> float:
        """
        Return the default temperature setting for Ollama models.

        Returns:
            float: Default temperature value (0.8) for Ollama response generation
        """
        return OLLAMA_DEFAULT_TEMP

    def _should_retry_on_exception(self, exception: Exception) -> bool:
        """
        Evaluate if an operation should be retried based on the exception type.

        Determines if the given exception indicates a recoverable error
        (like network timeouts) that warrants a retry attempt.

        Args:
            exception (Exception): The exception to evaluate

        Returns:
            bool: True if the operation should be retried, False otherwise
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
        Generate a model response for the given conversation.

        Processes the conversation history through the Ollama API to generate
        a contextually appropriate response using the configured model and
        temperature settings.

        Args:
            conversation (List[ConversationMessage]): Conversation history to process

        Returns:
            str: Generated response text from the Ollama model
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
