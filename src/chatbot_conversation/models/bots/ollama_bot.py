"""
This module contains the OllamaChatbot class, a concrete implementation of the ChatbotBase class,
which uses Ollama's API service to generate responses.

The OllamaChatbot class handles:
- Initialization of the Ollama client
- Formatting messages specific to Ollama's expected format
- Generating responses using the Ollama API

Classes:
    OllamaChatbot: Concrete implementation of chatbot using Ollama's API service.
"""

from typing import Any, List

import httpx
import ollama
from ollama import ChatResponse

from chatbot_conversation.models.base import ChatbotBase, ConversationMessage
from chatbot_conversation.models.bot_registry import register_bot


@register_bot("OLLAMA")
class OllamaChatbot(ChatbotBase):
    """
    Concrete implementation of chatbot using Ollama's API service.

    Handles initialization of Ollama client, message formatting specific to Ollama's
    expected format, and response generation.
    """

    def _initialize_api(self) -> Any:
        """
        Initialize connection to Ollama API.

        Returns:
            None: Ollama doesn't need initialization.
        """
        return None  # Ollama doesn't need initialization

    def _should_retry_on_exception(self, exception: Exception) -> bool:
        """
        Check if the exception is a network error or timeout.

        Args:
            exception (Exception): The exception to check.

        Returns:
            bool: True if the exception is a network error or timeout, False otherwise.
        """
        return isinstance(
            exception,
            (httpx.TimeoutException, httpx.NetworkError, httpx.HTTPStatusError),
        )

    def _generate_response(self, conversation: List[ConversationMessage]) -> str:
        """
        Private method to generate response using Ollama's chat model.
        No timeout handling is available in the Ollama API client.
        Timeout handling is done in ChatbotBase with @retry decorator.

        Args:
            conversation (List[ConversationMessage]): The conversation history.

        Returns:
            str: The response from the Ollama model.
        """
        response_content: str = ""
        formatted_messages = self._format_conv_for_api_util(conversation)
        response: ChatResponse = ollama.chat(  # type: ignore
            model=self.model_version, messages=formatted_messages
        )
        response_content = response["message"]["content"]
        return response_content
