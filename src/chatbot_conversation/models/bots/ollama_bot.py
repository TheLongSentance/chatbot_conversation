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

from typing import List

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

    def __init__(
        self,
        bot_model_version: str,
        bot_system_prompt: str,
        bot_name: str,
    ) -> None:
        """
        Initialize the OllamaChatbot with model version, system prompt, and bot name.

        Args:
            bot_model_version (str): The version of the bot model
            bot_system_prompt (str): The system prompt for the bot
            bot_name (str): The name of the bot
        """
        super().__init__(
            bot_model_version=bot_model_version,
            bot_system_prompt=bot_system_prompt,
            bot_name=bot_name,
        )

        # Ollama doesn't need specific __init__ implementation
        # but we can add any future specific initialization here

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
