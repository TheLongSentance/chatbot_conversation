"""
This module contains the OllamaChatbot class, a concrete implementation of the ChatbotBase class,
which uses Ollama's API service to generate responses.

The OllamaChatbot class handles:
- Initialization of the Ollama client
- Formatting messages specific to Ollama's expected format
- Generating responses using the Ollama API
"""

from typing import Any, List

import ollama
from ollama import ChatResponse

from .base import ChatbotBase, ConversationMessage


class OllamaChatbot(ChatbotBase):
    """Concrete implementation of chatbot using Ollama's API service.

    Handles initialization of Ollama client, message formatting specific to Ollama's
    expected format, and response generation.
    """

    def _initialize_api(self) -> Any:
        """Initialize connection to Ollama API."""
        return None  # Ollama doesn't need initialization

    def _generate_raw_response(self, conversation: List[ConversationMessage]) -> str:
        """Generate raw response using Ollama's chat model."""
        formatted_messages = self._format_conv_for_api_util(conversation)
        response: ChatResponse = ollama.chat(
            model=self.model_version, messages=formatted_messages
        )

        message = response["message"]
        if message is None or "content" not in message:
            raise KeyError("Expected 'message' key with 'content' in response")
        response_content = message["content"]
        if not isinstance(response_content, str):
            raise ValueError("Expected response content to be a string")
        return response_content
