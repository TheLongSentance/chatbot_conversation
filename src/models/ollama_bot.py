"""
This module contains the OllamaChatbot class, a concrete implementation of the ChatbotBase class,
which uses Ollama's API service to generate responses.

The OllamaChatbot class handles:
- Initialization of the Ollama client
- Formatting messages specific to Ollama's expected format
- Generating responses using the Ollama API
"""

import json
from typing import Any, List

import ollama
from ollama import ChatResponse

from ..utils.logging_util import get_logger
from .base import ChatbotBase, ChatMessage, ConversationMessage

logger = get_logger("models")


class OllamaChatbot(ChatbotBase[ChatMessage]):
    """Concrete implementation of chatbot using Ollama's API service.

    Handles initialization of Ollama client, message formatting specific to Ollama's
    expected format, and response generation.
    """

    def _initialize_api(self) -> Any:
        """Initialize connection to Ollama API."""
        return None  # Ollama doesn't need initialization

    def _format_message(
        self, conversation: List[ConversationMessage]
    ) -> List[ChatMessage]:
        """Format message history for Ollama API submission."""
        messages: List[ChatMessage] = [
            {"role": "system", "content": self.system_prompt}
        ]

        for contribution in conversation:
            role = (
                "assistant" if contribution["bot_index"] == self.bot_index else "user"
            )
            messages.append({"role": role, "content": contribution["content"]})

        logger.debug(
            "Bot Class: %s, Bot Name: %s, Bot Index: %s, Formatted Messages: %s",
            self.__class__.__name__,
            self.name,
            self.bot_index,
            json.dumps(messages, indent=2),
        )

        return messages

    def _generate_raw_response(self, conversation: List[ConversationMessage]) -> str:
        """Generate raw response using Ollama's chat model."""
        formatted_messages = self._format_message(conversation)
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
