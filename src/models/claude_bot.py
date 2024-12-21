"""
This module contains the ClaudeChatbot class, a concrete implementation of the ChatbotBase class,
which uses Claude's API service to generate responses.

The ClaudeChatbot class handles:
- Initialization of the Claude client
- Formatting messages specific to Claudes's expected format
- Generating responses using the Claude API
"""

import json
from typing import Any, List

import anthropic

from ..utils.logging_util import get_logger
from .base import ChatbotBase, ChatMessage, ConversationMessage

logger = get_logger("models")


class ClaudeChatbot(ChatbotBase[ChatMessage]):
    """Concrete implementation of chatbot using Claude's API service.

    Handles initialization of Claude client, message formatting specific to Claude's
    expected format, and response generation using the Claude model.

    Attributes:
        api: Claude client instance
        model_version: Version of Claude model to use
        system_prompt: System instruction for bot behavior
    """

    def _initialize_api(self) -> Any:
        """Initialize connection to Claude API.

        Returns:
            Claude: Configured Claude client instance
        """
        return anthropic.Anthropic()

    def _generate_raw_response(self, conversation: List[ConversationMessage]) -> str:
        """Generate raw response using Claude's chat model."""
        formatted_messages = self._format_message(conversation)
        message = self.api.messages.create(
            model=self.model_version,
            system=self.system_prompt,
            messages=formatted_messages,
            max_tokens=500,
            timeout=10,
        )
        response_content = message.content[0].text
        if not isinstance(response_content, str):
            raise ValueError("Expected response content to be a string")
        return response_content

    def _format_message(
        self, conversation: List[ConversationMessage]
    ) -> List[ChatMessage]:
        """Format message history for Claude API submission.

        Formats all messages according to Claude's expected structure.
        System prompt is not included in the message list for Claude.

        Args:
            conversation: List of conversation messages to format

        Returns:
            List[ChatMessage]: Messages formatted for Claude API
        """

        messages: List[ChatMessage] = []

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
