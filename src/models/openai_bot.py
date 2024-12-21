"""
This module contains the OpenAIChatbot class, a concrete implementation of the ChatbotBase class,
which uses OpenAI's API service to generate responses using the GPT model.

The OpenAIChatbot class handles:
- Initialization of the OpenAI client
- Formatting messages specific to OpenAI's expected format
- Generating responses using the GPT model
"""

import json
from typing import Any, List

from openai import OpenAI

from ..utils.logging_util import get_logger
from .base import ChatbotBase, ChatMessage, ConversationMessage

logger = get_logger("models")


class OpenAIChatbot(ChatbotBase[ChatMessage]):
    """Concrete implementation of chatbot using OpenAI's API service.

    Handles initialization of OpenAI client, message formatting specific to OpenAI's
    expected format, and response generation using the GPT model.

    Attributes:
        api: OpenAI client instance
        model_version: Version of GPT model to use
        system_prompt: System instruction for bot behavior
    """

    def _initialize_api(self) -> Any:
        """Initialize connection to OpenAI API.

        Returns:
            OpenAI: Configured OpenAI client instance
        """
        return OpenAI()

    def _generate_raw_response(self, conversation: List[ConversationMessage]) -> str:
        """Generate raw response using OpenAI's chat completion.

        Args:
            conversation (List[ConversationMessage]): List of conversation messages

        Returns:
            str: Generated response from the model
        """
        formatted_messages = self._format_message(conversation)
        completion = self.api.chat.completions.create(
            model=self.model_version, messages=formatted_messages, timeout=10
        )
        response_content = completion.choices[0].message.content
        if not isinstance(response_content, str):
            raise ValueError("Expected response content to be a string")
        return response_content

    def _format_message(
        self, conversation: List[ConversationMessage]
    ) -> List[ChatMessage]:
        """Format message history for OpenAI API submission.

        Prepends system prompt and formats all messages according to
        OpenAI's expected structure.

        Args:
            conversation (List[ConversationMessage]): List of conversation messages to format

        Returns:
            List[ChatMessage]: Messages formatted for OpenAI API
        """
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
