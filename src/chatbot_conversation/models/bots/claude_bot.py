"""
This module contains the ClaudeChatbot class, a concrete implementation of the ChatbotBase class,
which uses Claude's API service to generate responses.

The ClaudeChatbot class handles:
- Initialization of the Claude client
- Formatting messages specific to Claude's expected format
- Generating responses using the Claude API

Classes:
    ClaudeChatbot: Concrete implementation of chatbot using Claude's API service.
"""

from typing import Any, List

import anthropic

from chatbot_conversation.models.base import ChatbotBase, ConversationMessage
from chatbot_conversation.models.bot_registry import register_bot


@register_bot("CLAUDE")
class ClaudeChatbot(ChatbotBase):
    """
    Concrete implementation of chatbot using Claude's API service.

    Handles initialization of Claude client, message formatting specific to Claude's
    expected format, and response generation using the Claude model.

    Attributes:
        api: Claude client instance.
        model_version: Version of Claude model to use.
        system_prompt: System instruction for bot behavior.
    """

    def _initialize_api(self) -> Any:
        """
        Initialize connection to Claude API.

        Returns:
            Claude: Configured Claude client instance.
        """
        return anthropic.Anthropic()

    def _generate_response(self, conversation: List[ConversationMessage]) -> str:
        """
        Private method to generate response using Claude's chat model.

        Args:
            conversation (List[ConversationMessage]): The conversation history.

        Returns:
            str: The response from the Claude model.
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
            timeout=10,
        )
        response_content = message.content[0].text
        return response_content
