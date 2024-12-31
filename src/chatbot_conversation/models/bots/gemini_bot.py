"""
This module contains the GeminiChatbot class, a concrete implementation of the ChatbotBase class,
which uses Google's Gemini API service to generate responses.

The GeminiChatbot class handles:
- Initialization of the Gemini client
- Formatting messages specific to Gemini's expected format
- Generating responses using the Gemini API

Classes:
    _GeminiMessage: Represents a Gemini-specific message with a role and parts.
    GeminiChatbot: Concrete implementation of chatbot using Google's Gemini API service.
"""

import asyncio
import json
from typing import Any, List, TypedDict

# no stub file from google.generativeai so ignore for pylance etc
import google.generativeai  # type: ignore

from chatbot_conversation.models.base import ChatbotBase, ConversationMessage
from chatbot_conversation.models.bot_registry import register_bot
from chatbot_conversation.utils import get_logger


class _GeminiMessage(TypedDict):
    """Represents a Gemini-specific message with a role and parts."""

    role: str
    parts: str


logger = get_logger("models")


@register_bot("GEMINI")
class GeminiChatbot(ChatbotBase):
    """
    Concrete implementation of chatbot using Google's Gemini API service.

    Handles initialization of Gemini model with system prompt during setup,
    message formatting specific to Gemini's expected format using 'parts' instead
    of 'content', and response generation.
    """

    def _initialize_api(self) -> Any:
        """
        Initialize connection to Gemini API with system prompt.

        Returns:
            Any: Initialized Gemini API client.
        """
        # no stub file from google.generativeai so ignore for pylance etc
        google.generativeai.configure()  # type: ignore
        return google.generativeai.GenerativeModel(
            model_name=self.model_version, system_instruction=self.system_prompt
        )

    async def _generate_with_timeout(
        self, formatted_messages: List[_GeminiMessage], timeout: int = 30
    ) -> str:
        """
        Wrapper to call Gemini API with timeout.

        Args:
            formatted_messages (List[_GeminiMessage]): Formatted messages for the API.
            timeout (int, optional): Timeout in seconds. Defaults to 30.

        Returns:
            str: The response text from the Gemini API.
        """
        response_content: str = ""
        message = await asyncio.wait_for(
            asyncio.get_event_loop().run_in_executor(
                None, lambda: self.api.generate_content(formatted_messages)
            ),
            timeout=timeout,
        )
        response_content = message.text
        return response_content

    def _generate_response(self, conversation: List[ConversationMessage]) -> str:
        """
        Private method to generate response using Gemini model with timeout.

        Args:
            conversation (List[ConversationMessage]): The conversation history.

        Returns:
            str: The response from the Gemini model.
        """
        formatted_messages = self._format_conv_for_gemini_api(conversation)
        response = asyncio.run(self._generate_with_timeout(formatted_messages))
        return response

    def _format_conv_for_gemini_api(
        self, conversation: List[ConversationMessage]
    ) -> List[_GeminiMessage]:
        """
        Format message history for Gemini API submission.

        Args:
            conversation (List[ConversationMessage]): List of conversation messages to format.

        Returns:
            List[_GeminiMessage]: Messages formatted for Gemini API.
        """
        messages: List[_GeminiMessage] = []

        for contribution in conversation:
            role = "model" if contribution["bot_index"] == self.bot_index else "user"
            messages.append({"role": role, "parts": contribution["content"]})

        self._log_debug(json.dumps(messages, indent=2))

        return messages
