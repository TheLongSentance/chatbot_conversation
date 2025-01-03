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

import json
from typing import Any, List, TypedDict

import google.api_core.exceptions

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

        # system prompt is not part of each response unlike most other models
        # so we set it here (for consistency with the intended purpose of the method)
        # but reset self.api each time the _generate_response method is called below
        return google.generativeai.GenerativeModel(
            model_name=self.model_version, system_instruction=self.system_prompt
        )

    def _should_retry_on_exception(self, exception: Exception) -> bool:
        """
        Flags whether a gemini API exception should trigger a retry.

        Returns:
            bool: True if the exception should trigger a retry, False otherwise.
        """
        return isinstance(
            exception,
            (
                google.api_core.exceptions.DeadlineExceeded,
                google.api_core.exceptions.ServiceUnavailable,
            ),
        )

    def _generate_response(self, conversation: List[ConversationMessage]) -> str:
        """
        Private method to generate response using Gemini model.
        No timeout handling is available for Gemini, so handled in ChatbotBase.
        No passing of system prompt in generate_content, so reset self.api each time.

        Args:
            conversation (List[ConversationMessage]): The conversation history.

        Returns:
            str: The response from the Gemini model.
        """
        formatted_messages = self._format_conv_for_gemini_api(conversation)

        # re-initialize the API each time in order to reset the system prompt
        # not typical for other models but necessary for Gemini
        self.api = google.generativeai.GenerativeModel(
            model_name=self.model_version, system_instruction=self.system_prompt
        )

        message = self.api.generate_content(formatted_messages)  # type: ignore
        response: str = message.text
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
