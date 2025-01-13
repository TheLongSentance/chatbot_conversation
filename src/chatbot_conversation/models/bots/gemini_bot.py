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
from typing import List, TypedDict

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

    def __init__(
        self,
        bot_name: str,
        bot_system_prompt: str,
        bot_model_version: str,
        bot_temp: float = 0.7,
    ) -> None:
        """
        Initialize the GeminiChatbot with model version, system prompt, and bot name.

        Args:
            bot_model_version (str): The version of the bot model
            bot_system_prompt (str): The system prompt for the bot
            bot_name (str): The name of the bot
        """
        super().__init__(  # pylint: disable=duplicate-code
            bot_name=bot_name,
            bot_system_prompt=bot_system_prompt,
            bot_model_version=bot_model_version,
            bot_temp=bot_temp,
        )

        # no stub file from google.generativeai so ignore for pylance (-> pyright) etc
        google.generativeai.configure()  # pyright: ignore[reportUnknownMemberType]

        # initialise api here, but will be updated in _generate_response
        # when system prompt is set or updated since it is not passed in
        # the generate_content call for Gemini as either a parameter or
        # part of the message history

        self.api = google.generativeai.GenerativeModel(
            model_name=self.model_version,
            system_instruction=self.system_prompt,
            generation_config={"temperature": self.temp},
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

        # test if system prompt has changed and re-initialize API in order
        # to reset the system prompt for Gemini API. This is not typical
        # for other models as they include system prompt in either:
        # - as a parameter in the api call (e.g. Claude)
        # - or as part of the message history (e.g. OpenAI, Ollama)
        # for Gemini, this will happen when the system prompt is first set
        # and whenever it is updated (first round, after first round, before last)

        if self.system_prompt_needs_update:
            self.api = google.generativeai.GenerativeModel(
                model_name=self.model_version,
                system_instruction=self.system_prompt,
                generation_config={"temperature": self.temp},
            )
            self.system_prompt_updated()

        message = self.api.generate_content(formatted_messages)
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
