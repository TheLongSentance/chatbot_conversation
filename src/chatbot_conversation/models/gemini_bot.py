"""
This module contains the GeminiChatbot class, a concrete implementation of the ChatbotBase class,
which uses Google's Gemini API service to generate responses.

The GeminiChatbot class handles:
- Initialization of the Gemini client
- Formatting messages specific to Gemini's expected format
- Generating responses using the Gemini API
"""

import asyncio
import json
from concurrent.futures import TimeoutError as FuturesTimeoutError
from typing import Any, List, TypedDict

import google.generativeai

from chatbot_conversation.models.base import ChatbotBase, ConversationMessage
from chatbot_conversation.utils.logging_util import get_logger


class _GeminiMessage(TypedDict):
    """Represents a Gemini-specific message with a role and parts."""

    role: str
    parts: str


logger = get_logger("models")


class GeminiChatbot(ChatbotBase):
    """Concrete implementation of chatbot using Google's Gemini API service.

    Handles initialization of Gemini model with system prompt during setup,
    message formatting specific to Gemini's expected format using 'parts' instead
    of 'content', and response generation.
    """

    def _initialize_api(self) -> Any:
        """Initialize connection to Gemini API with system prompt."""
        google.generativeai.configure()
        return google.generativeai.GenerativeModel(
            model_name=self.model_version, system_instruction=self.system_prompt
        )

    async def _generate_with_timeout(
        self, formatted_messages: List[_GeminiMessage], timeout: int = 30
    ) -> str:
        """Wrapper to call Gemini API with timeout."""
        try:
            message = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(
                    None, lambda: self.api.generate_content(formatted_messages)
                ),
                timeout=timeout,
            )
            if not isinstance(message.text, str):
                raise TypeError("Expected message.text to be a string")
            return message.text
        except FuturesTimeoutError as error:
            raise FuturesTimeoutError(
                f"Gemini API call timed out after {timeout} seconds"
            ) from error

    def _generate_raw_response(self, conversation: List[ConversationMessage]) -> str:
        """Generate raw response using Gemini model with timeout."""
        formatted_messages = self._format_conv_for_gemini_api(conversation)
        response = asyncio.run(self._generate_with_timeout(formatted_messages))
        return response

    def _format_conv_for_gemini_api(
        self, conversation: List[ConversationMessage]
    ) -> List[_GeminiMessage]:
        """Format message history for Gemini API submission."""
        messages: List[_GeminiMessage] = []

        for contribution in conversation:
            role = "model" if contribution["bot_index"] == self.bot_index else "user"
            messages.append({"role": role, "parts": contribution["content"]})

        logger.debug(
            "Bot Class: %s, Bot Name: %s, Bot Index: %s, Formatted Messages: %s",
            self.__class__.__name__,
            self.name,
            self.bot_index,
            json.dumps(messages, indent=2),
        )

        return messages
