"""
This module contains the OpenAIChatbot class, a concrete implementation of the ChatbotBase class,
which uses OpenAI's API service to generate responses using the GPT model.

The OpenAIChatbot class handles:
- Initialization of the OpenAI client
- Formatting messages specific to OpenAI's expected format
- Generating responses using the GPT model

Classes:
    OpenAIChatbot: Concrete implementation of chatbot using OpenAI's API service.
"""

from typing import Any, List

from openai import OpenAI

from chatbot_conversation.models.base import ChatbotBase, ConversationMessage


class OpenAIChatbot(ChatbotBase):
    """
    Concrete implementation of chatbot using OpenAI's API service.

    Handles initialization of OpenAI client, message formatting specific to OpenAI's
    expected format, and response generation using the GPT model.

    Attributes:
        api: OpenAI client instance.
        model_version: Version of GPT model to use.
        system_prompt: System instruction for bot behavior.
    """

    def _initialize_api(self) -> Any:
        """
        Initialize connection to OpenAI API.

        Returns:
            OpenAI: Configured OpenAI client instance.
        """
        return OpenAI()

    def generate_response(self, conversation: List[ConversationMessage]) -> str:
        """
        Generate response using OpenAI's chat completion.

        Args:
            conversation (List[ConversationMessage]): List of conversation messages.

        Returns:
            str: Generated response from the model.
        """
        formatted_messages = self._format_conv_for_api_util(conversation)
        completion = self.api.chat.completions.create(
            model=self.model_version, messages=formatted_messages, timeout=10
        )
        response_content = completion.choices[0].message.content
        if not isinstance(response_content, str):
            raise ValueError("Expected response content to be a string")
        return response_content
