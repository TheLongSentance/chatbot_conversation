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

    def generate_response(self, conversation: List[ConversationMessage]) -> str:
        """
        Generate response using Claude's chat model.

        Args:
            conversation (List[ConversationMessage]): The conversation history.

        Returns:
            str: The response from the Claude model.
        """
        formatted_messages = self._format_conv_for_api_util(
            conversation, add_system_prompt=False
        )
        try:
            message = self.api.messages.create(
                model=self.model_version,
                system=self.system_prompt,
                messages=formatted_messages,
                max_tokens=500,
                timeout=10,
            )
        except anthropic.AnthropicError as e:
            response_content = f"Exception: Anthropic Claude API error generating response: {e}"
            self.log_error(response_content) 
            return response_content
        else:
            try:
                response_content = message.content[0].text
                if response_content is None or response_content == "":
                    raise ValueError("Text is empty")
            except IndexError as e:
                # Handle the case where message.content is empty
                response_content = f"Exception: message.content[0].text from Claude API is empty: {e}"
                self.log_error(response_content)
                return response_content
            except AttributeError as e:
                # Handle the case where message.content[0] does not have a text attribute
                response_content = f"Exception: message.content[0] does not have a .text attribute: {e}"
                self.log_error(response_content)
                return response_content
            except ValueError as e:
                # Handle the case where message.content[0].text is empty
                response_content = f"Exception: message.content[0].text from Claude API is empty: {e}]"
                self.log_error(response_content)
                return response_content

        return response_content
