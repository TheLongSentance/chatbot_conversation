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

    def _generate_response(self, conversation: List[ConversationMessage]) -> str:
        """
        Private method to generate response using OpenAI's chat completion.

        Args:
            conversation (List[ConversationMessage]): List of conversation messages.

        Returns:
            str: Generated response from the model.
        """
        response_content: str = ""

        formatted_messages = self._format_conv_for_api_util(conversation)

        try:
            completion = self.api.chat.completions.create(
                model=self.model_version, messages=formatted_messages, timeout=10
            )
            response_content = completion.choices[0].message.content
            if response_content == "":
                raise ValueError("Text is empty")
        except IndexError as e:
            response_content = f"Exception: completion.choices[0] is empty: {e}"
            self.log_error(response_content)
            return response_content
        except AttributeError as e:
            response_content = (
                f"Exception: missing attribute in completion.choices[0].message.content: {e}"
            )
            self.log_error(response_content)
            return response_content
        except ValueError as e:
            response_content = (
                f"Exception: completion.choices[0].message.content is empty: {e}"
            )
            self.log_error(response_content)
            return response_content
        except Exception as e:
            response_content = f"Exception: OpenAI API error generating response: {e}"
            self.log_error(response_content)
            return response_content

        return response_content
