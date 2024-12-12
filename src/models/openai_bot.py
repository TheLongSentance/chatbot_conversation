from typing import List, Any
from openai import OpenAI
from .base import ChatbotBase, ChatMessage, ConversationMessage

class OpenAIChatbot(ChatbotBase[ChatMessage]):
    """Concrete implementation of chatbot using OpenAI's API service.
    
    Handles initialization of OpenAI client, message formatting specific to OpenAI's
    expected format, and response generation using the GPT model.
    
    Attributes:
        api: OpenAI client instance
        model_version: Version of GPT model to use
        system_prompt: System instruction for bot behavior
    """

    def __init__(self, model_version: str, system_prompt: str, name: str):
        """Initialize OpenAI chatbot with specific model and behavior.

        Args:
            model_version: GPT model version to use (e.g. "gpt-4")
            system_prompt: System instruction defining bot behavior
            name: Name of the chatbot
        """
        super().__init__(model_version, system_prompt, name)
    
    def _initialize_api(self) -> Any:
        """Initialize connection to OpenAI API.

        Returns:
            OpenAI: Configured OpenAI client instance
        """
        return OpenAI()
        
    def generate_response(self, conversation: List[ConversationMessage]) -> str:
        """Generate next response using OpenAI's chat completion.

        Args:
            conversation: List of previous conversation messages

        Returns:
            str: Generated response from GPT model

        Note:
            Includes system prompt at start of every request
        """
        formatted_messages = self._format_message(conversation)
        
        try:
            completion = self.api.chat.completions.create(
                model=self.model_version,
                messages=formatted_messages,
                timeout=10
            )
            response = completion.choices[0].message.content
            return f"{self.name}: {response}"
        except Exception as e:
            print(f"Error calling GPT model: {e}")
            return f"{self.name}: Error: Unable to generate response from GPT model."

    def _format_message(self, conversation: List[ConversationMessage]) -> List[ChatMessage]:
        """Format message history for OpenAI API submission.

        Prepends system prompt and formats all messages according to
        OpenAI's expected structure.

        Args:
            conversation: List of conversation messages to format

        Returns:
            List[ChatMessage]: Messages formatted for OpenAI API
        """
        messages: List[ChatMessage] = [{"role": "system", "content": self.system_prompt}]

        for contribution in conversation:
            role = "assistant" if contribution["bot_index"] == self.bot_index else "user"
            messages.append({"role": role, "content": contribution["content"]})

        return messages
    

