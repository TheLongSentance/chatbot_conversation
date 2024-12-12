from typing import List, Any
import ollama  # type: ignore
from .base import ChatbotBase, ChatMessage, ConversationMessage

class OllamaChatbot(ChatbotBase[ChatMessage]):
    """Concrete implementation of chatbot using Ollama's API service.
    
    Handles initialization of Ollama client, message formatting specific to Ollama's
    expected format, and response generation.
    """

    def __init__(self, model_version: str, system_prompt: str, name: str):
        """Initialize Ollama chatbot with specific model and behavior."""
        super().__init__(model_version, system_prompt, name)
    
    def _initialize_api(self) -> Any:
        """Initialize connection to Ollama API."""
        return None  # Ollama doesn't need initialization
        
    def generate_response(self, conversation: List[ConversationMessage]) -> str:
        """Generate next response using Ollama's chat model."""
        formatted_messages = self._format_message(conversation)
        
        try:
            response = ollama.chat(                 # type: ignore
                model=self.model_version,
                messages=formatted_messages
            )
            response = response['message']['content'] # type: ignore
            return f"{self.name}: {response}"
        except Exception as e:
            print(f"Error calling Ollama model: {e}")
            return f"{self.name}: Error: Unable to generate response from Ollama model."

    def _format_message(self, conversation: List[ConversationMessage]) -> List[ChatMessage]:
        """Format message history for Ollama API submission."""
        messages: List[ChatMessage] = [{"role": "system", "content": self.system_prompt}]

        for contribution in conversation:
            role = "assistant" if contribution["bot_index"] == self.bot_index else "user"
            messages.append({"role": role, "content": contribution["content"]})

        return messages