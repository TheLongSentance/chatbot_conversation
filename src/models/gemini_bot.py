from typing import List, Any
import google.generativeai  # type: ignore
from .base import ChatbotBase, GeminiMessage, ConversationMessage

class GeminiChatbot(ChatbotBase[GeminiMessage]):
    """Concrete implementation of chatbot using Google's Gemini API service.
    
    Handles initialization of Gemini model with system prompt during setup,
    message formatting specific to Gemini's expected format using 'parts' instead
    of 'content', and response generation.
    """

    def __init__(self, model_version: str, system_prompt: str, name: str):
        """Initialize Gemini chatbot with specific model and behavior."""
        super().__init__(model_version, system_prompt, name)
    
    def _initialize_api(self) -> Any:
        """Initialize connection to Gemini API with system prompt."""
        google.generativeai.configure() # type: ignore
        return google.generativeai.GenerativeModel(
            model_name=self.model_version,
            system_instruction=self.system_prompt
        )
        
    def generate_response(self, conversation: List[ConversationMessage]) -> str:
        """Generate next response using Gemini model."""
        formatted_messages = self._format_message(conversation)
        
        try:
            message = self.api.generate_content(formatted_messages) # type: ignore
            response = message.text
            return f"{self.name}: {response}"
        except Exception as e:
            print(f"Error calling Gemini model: {e}")
            return f"{self.name}: Error: Unable to generate response from Gemini model."

    def _format_message(self, conversation: List[ConversationMessage]) -> List[GeminiMessage]:
        """Format message history for Gemini API submission."""
        messages: List[GeminiMessage] = []

        for contribution in conversation:
            role = "model" if contribution["bot_index"] == self.bot_index else "user"
            messages.append({"role": role, "parts": contribution["content"]})

        return messages
