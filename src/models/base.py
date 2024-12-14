from abc import ABC, abstractmethod
from typing import List, Any, TypedDict, TypeVar, Generic
from enum import Enum, auto

class ChatMessage(TypedDict):
    role: str
    content: str

class GeminiMessage(TypedDict):
    role: str
    parts: str

# Define generic model input message type
T = TypeVar('T', ChatMessage, GeminiMessage)

class ConversationMessage(TypedDict):
    bot_index: int
    content: str

class BotType(Enum):
    GPT = auto()
    CLAUDE = auto()
    GEMINI = auto()
    OLLAMA = auto()

class ChatbotBase(ABC, Generic[T]):
    """Abstract base class defining interface for AI chatbot implementations."""
    
    _total_count: int = 0  # Class variable to track total instances

    SYSTEM_PROMPT_TEMPLATE: str = "You are about to take part in a conversation with multiple participants. " \
    "Your identity in this conversation is {name}. Other participants will also have names " \
    "that you can use to refer to them if needed, often in the example form of \"name: \" at the " \
    "start of their contributions to the conversation. Your past contributions to the conversation will also " \
    " be prefixed, in the form \"{name}: \". It is important that you only take on the role of yourself as " \
    "{name} in each response you make. Do not impersonate others in a response and do not make contibutions " \
    "impersonating/imitating multiple roles within a single response i.e. respond only as you in your role. " \
    "It is important not to prefix your response with \"{name}: \" - please do not do this. " \
    "And when refering to other participants, refer to them naturally within your response, you do not have to prefix " \
    "each paragraph within your response with their name. Follow the role instructions supplied in this prompt. {system_prompt}"

    def __init__(self, model_version: str, system_prompt: str, name: str):
        self.model_version: str = model_version
        self.system_prompt: str = self.SYSTEM_PROMPT_TEMPLATE.format(
            name=name, system_prompt=system_prompt)
        self.name: str = name
        self.api = self._initialize_api()
        ChatbotBase._total_count += 1
        self._bot_index: int = ChatbotBase._total_count

    @property
    def bot_index(self) -> int:
        return self._bot_index

    @classmethod
    def get_total_bots(cls) -> int:
        return cls._total_count

    @abstractmethod
    def _initialize_api(self) -> Any:
        pass

    @abstractmethod
    def _generate_raw_response(self, conversation: List[ConversationMessage]) -> str:
        """Generate raw response from the model without any formatting."""
        pass

    def generate_response(self, conversation: List[ConversationMessage]) -> str:
        """Generate and format response with proper name prefix."""
        try:
            raw_response = self._generate_raw_response(conversation)
            return self.format_response(raw_response)
        except Exception as e:
            print(f"Error generating response: {e}")
            return self.format_response("Error: Unable to generate response.")

    def _strip_name_prefix(self, response: str) -> str:
        """Remove bot name prefix if present in response."""
        prefix = f"{self.name}: "
        return response[len(prefix):] if response.startswith(prefix) else response

    def format_response(self, response: str) -> str:
        """Format response with bot name prefix."""
        clean_response = self._strip_name_prefix(response)
        return f"{self.name}: {clean_response}"

    @abstractmethod
    def _format_message(self, conversation: List[ConversationMessage]) -> List[T]:
        pass
