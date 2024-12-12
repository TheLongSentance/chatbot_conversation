from abc import ABC, abstractmethod
# from dataclasses import dataclass
from typing import List, Any, TypedDict, TypeVar, Generic
import os # type: ignore
from openai import OpenAI
import anthropic # type: ignore
import google.generativeai # type: ignore
import ollama # type: ignore
from enum import Enum, auto
from llms_env_ai import APIConfig

class BotType(Enum):
    GPT = auto()
    CLAUDE = auto()
    GEMINI = auto()
    OLLAMA = auto()

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


class ChatbotBase(ABC, Generic[T]):
    """Abstract base class defining interface for AI chatbot implementations.

    Provides contract for initialization, API setup, response generation and message formatting
    that concrete chatbot implementations must fulfill.
    """
    _total_count: int = 0  # Class variable to track total instances

    def __init__(self, model_version: str, system_prompt: str, name: str):
        """Initialize chatbot with model and system prompt.

        Args:
            model_version: Version/name of the AI model to use
            system_prompt: System instruction to set chatbot behavior
            name: Name of the chatbot
        """
        self.model_version: str = model_version
        self.system_prompt: str = system_prompt
        self.name: str = name
        self.api = self._initialize_api()
        ChatbotBase._total_count += 1
        self._bot_index: int = ChatbotBase._total_count

    @property
    def bot_index(self) -> int:
        """Get bot's assigned index number."""
        return self._bot_index

    @classmethod
    def get_total_bots(cls) -> int:
        """Get total number of chatbot instances created.

        Returns:
            Total number of chatbot instances created        
        """
        return cls._total_count

    @abstractmethod
    def _initialize_api(self) -> Any:
        """Set up connection to AI service API.

        Returns:
            API client instance for the service
        """
        pass

    @abstractmethod
    def generate_response(self, conversation: List[ConversationMessage]) -> str:
        """Generate next response based on conversation history.

        Args:
            conversation: List of previous messages in conversation

        Returns:
            Generated response text
        """
        pass

    @abstractmethod
    def _format_message(self, conversation: List[ConversationMessage]) -> List[T]:
        """Format conversation for API submission.

        Args:
            conversation: List of previous messages to format

        Returns:
            List of formatted messages ready for API submission
        """
        pass

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
    
class ClaudeChatbot(ChatbotBase[ChatMessage]):
    """Concrete implementation of chatbot using Claude's API service.
    
    Handles initialization of Claude client, message formatting specific to Claude's
    expected format, and response generation using the Claude model.
    
    Attributes:
        api: Claude client instance
        model_version: Version of Claude model to use
        system_prompt: System instruction for bot behavior
    """

    def __init__(self, model_version: str, system_prompt: str, name: str):
        """Initialize Claude chatbot with specific model and behavior.

        Args:
            model_version: Claude model version to use (e.g. "claude-3")
            system_prompt: System instruction defining bot behavior
            name: Name of the chatbot
        """
        super().__init__(model_version, system_prompt, name)
    
    def _initialize_api(self) -> Any:
        """Initialize connection to Claude API.

        Returns:
            Claude: Configured Claude client instance
        """
        return anthropic.Anthropic() # type: ignore
        
    def generate_response(self, conversation: List[ConversationMessage]) -> str:
        """Generate next response using Claude's chat model.

        Args:
            conversation: List of previous conversation messages

        Returns:
            str: Generated response from Claude model

        Note:
            Includes system prompt at start of every request
        """
        formatted_messages = self._format_message(conversation)
        
        try:
            message = self.api.messages.create(
                model=self.model_version,
                system=self.system_prompt,
                messages=formatted_messages,
                max_tokens=500,
                timeout=10
            )
            response = message.content[0].text
            return f"{self.name}: {response}"
        except Exception as e:
            print(f"Error calling Claude model: {e}")
            return f"{self.name}: Error: Unable to generate response from Claude model."

    def _format_message(self, conversation: List[ConversationMessage]) -> List[ChatMessage]:
        """Format message history for Claude API submission.

        Formats all messages according to Claude's expected structure.
        System prompt is not included in the message list for Claude.

        Args:
            conversation: List of conversation messages to format

        Returns:
            List[ChatMessage]: Messages formatted for Claude API
        """

        messages: List[ChatMessage] = []

        for contribution in conversation:
            role = "assistant" if contribution["bot_index"] == self.bot_index else "user"
            messages.append({"role": role, "content": contribution["content"]})

        return messages


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
    

class ChatbotFactory:
    """Factory for creating different types of chatbots."""
    
    @staticmethod
    def create_bot(bot_type: BotType, model_version: str, system_prompt: str, name: str) -> ChatbotBase[Any]:
        """Create a chatbot of specified type.
        
        Args:
            bot_type: Type of bot to create
            model_version: Model version to use
            system_prompt: System instruction for bot behavior
            name: Name of the chatbot
        
        Returns:
            Initialized chatbot instance
        """
        if bot_type == BotType.GPT:
            return OpenAIChatbot(model_version, system_prompt, name)
        elif bot_type == BotType.CLAUDE:
            return ClaudeChatbot(model_version, system_prompt, name)
        elif bot_type == BotType.GEMINI:
            return GeminiChatbot(model_version, system_prompt, name)
        elif bot_type == BotType.OLLAMA:
            return OllamaChatbot(model_version, system_prompt, name)
        else:
            raise ValueError(f"Unknown bot type: {bot_type}")


class ConversationManager:
    """Manages conversation between multiple chatbots."""
    
    def __init__(self, initial_message: str):
        """Initialize conversation with starting message.
        
        Args:
            initial_message: First message to start conversation
        """
        self.bots: List[ChatbotBase[Any]] = []
        self.conversation: List[ConversationMessage] = [
            {"bot_index": 0, "content": initial_message}
        ]
    
    def add_bot(self, bot: ChatbotBase[Any]) -> None:
        """Add a chatbot to the conversation.
        
        Args:
            bot: Initialized chatbot instance
        """
        self.bots.append(bot)
    
    def run_round(self) -> None:
        """Run one round of responses from all bots."""
        for bot in self.bots:
            response = bot.generate_response(self.conversation)
            self.conversation.append({
                "bot_index": bot.bot_index,
                "content": response
            })
            print(f"\n{bot.__class__.__name__}:\n{response}\n")



if __name__ == "__main__":

    gpt_model_version = "gpt-4o-mini"
    claude_model_version = "claude-3-haiku-20240307"
    gemini_model_version = "gemini-1.5-flash"
    ollama_model_version = "llama3.2"


    gpt_system = "You are an expert on modern professional tennis. \
    You think that Roger Federer is the greatest tennis player of all time (the 'GOAT') and are keen to \
    justify your opinion through your knowledge of tennis technique and results."

    claude_system = "You are an expert on modern professional tennis. \
    You think that Rafa Nadal is the greatest tennis player of all time (the 'GOAT') and are keen to \
    justify your opinion through your knowledge of tennis technique and results."

    gemini_system = "You are an expert on modern professional tennis. \
    You think that Novak Djokovic is the greatest tennis player of all time (the 'GOAT') and are keen to \
    justify your opinion through your knowledge of tennis technique and results."

    ollama_system = "You are an expert on modern professional tennis. \
    You are not sure who should have the title of the greatest player of all time \
    (the 'GOAT) but you hope to base your decision on the opinions of others"

    APIConfig.setup_env()
    factory = ChatbotFactory()

    gpt_bot = factory.create_bot(BotType.GPT, gpt_model_version, gpt_system, "RogerFan")
    claude_bot = factory.create_bot(BotType.CLAUDE, claude_model_version, claude_system, "RafaFan")
    gemini_bot = factory.create_bot(BotType.GEMINI, gemini_model_version, gemini_system, "NovakFan")
    ollama_bot = factory.create_bot(BotType.OLLAMA, ollama_model_version, ollama_system, "AndyFan")
    
    # Initialize manager and add bots
    manager = ConversationManager("I think Roger Federer is the GOAT!")
    for bot in [gpt_bot, claude_bot, gemini_bot, ollama_bot]:
        manager.add_bot(bot)
    
    # Run conversation
    for _ in range(2):
        manager.run_round()











