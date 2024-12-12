from typing import List, Any, TypedDict
from ..models import ChatbotBase, ConversationMessage
from ..models.base import BotType
from ..models.factory import ChatbotFactory
from .loader import ConfigurationLoader

class BotConfig(TypedDict):
    name: str
    bot_type: str
    model_version: str
    system_prompt: str

class ConversationConfig(TypedDict):
    conversation_seed: str
    rounds: int
    bots: List[BotConfig]

class ConversationManager:
    """Manages conversation between multiple chatbots."""
    
    @classmethod
    def from_config(cls, config_path: str) -> 'ConversationManager':
        """Create ConversationManager from config file.
        
        Args:
            config_path: Path to JSON configuration file
        """
        config = ConfigurationLoader.load_config(config_path)
        manager = cls(config['conversation_seed'])
        
        factory = ChatbotFactory()
        for bot_config in config['bots']:
            bot = factory.create_bot(
                BotType[bot_config['bot_type']], 
                bot_config['model_version'],
                bot_config['system_prompt'],
                bot_config['name']
            )
            manager.add_bot(bot)
        
        return manager

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
