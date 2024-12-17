from typing import List, Any
from ..models import ChatbotBase, ConversationMessage
from ..models.base import BotType
from ..models.factory import ChatbotFactory
from .loader import ConfigurationLoader, ConversationConfig
import logging
import logging.config

# Set up logging from config file
logging.config.fileConfig('logging.conf')
logger = logging.getLogger('chatbot_conversation')

class ConversationManager:
    """Manages conversation between multiple chatbots."""
    
    @classmethod
    def from_config(cls, config_path: str) -> 'ConversationManager':
        """Create ConversationManager from config file.
        
        Args:
            config_path: Path to JSON configuration file
        """
        config = ConfigurationLoader.load_config(config_path)

        # Check if conversation_seed is empty
        if not config.get("conversation_seed"):
            raise ValueError("Conversation seed cannot be empty")
        
        # Check if rounds is a positive integer
        if config["rounds"] <= 0:
            raise ValueError("Rounds must be a positive integer")
        
        # Check if conversation_seed is empty
        if not config.get("shared_system_prompt_prefix"):
            raise ValueError("Shared system prompt prefix cannot be empty")

        # Check if bots list is not empty
        if not config.get("bots") or len(config["bots"]) == 0:
            raise ValueError("Bots list cannot be empty")
        
        # Check if each bot has the required fields and they are not empty
        required_bot_fields = ["bot_name", "bot_type", "bot_model_version", "bot_specific_system_prompt"]
        for bot in config["bots"]:
            for field in required_bot_fields:
                if field not in bot or not bot[field]:
                    raise ValueError(f"Each bot must have a non-empty '{field}' field")

        return cls(config)
    
    def __init__(self, config: ConversationConfig):
        """Initialize conversation with starting message.
        
        Args:
            config: Conversation configuration
        """
        logger.info("Initializing conversation manager")
        self.config = config
        self.bots: List[ChatbotBase[Any]] = []
        self.conversation: List[ConversationMessage] = [
            {"bot_index": 0, "content": config['conversation_seed']}
        ]
        
        factory = ChatbotFactory()
        shared_system_prompt_prefix = config.get('shared_system_prompt_prefix', '')
        for bot_config in config['bots']:
            bot = factory.create_bot(
                BotType[bot_config['bot_type']], 
                bot_config['bot_model_version'],
                bot_config['bot_specific_system_prompt'],
                bot_config['bot_name'],
                shared_system_prompt_prefix
            )
            self.add_bot(bot)
    
    def add_bot(self, bot: ChatbotBase[Any]) -> None:
        """Add a chatbot to the conversation.
        
        Args:
            bot: Initialized chatbot instance
        """
        self.bots.append(bot)
    
    def run_round(self) -> None:
        """Run one round of responses from all bots."""
        logger.debug("Starting new conversation round")
        try:
            for bot in self.bots:
                response = bot.generate_response(self.conversation)
                self.conversation.append({
                    "bot_index": bot.bot_index,
                    "content": response
                })
                print(f"\n*** {bot.__class__.__name__} Bot#{bot.bot_index} ***\n\n{response}\n")
            logger.info("Round completed successfully")
        except Exception as e:
            logger.error(f"Error during conversation round: {str(e)}")
    
    def run_conversation(self) -> None:
        """Run the conversation for the configured number of rounds."""
        print("\n\n*** Starting conversation: ***\n")
        print(f'{self.conversation[0]["content"]}\n')
        
        for round_num in range(self.config['rounds']):
            print(f"\n--- Round {round_num + 1} ---\n")
            self.run_round()