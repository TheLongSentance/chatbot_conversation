"""
This module manages the conversation flow for the chatbot application.

It includes the following functionalities:
- Setting up logging configuration
- Defining error messages as constants
- Importing necessary classes and functions from other modules

Classes:
- ChatbotBase
- ConversationMessage
- BotType
- ChatbotFactory
- ConfigurationLoader
- ConversationConfig

Constants:
- ERROR_EMPTY_CONVERSATION_SEED
- ERROR_INVALID_ROUNDS
- ERROR_EMPTY_SHARED_SYSTEM_PROMPT_PREFIX
- ERROR_EMPTY_BOTS_LIST
- ERROR_EMPTY_BOT_FIELD
"""

from typing import List, Any
import os
import logging
import logging.config

from ..models import ChatbotBase, ConversationMessage
from ..models.base import BotType
from ..models.factory import ChatbotFactory
from .loader import ConfigurationLoader, ConversationConfig

# Set up logging from config file
logging.config.fileConfig('logging.conf')
logger = logging.getLogger('chatbot_conversation')

# Define error messages as constants
ERROR_EMPTY_CONVERSATION_SEED = "Conversation seed cannot be empty"
ERROR_INVALID_ROUNDS = "Rounds must be a positive integer"
ERROR_EMPTY_SHARED_SYSTEM_PROMPT_PREFIX = "Shared system prompt prefix cannot be empty"
ERROR_EMPTY_BOTS_LIST = "Bots list cannot be empty"
ERROR_EMPTY_BOT_FIELD = "Each bot must have a non-empty '{field}' field"

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
            raise ValueError(ERROR_EMPTY_CONVERSATION_SEED)

        # Check if rounds is a positive integer
        if config["rounds"] <= 0:
            raise ValueError(ERROR_INVALID_ROUNDS)

        # Check if conversation_seed is empty
        if not config.get("shared_system_prompt_prefix"):
            raise ValueError(ERROR_EMPTY_SHARED_SYSTEM_PROMPT_PREFIX)

        # Check if bots list is not empty
        if not config.get("bots") or len(config["bots"]) == 0:
            raise ValueError(ERROR_EMPTY_BOTS_LIST)

        # Check each bot field individually with literal keys
        for bot in config["bots"]:
            if not bot["bot_name"]:
                raise ValueError(ERROR_EMPTY_BOT_FIELD.format(field="bot_name"))
            if not bot["bot_type"]:
                raise ValueError(ERROR_EMPTY_BOT_FIELD.format(field="bot_type"))
            if not bot["bot_model_version"]:
                raise ValueError(ERROR_EMPTY_BOT_FIELD.format(field="bot_model_version"))
            if not bot["bot_specific_system_prompt"]:
                raise ValueError(ERROR_EMPTY_BOT_FIELD.format(field="bot_specific_system_prompt"))

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
        except Exception as e:      # pylint: disable=broad-exception-caught
            logger.error("Unexpected error during conversation round: %s", str(e))

    def run_conversation(self) -> None:
        """Run the conversation for the configured number of rounds."""

        # Clear the terminal screen
        os.system('cls' if os.name == 'nt' else 'clear')

        print("**********************************")
        print("***   Starting conversation:   ***")
        print("**********************************\n")
        print(f'{self.conversation[0]["content"]}\n')
        print("**********************************\n")

        for round_num in range(self.config['rounds']):
            print(f"\n--- Round {round_num + 1} ---")
            self.run_round()

        print("\n**********************************")
        print("***   Conversation completed   ***")
        print("**********************************\n\n")
