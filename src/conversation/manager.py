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
"""

import logging
import logging.config
import os
from typing import Any, List

from ..models import ChatbotBase, ConversationMessage
from ..models.base import BotType
from ..models.factory import ChatbotFactory
from .loader import (
    BOT_MODEL_VERSION,
    BOT_NAME,
    BOT_SYSTEM_PROMPT,
    BOT_TYPE,
    BOTS,
    CONVERSATION_SEED,
    ERROR_EMPTY_BOTS,
    ERROR_EMPTY_FIELD,
    ERROR_EMPTY_PREFIX,
    ERROR_EMPTY_SEED,
    ERROR_INVALID_ROUNDS,
    ROUNDS,
    SHARED_SYSTEM_PROMPT_PREFIX,
    ConfigurationLoader,
    ConversationConfig,
)

# Set up logging from config file
logging.config.fileConfig("logging.conf")
logger = logging.getLogger("chatbot_conversation")


class ConversationManager:
    """Manages conversation between multiple chatbots."""

    @classmethod
    def from_config(cls, config_path: str) -> "ConversationManager":
        """Create ConversationManager from config file.

        Args:
            config_path: Path to JSON configuration file
        """
        config = ConfigurationLoader.load_config(config_path)

        # Check if conversation_seed is empty
        if not config.get(CONVERSATION_SEED):
            raise ValueError(ERROR_EMPTY_SEED)

        # Check if rounds is a positive integer
        if config[ROUNDS] <= 0:
            raise ValueError(ERROR_INVALID_ROUNDS)

        # Check if shared_system_prompt_prefix is empty
        if not config.get(SHARED_SYSTEM_PROMPT_PREFIX):
            raise ValueError(ERROR_EMPTY_PREFIX)

        # Check if bots list is not empty
        if not config.get(BOTS) or len(config[BOTS]) == 0:
            raise ValueError(ERROR_EMPTY_BOTS)

        # Check each bot field individually with constants
        for bot in config[BOTS]:
            if not bot[BOT_NAME]:
                raise ValueError(ERROR_EMPTY_FIELD.format(field=BOT_NAME))
            if not bot[BOT_TYPE]:
                raise ValueError(ERROR_EMPTY_FIELD.format(field=BOT_TYPE))
            if not bot[BOT_MODEL_VERSION]:
                raise ValueError(ERROR_EMPTY_FIELD.format(field=BOT_MODEL_VERSION))
            if not bot[BOT_SYSTEM_PROMPT]:
                raise ValueError(ERROR_EMPTY_FIELD.format(field=BOT_SYSTEM_PROMPT))

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
            {"bot_index": 0, "content": config[CONVERSATION_SEED]}
        ]

        factory = ChatbotFactory()
        shared_system_prompt_prefix = config.get(SHARED_SYSTEM_PROMPT_PREFIX, "")
        for bot_config in config[BOTS]:
            bot = factory.create_bot(
                BotType[bot_config[BOT_TYPE]],
                str(bot_config[BOT_MODEL_VERSION]),
                bot_config[BOT_SYSTEM_PROMPT],
                bot_config[BOT_NAME],
                shared_system_prompt_prefix,
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
                self.conversation.append(
                    {"bot_index": bot.bot_index, "content": response}
                )
                print(
                    f"\n*** {bot.__class__.__name__} Bot#{bot.bot_index} ***\n\n{response}\n"
                )
            logger.info("Round completed successfully")
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.error("Unexpected error during conversation round: %s", str(e))

    def run_conversation(self) -> None:
        """Run the conversation for the configured number of rounds."""

        # Clear the terminal screen
        os.system("cls" if os.name == "nt" else "clear")

        print("**********************************")
        print("***   Starting conversation:   ***")
        print("**********************************\n")
        print(f'{self.conversation[0]["content"]}\n')
        print("**********************************\n")

        for round_num in range(self.config[ROUNDS]):
            print(f"\n--- Round {round_num + 1} ---")
            self.run_round()

        print("\n**********************************")
        print("***   Conversation completed   ***")
        print("**********************************\n\n")
