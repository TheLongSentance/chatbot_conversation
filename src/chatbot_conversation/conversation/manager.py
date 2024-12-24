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

import json
import os
from typing import List

from chatbot_conversation.conversation.loader import (
    ERROR_EMPTY_BOTS,
    ERROR_EMPTY_FIELD,
    ERROR_EMPTY_PREFIX,
    ERROR_EMPTY_SEED,
    ERROR_INVALID_ROUNDS,
    ConfigurationLoader,
    ConversationConfig,
)
from chatbot_conversation.models import (
    BotType,
    ChatbotBase,
    ChatbotFactory,
    ConversationMessage,
)
from chatbot_conversation.utils import get_logger
from chatbot_conversation.models import BotRegistry
from chatbot_conversation.models import ClaudeChatbot
from chatbot_conversation.models import GeminiChatbot
from chatbot_conversation.models import OllamaChatbot
from chatbot_conversation.models import OpenAIChatbot

logger = get_logger("conversation")


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
        if not config.get("conversation_seed"):
            raise ValueError(ERROR_EMPTY_SEED)

        # Check if rounds is a positive integer
        if config["rounds"] <= 0:
            raise ValueError(ERROR_INVALID_ROUNDS)

        # Check if shared_system_prompt_prefix is empty
        if not config.get("shared_system_prompt_prefix"):
            raise ValueError(ERROR_EMPTY_PREFIX)

        # Check if bots list is not empty
        if not config.get("bots") or len(config["bots"]) == 0:
            raise ValueError(ERROR_EMPTY_BOTS)

        # Check each bot field individually with constants
        for bot in config["bots"]:
            if not bot["bot_name"]:
                raise ValueError(ERROR_EMPTY_FIELD.format(field="bot_name"))
            if not bot["bot_type"]:
                raise ValueError(ERROR_EMPTY_FIELD.format(field="bot_type"))
            if not bot["bot_model_version"]:
                raise ValueError(ERROR_EMPTY_FIELD.format(field="bot_model_version"))
            if not bot["bot_specific_system_prompt"]:
                raise ValueError(
                    ERROR_EMPTY_FIELD.format(field="bot_specific_system_prompt")
                )

        return cls(config)

    def __init__(self, config: ConversationConfig):
        """Initialize conversation with starting message.

        Args:
            config: Conversation configuration
        """
        logger.info("Initializing conversation manager")
        self.config = config
        self.bots: List[ChatbotBase] = []
        self.conversation: List[ConversationMessage] = [
            {"bot_index": 0, "content": config["conversation_seed"]}
        ]

        bot_registry = BotRegistry()  # create an instance of BotRegistry

        # Register the existing chatbot models
        bot_registry.register_bot(BotType.GPT, OpenAIChatbot)
        bot_registry.register_bot(BotType.CLAUDE, ClaudeChatbot)
        bot_registry.register_bot(BotType.GEMINI, GeminiChatbot)
        bot_registry.register_bot(BotType.OLLAMA, OllamaChatbot)

        factory = ChatbotFactory(bot_registry)
        shared_system_prompt_prefix = config.get("shared_system_prompt_prefix", "")
        for bot_config in config["bots"]:
            bot = factory.create_bot(
                BotType[bot_config["bot_type"]],
                str(bot_config["bot_model_version"]),
                bot_config["bot_specific_system_prompt"],
                bot_config["bot_name"],
                shared_system_prompt_prefix,
            )
            self.add_bot(bot)

    def add_bot(self, bot: ChatbotBase) -> None:
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

                logger.debug(
                    "Bot Class: %s, Bot Name: %s, Bot Index: %s, Updated conversation: : %s",
                    bot.__class__.__name__,
                    bot.name,
                    bot.bot_index,
                    json.dumps(self.conversation, indent=2),
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

        for round_num in range(self.config["rounds"]):
            print(f"\n--- Round {round_num + 1} ---")
            self.run_round()

        print("\n**********************************")
        print("***   Conversation completed   ***")
        print("**********************************\n\n")
