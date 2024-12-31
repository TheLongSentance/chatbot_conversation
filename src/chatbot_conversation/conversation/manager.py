"""
This module manages the conversation flow for the chatbot application.

It includes the following functionalities:
- Setting up logging configuration
- Defining error messages as constants
- Importing necessary classes and functions from other modules

Classes:
    ConversationManager: Manages conversation between multiple chatbots.
"""

import json
import os
from typing import List

from chatbot_conversation.conversation import (
    ERROR_EMPTY_BOTS,
    ERROR_EMPTY_FIELD,
    ERROR_EMPTY_PREFIX,
    ERROR_EMPTY_SEED,
    ERROR_INVALID_ROUNDS,
    ConfigurationLoader,
    ConversationConfig,
)
from chatbot_conversation.models import (
    BotConfig,
    BotRegistry,
    ChatbotBase,
    ChatbotFactory,
    ConversationMessage,
)
from chatbot_conversation.utils import get_logger

logger = get_logger("conversation")


class ConversationManager:
    """Manages conversation between multiple chatbots."""

    @classmethod
    def from_config(cls, config_path: str) -> "ConversationManager":
        """
        Create ConversationManager from config file.

        Args:
            config_path (str): Path to JSON configuration file.

        Returns:
            ConversationManager: An instance of ConversationManager.
        """
        config = ConfigurationLoader.load_config(config_path)

        # Check if conversation_seed is empty
        if not config.get("conversation_seed"):
            raise ValueError(ERROR_EMPTY_SEED)

        # Check if rounds is a positive integer
        if config["rounds"] <= 0:
            raise ValueError(ERROR_INVALID_ROUNDS)

        # Check if shared_prefix is empty
        if not config.get("shared_prefix"):
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
            if not bot["bot_version"]:
                raise ValueError(ERROR_EMPTY_FIELD.format(field="bot_version"))
            if not bot["bot_prompt"]:
                raise ValueError(ERROR_EMPTY_FIELD.format(field="bot_prompt"))

        return cls(config)

    def __init__(self, config: ConversationConfig):
        """
        Initialize conversation with starting message.

        Args:
            config (ConversationConfig): Conversation configuration.
        """
        logger.info("Initializing conversation manager")
        self.config = config
        self.conversation_seed: str = config["conversation_seed"]
        self.num_rounds = self.config["rounds"]
        self.system_prompts: dict[str, str] = {
            "shared_prefix": config.get("shared_prefix", ""),
            "first_round_postfix": config.get("first_round_postfix", ""),
            "last_round_postfix": config.get("last_round_postfix", "")
        }
        self.bots: List[ChatbotBase] = []

        # This is the seed message with the bot index set to a dummy bot index value of 0
        self.conversation: List[ConversationMessage] = [
            {"bot_index": 0, "content": self.conversation_seed}
        ]

        bot_registry = BotRegistry()  # get the singleton instance

        factory = ChatbotFactory(bot_registry)
        for bot_config in config["bots"]:

            # Format bot_name into shared prefix and add bot-specific prompt
            bot_name = bot_config.get("bot_name", "")
            formatted_prefix = self.system_prompts["shared_prefix"].replace(
                "{bot_name}", bot_name
            )
            bot_specific_system_prompt = bot_config.get(
                "bot_prompt", ""
            )
            bot_system_prompt = formatted_prefix + bot_specific_system_prompt

            bot = factory.create_bot(
                BotConfig(
                    bot_type=bot_config.get("bot_type", ""),
                    bot_version=bot_config.get("bot_version", ""),
                    bot_system_prompt=bot_system_prompt,
                    bot_name=bot_name,
                )
            )
            self.add_bot(bot)

    def add_bot(self, bot: ChatbotBase) -> None:
        """
        Add a chatbot to the conversation.

        Args:
            bot (ChatbotBase): Initialized chatbot instance.
        """
        self.bots.append(bot)

    def run_round(self) -> None:
        """
        Run one round of responses from all bots.
        """
        logger.debug("Starting new conversation round")
        try:
            for bot in self.bots:
                raw_response = bot.generate_response(self.conversation)

                clean_response: str = self._format_response(bot.name, raw_response)

                self.conversation.append(
                    {"bot_index": bot.bot_index, "content": clean_response}
                )

                logger.debug(
                    "Bot Class: %s, Bot Name: %s, Bot Index: %s, Updated conversation: : %s",
                    bot.__class__.__name__,
                    bot.name,
                    bot.bot_index,
                    json.dumps(self.conversation, indent=2),
                )

                print(
                    f"\n*** {bot.__class__.__name__} Bot#{bot.bot_index} ***\n\n{clean_response}\n"
                )
            logger.info("Round completed successfully")
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.error("Unexpected error during conversation round: %s", str(e))

    def _format_response(self, bot_name: str, raw_response: str) -> str:
        """
        Format response with bot name prefix.

        Args:
            bot_name (str): The name of the bot.
            raw_response (str): The raw response string.

        Returns:
            str: The formatted response with bot name prefix.
        """
        clean_response = self._strip_name_prefix(bot_name, raw_response)
        return f"<<< {bot_name} >>> {clean_response}"

    def _strip_name_prefix(self, bot_name: str, response: str) -> str:
        """
        Remove bot name prefix if present in response.

        Args:
            bot_name (str): The name of the bot.
            response (str): The response string.

        Returns:
            str: The response without the bot name prefix.
        """
        prefix = f"<<< {bot_name} >>> "
        return response[len(prefix) :] if response.startswith(prefix) else response

    def run_conversation(self) -> None:
        """
        Run the conversation for the configured number of rounds.
        """

        # Clear the terminal screen
        os.system("cls" if os.name == "nt" else "clear")

        print("**********************************")
        print("***   Starting conversation:   ***")
        print("**********************************\n")
        print(f'{self.conversation[0]["content"]}\n')
        print("**********************************\n")

        for round_num in range(
            self.num_rounds
        ):  # Run the second to the last but one round
            print(f"\n--- Round {round_num + 1} of {self.num_rounds} ---")
            if round_num == 0:
                self.tell_bots_first_round()  # Add the first round system prompt postfix
            if round_num == self.num_rounds - 1:
                self.tell_bots_last_round()  # Add the last round system prompt postfix
            self.run_round()
            if round_num == 0:
                self.tell_bots_not_first_round()  # Remove the first round system prompt postfix

        print("\n**********************************")
        print("***   Conversation completed   ***")
        print("**********************************\n\n")

    def tell_bots_last_round(self) -> None:
        """
        Inform bots that the conversation is about to end.
        """
        for bot in self.bots:
            bot.append_to_system_prompt(self.system_prompts["last_round_postfix"])

    def tell_bots_first_round(self) -> None:
        """
        Inform bots that the conversation is about to start.
        """
        for bot in self.bots:
            bot.append_to_system_prompt(self.system_prompts["first_round_postfix"])

    def tell_bots_not_first_round(self) -> None:
        """
        Remove the first round system prompt postfix from the system prompt.
        """
        for bot in self.bots:
            bot.remove_from_system_prompt(self.system_prompts["first_round_postfix"])
