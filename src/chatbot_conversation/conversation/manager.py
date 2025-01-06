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

from chatbot_conversation.conversation import ConfigurationLoader
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

    def __init__(self, config_path: str):
        """
        Initialize conversation manager from config file.

        Args:
            config_path (str): Path to JSON configuration file.
        """
        logger.info("Initializing conversation manager")
        try:
            self.config = ConfigurationLoader.load_config(config_path)
        except Exception as e:
            logger.error("Failed to initialize conversation manager: %s", str(e))
            raise RuntimeError(
                f"Conversation manager initialization failed: {str(e)}"
            ) from e

        self.bots: List[ChatbotBase] = []

        # This is the seed message with the bot index set to a dummy bot index value of 0
        self.conversation: List[ConversationMessage] = [
            {"bot_index": 0, "content": self.config.conversation_seed}
        ]

        bot_registry = BotRegistry()  # get the singleton instance

        factory = ChatbotFactory(bot_registry)
        for bot_config in self.config.bots:

            # Format bot_name into shared prefix and add bot-specific prompt
            formatted_prefix = self.config.shared_prefix.replace(
                "{bot_name}", bot_config.bot_name
            )
            bot_specific_system_prompt = bot_config.bot_prompt
            bot_system_prompt = formatted_prefix + bot_specific_system_prompt

            bot = factory.create_bot(
                BotConfig(
                    bot_type=bot_config.bot_type,
                    bot_version=bot_config.bot_version,
                    bot_system_prompt=bot_system_prompt,
                    bot_name=bot_config.bot_name,
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
                try:
                    raw_response = bot.generate_response(self.conversation)
                except (IndexError, KeyError, AttributeError, ValueError) as e:
                    error_message = f"Exception: index/key/attribute/value error: {e}"
                    logger.error(error_message)
                    raw_response = (
                        "I'm sorry, I can't think of a response right now. "
                        "The values in my head are all over the place."
                    )
                except Exception as e:  # pylint: disable=broad-exception-caught
                    error_message = (
                        f"Exception: Unknown/API error generating response: {e}"
                    )
                    logger.error(error_message)
                    raw_response = (
                        "I'm sorry, I can't think of a response right now. "
                        "My mind seems to be elsewhere."
                    )
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

        for round_num in range(self.config.rounds):
            print(f"\n--- Round {round_num + 1} of {self.config.rounds} ---")
            if round_num == 0:
                self.tell_bots_first_round()  # Add the first round system prompt postfix
            if round_num == self.config.rounds - 1:
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
            bot.system_prompt_add_suffix(self.config.last_round_postfix)

    def tell_bots_first_round(self) -> None:
        """
        Inform bots that the conversation is about to start.
        """
        for bot in self.bots:
            bot.system_prompt_add_suffix(self.config.first_round_postfix)

    def tell_bots_not_first_round(self) -> None:
        """
        Remove the first round system prompt postfix from the system prompt.
        """
        for bot in self.bots:
            bot.system_prompt_remove_suffix(self.config.first_round_postfix)
