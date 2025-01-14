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
from datetime import datetime
from typing import List

from rich.console import Console
from rich.markdown import Markdown

from chatbot_conversation.conversation.loader import ConfigurationLoader
from chatbot_conversation.models import (
    BotConfig,
    BotRegistry,
    ChatbotBase,
    ChatbotFactory,
    ConversationMessage,
)
from chatbot_conversation.utils import get_logger

TRANSCRIPT_FILE_STUB = "transcript_"
TRANSCRIPT_OUTPUT_DIR = "./output/"

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

        self.config_path = config_path
        self.config = ConfigurationLoader.load_config(config_path)
        self.bots: List[ChatbotBase] = []

        # This is the seed message with the bot index set to a dummy bot index value of 0
        self.conversation: List[ConversationMessage] = [
            {"bot_index": 0, "content": self.config.conversation_seed}
        ]

        bot_registry = BotRegistry()  # get the singleton instance

        factory = ChatbotFactory(bot_registry)
        for bot_config in self.config.bots:

            # Format bot_name into shared prefix and add bot-specific prompt
            bot_system_prompt = self.config.shared_prefix + bot_config.bot_prompt
            bot_system_prompt = self.insert_bot_name(
                bot_system_prompt, bot_config.bot_name
            )

            bot = factory.create_bot(
                BotConfig(
                    bot_type=bot_config.bot_type,
                    bot_version=bot_config.bot_version,
                    bot_system_prompt=bot_system_prompt,
                    bot_name=bot_config.bot_name,
                )
            )
            self.add_bot(bot)

        self.console = Console()  # Initialize the console for rich text output

    def add_bot(self, bot: ChatbotBase) -> None:
        """
        Add a chatbot to the conversation.

        Args:
            bot (ChatbotBase): Initialized chatbot instance.
        """
        self.bots.append(bot)

    def display_clear(self) -> None:
        """
        Clear the terminal screen.
        """
        os.system("cls" if os.name == "nt" else "clear")

    def display_text(self, text: str) -> None:
        """
        Display text in the terminal.

        Args:
            text (str): Text to display.
        """
        self.console.print(Markdown(f"{text}"))

    def run_conversation(self) -> None:
        """
        Run the conversation for the configured number of rounds.
        """
        self.display_clear()
        # Display conversation seed as title
        self.display_text(f"# {self.config.conversation_seed}\n")

        for round_index in range(self.config.rounds):
            self.manage_round(round_index + 1)

        # Conversation completed
        completion_message = (
            f"## Conversation Finished - {self.config.rounds} Rounds With "
            f"{len(self.bots)} Bots Completed!\n\n---\n\n"
        )
        self.display_text(completion_message)

        self.write_conversation_to_file()

    def manage_round(self, round_num: int) -> None:
        """
        Manage the conversation for a single round.

        Args:
            round_index (int): Index of the current round.
        """
        self.display_text(f"## Round {round_num} of {self.config.rounds}\n\n---\n\n")

        if round_num == 1:  # Add the first round system prompt postfix
            self.tell_bots_first_round()
        if round_num == self.config.rounds:  # Add the last round postfix
            self.tell_bots_last_round()

        self.run_round()

        if round_num == 1:  # Remove the first round system prompt postfix
            self.tell_bots_not_first_round()

        # No need to remove the last round postfix since conversation over

    def run_round(self) -> None:
        """
        Run one round of responses from all bots.
        """
        logger.debug("Starting new conversation round")
        for bot in self.bots:
            try:
                response = bot.generate_response(self.conversation)
            except (IndexError, KeyError, AttributeError, ValueError) as e:
                error_message = f"Exception: index/key/attribute/value error: {e}"
                logger.error(error_message)
                response = (
                    f"**{bot.name}**: I'm sorry, I can't think of a response right now. "
                    "The values in my head are all over the place."
                )
            except Exception as e:  # pylint: disable=broad-exception-caught
                error_message = f"Exception: Unknown/API error generating response: {e}"
                logger.error(error_message)
                response = (
                    f"**{bot.name}**: I'm sorry, I can't think of a response right now. "
                    "My mind seems to be elsewhere."
                )
            self.conversation.append({"bot_index": bot.bot_index, "content": response})
            logger.debug(
                "Bot Class: %s, Bot Name: %s, Bot Index: %s, Updated conversation: : %s",
                bot.__class__.__name__,
                bot.name,
                bot.bot_index,
                json.dumps(self.conversation, indent=2),
            )
            self.display_text(f"{response}\n\n---\n\n")
        logger.info("Round completed successfully")

    def tell_bots_first_round(self) -> None:
        """
        Inform bots that the conversation is about to start.
        """
        for bot in self.bots:
            suffix = self.insert_bot_name(self.config.first_round_postfix, bot.name)
            bot.system_prompt_add_suffix(suffix)

    def tell_bots_not_first_round(self) -> None:
        """
        Remove the first round system prompt postfix from the system prompt.
        """
        for bot in self.bots:
            suffix = self.insert_bot_name(self.config.first_round_postfix, bot.name)
            bot.system_prompt_remove_suffix(suffix)

    def tell_bots_last_round(self) -> None:
        """
        Inform bots that the conversation is about to end.
        """
        for bot in self.bots:
            suffix = self.insert_bot_name(self.config.last_round_postfix, bot.name)
            bot.system_prompt_add_suffix(suffix)

    def insert_bot_name(self, text: str, bot_name: str) -> str:
        """
        Insert the bot name into the text.

        Args:
            text (str): Text to insert the bot name into.
            bot_name (str): Name of the bot.

        Returns:
            str: Text with the bot name inserted.
        """
        return text.replace("{bot_name}", bot_name)

    def write_conversation_to_file(self) -> None:
        """
        Write the entire conversation to a markdown file.
        """
        try:
            timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
            file_path = os.path.join(
                TRANSCRIPT_OUTPUT_DIR, f"{TRANSCRIPT_FILE_STUB}{timestamp}.md"
            )
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, "w", encoding="utf-8") as file:
                # Write the title
                file.write(f"# {self.conversation[0]['content']}\n\n")

                # Write each round and responses
                round_index = 1
                for i, message in enumerate(self.conversation[1:], start=1):
                    if (i - 1) % len(self.bots) == 0:
                        file.write(
                            f"## Round {round_index} of {self.config.rounds}\n\n"
                        )
                        round_index += 1
                    file.write(f"{message['content']}\n\n---\n\n")

                # Write the finish message
                file.write(
                    f"## Conversation Finished - {self.config.rounds} Rounds With "
                    f"{len(self.bots)} Bots Completed!\n\n"
                )

                # Write the conversation generated datetime
                file.write(
                    "## *Conversation Generated* : "
                    f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
                )

                # Write the configuration author name
                file.write(f"## *Configuration Author* : {self.config.author}\n\n")

                # Write the configuration file name
                file.write(f"## *Configuration File* : {self.config_path}\n\n")

                # Write the configuration data
                file.write("```json\n")
                file.write(json.dumps(self.config.model_dump(), indent=4))
                file.write("\n```\n")

            self.display_text(
                "Conversation transcript and configuration data saved to: "
                f"`{file_path}`\n\n---\n\n"
            )
            logger.info("Conversation successfully written to %s", file_path)

        except (IOError, ValueError) as e:
            logger.error("Failed to write conversation to file: %s", str(e))
