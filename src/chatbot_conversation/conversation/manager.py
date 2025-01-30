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
from pathlib import Path
from typing import List

from chatbot_conversation.conversation.bots_initializer import BotsInitializer
from chatbot_conversation.conversation.display import create_display
from chatbot_conversation.conversation.loader import ConfigurationLoader
from chatbot_conversation.conversation.prompt import SuffixManager
from chatbot_conversation.conversation.transcript import TranscriptManager
from chatbot_conversation.models.base import ChatbotBase, ConversationMessage
from chatbot_conversation.utils.logging_util import get_logger

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

        bots_initializer = BotsInitializer()
        self.bots = bots_initializer.initialize_bots(self.config)

        self.display_manager = create_display()  # Use create_display for display
        self.suffix_manager = SuffixManager()

    def run_conversation(self) -> None:
        """
        Run the conversation for the configured number of rounds.
        """
        self.display_manager.clear()
        # Display conversation seed as title
        self.display_manager.show_text(f"# {self.config.conversation_seed}\n")

        for round_index in range(self.config.rounds):
            self.manage_round(round_index + 1)

        # Conversation completed
        completion_message = (
            f"## Conversation Finished - {self.config.rounds} Rounds With "
            f"{len(self.bots)} Bots Completed!\n\n---\n\n"
        )
        self.display_manager.show_text(completion_message)

        transcript_path: Path = TranscriptManager.save_transcript(
            self.conversation, self.config, self.config_path
        )

        self.display_manager.show_text(
            "Conversation transcript and configuration data saved to: "
            f"`{transcript_path}`\n\n---\n\n"
        )

    def manage_round(self, round_num: int) -> None:
        """
        Manage the conversation for a single round.

        Args:
            round_index (int): Index of the current round.
        """
        self.display_manager.show_text(
            f"## Round {round_num} of {self.config.rounds}\n\n---\n\n"
        )

        self.round_setup(round_num)

        self.run_round()

        self.round_cleanup(round_num)

    def run_round(self) -> None:
        """
        Run one round of responses from all bots.
        """
        logger.debug("Starting new conversation round")
        for bot in self.bots:
            try:
                # Use show_streaming_text to handle streaming response
                response = self.display_manager.show_streaming_text(
                    bot.stream_response(self.conversation)
                )
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
                    "My mind seems to be focussed elsewhere."
                )

            # Store the complete response in conversation history
            self.conversation.append({"bot_index": bot.bot_index, "content": response})
            logger.debug(
                "Bot Class: %s, Bot Name: %s, Bot Index: %s, Updated conversation: : %s",
                bot.__class__.__name__,
                bot.name,
                bot.bot_index,
                json.dumps(self.conversation, indent=2),
            )
            # Add separator after complete response
            self.display_manager.show_text("\n\n---\n\n")
        logger.info("Round completed successfully")

    def round_setup(self, round_num: int) -> None:
        """
        Perform setup actions before starting a new round.

        Args:
            round_num (int): Index of the current round.
        """
        postfix: str = ""
        if round_num == 1:
            postfix = self.config.first_round_postfix
        elif round_num == self.config.rounds:
            postfix = self.config.last_round_postfix

        if round_num in (1, self.config.rounds):
            for bot in self.bots:
                self.suffix_manager.setup_round_suffix(bot, postfix)

    def round_cleanup(self, round_num: int) -> None:
        """
        Perform cleanup actions after finishing a round.

        Args:
            round_num (int): Index of the current round.
        """
        # Post-round actions undoing system prompt adjustments
        if round_num in (1, self.config.rounds):
            for bot in self.bots:
                self.suffix_manager.cleanup_round_suffix(bot)
