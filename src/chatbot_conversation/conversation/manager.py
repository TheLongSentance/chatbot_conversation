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
from chatbot_conversation.conversation.loader import load_conversation_config
from chatbot_conversation.conversation.transcript import TranscriptManager
from chatbot_conversation.models.base import ChatbotBase, ConversationMessage
from chatbot_conversation.utils import (
    LOGNAME_SYSTEM,
    ErrorSeverity,
    ModelException,
    get_logger,
)

logger = get_logger(LOGNAME_SYSTEM)


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
        self.config = load_conversation_config(config_path)
        self.bots: List[ChatbotBase] = []

        # This is the seed message with the bot index set to a dummy bot index value of 0
        self.conversation: List[ConversationMessage] = [
            {"bot_index": 0, "content": self.config.conversation_seed}
        ]

        bots_initializer = BotsInitializer()
        self.bots = bots_initializer.initialize_bots(self.config)

        self.display_manager = create_display()  # Use create_display for display

    def run_conversation(self) -> None:
        """
        Run the conversation for the configured number of rounds.
        """
        self.display_manager.clear()
        # Display conversation seed as title
        self.display_manager.show_text(f"# {self.config.conversation_seed}\n")

        # Run conversation for configured number of rounds 1 to num_rounds
        for round_num in range(1, self.config.rounds + 1):
            self.display_manager.show_text(
                f"## Round {round_num} of {self.config.rounds}\n\n---\n\n"
            )
            self.run_round(round_num)

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

    def run_round(self, round_num: int) -> None:
        """
        Run one round of responses from all bots.
        """
        logger.debug("Starting new conversation round")

        # Check for moderator message for this round
        for moderator_msg in self.config.moderator_messages_opt:
            if moderator_msg.round_number == round_num:
                moderator_content = f"**Moderator**: {moderator_msg.content}"
                # Always add to conversation history
                self.conversation.append({"bot_index": 0, "content": moderator_content})
                # Only display if display_opt is True
                if moderator_msg.display_opt:
                    self.display_manager.show_text(f"{moderator_content}\n\n---\n\n")
                break  # Only one moderator message per round

        # After checking for moderator, now run responses from all bots
        for bot in self.bots:
            try:
                # Use show_streaming_text to handle streaming response
                # Note: more complex to truncate streaming response to
                # last complete sentence since you don't know when it ends
                response = self.display_manager.show_streaming_text(
                    bot.stream_response(self.conversation)
                )

                # Clean potentially truncated response for use in
                # conversation history and transcript
                response = self.clean_truncated_response(response)

            except (IndexError, KeyError, AttributeError, ValueError) as e:
                # Handle data/validation errors
                raise ModelException(
                    message=f"Data error in bot response: {str(e)}",
                    user_message=f"**{bot.name}**: I'm sorry, I can't think of a response right now. The values in my head are all over the place.",
                    severity=ErrorSeverity.ERROR,
                    retry_allowed=False,
                    original_error=e,
            )

            # Store the complete response in conversation history
            self.conversation.append({"bot_index": bot.bot_index, "content": response})

            # Log the conversation content for debugging
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

    def clean_truncated_response(self, response: str) -> str:
        """
        Clean a potentially truncated response by finding the last complete sentence.
        Handles sentences ending with periods, question marks, or exclamation marks.
        Preserves any markdown formatting.

        Args:
            response (str): The potentially truncated response

        Returns:
            str: Response truncated to last complete sentence
        """
        sentence_endings = {".", "?", "!"}
        last_ending = -1
        for i in range(len(response) - 1, -1, -1):
            if response[i] in sentence_endings and i > 0:
                # Check not part of ellipsis
                if (
                    response[i] == "."
                    and i > 1
                    and response[i - 1] == "."
                    and response[i - 2] == "."
                ):
                    last_ending = i
                    break

                # Check has non-whitespace before
                if not response[i - 1].isspace():
                    # Check is end of string or has space after
                    if i == len(response) - 1 or response[i + 1].isspace():
                        last_ending = i
                        break

        if last_ending != -1:
            return response[: last_ending + 1]
        return response
