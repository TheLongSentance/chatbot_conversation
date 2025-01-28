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
from pathlib import Path

from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown

from chatbot_conversation.conversation.loader import (
    ChatbotConfigData,
    ConfigurationLoader,
)
from chatbot_conversation.models import (
    ChatbotBase,
    ConversationMessage,
)
from chatbot_conversation.conversation.bots_initializer import BotsInitializer
from chatbot_conversation.conversation.transcript import TranscriptManager
from chatbot_conversation.models.base import DEFAULT_MAX_TOKENS

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

        self.config_path = config_path
        self.config = ConfigurationLoader.load_config(config_path)
        self.bots: List[ChatbotBase] = []

        # This is the seed message with the bot index set to a dummy bot index value of 0
        self.conversation: List[ConversationMessage] = [
            {"bot_index": 0, "content": self.config.conversation_seed}
        ]

        bots_initializer = BotsInitializer(self.config)
        self.bots = bots_initializer.initialize_bots(self.config)

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

        transcript_path: Path = TranscriptManager.save_transcript(
            self.conversation, 
            self.config, 
            self.config_path
        )

        self.display_text(
            "Conversation transcript and configuration data saved to: "
            f"`{transcript_path}`\n\n---\n\n"
        )

    def manage_round(self, round_num: int) -> None:
        """
        Manage the conversation for a single round.

        Args:
            round_index (int): Index of the current round.
        """
        self.display_text(f"## Round {round_num} of {self.config.rounds}\n\n---\n\n")

        # Pre-round actions adjusting system prompt
        if round_num == 1:  # Add the first round system prompt postfix
            self.tell_bots_first_round()
        if round_num == self.config.rounds:  # Add the last round postfix
            self.tell_bots_last_round()

        # Run the round now that the system prompt is set
        self.run_round()

        # Post-round actions undoing system prompt adjustments
        if round_num == 1:  # Remove the first round system prompt postfix
            self.tell_bots_not_first_round()
        # if round_num == self.config.rounds then no need to remove the
        #   last round postfix since conversation is finished

    def run_round(self) -> None:
        """
        Run one round of responses from all bots.
        """
        logger.debug("Starting new conversation round")
        for bot in self.bots:
            try:
                # Initialize buffer to collect full response
                full_response: List[str] = []
                current_text = ""  # Remove bot name prefix, let bot handle it

                # Use rich.live to update markdown in real-time
                with Live(Markdown(current_text), refresh_per_second=4) as live:
                    # Use streaming response generation
                    for chunk in bot.stream_response(self.conversation):
                        full_response.append(chunk)
                        current_text += chunk
                        # Update the live display with markdown
                        live.update(Markdown(current_text))

                # Combine chunks into complete response
                response = "".join(full_response)  # No need to add bot name prefix

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
            self.display_text("\n\n---\n\n")
        logger.info("Round completed successfully")

    def tell_bots_first_round(self) -> None:
        """
        Inform bots that the conversation is about to start.
        """
        for bot in self.bots:
            suffix = self.replace_variables(
                self.config.first_round_postfix,
                {"bot_name": bot.name, "max_tokens": str(bot.model_max_tokens)},
            )
            self.system_prompt_add_suffix(bot, suffix)

    def tell_bots_not_first_round(self) -> None:
        """
        Remove the first round system prompt postfix from the system prompt.
        """
        for bot in self.bots:
            suffix = self.replace_variables(
                self.config.first_round_postfix,
                {"bot_name": bot.name, "max_tokens": str(bot.model_max_tokens)},
            )
            self.system_prompt_remove_suffix(bot, suffix)

    def tell_bots_last_round(self) -> None:
        """
        Inform bots that the conversation is about to end.
        """
        for bot in self.bots:
            suffix = self.replace_variables(
                self.config.last_round_postfix,
                {"bot_name": bot.name, "max_tokens": str(bot.model_max_tokens)},
            )
            self.system_prompt_add_suffix(bot, suffix)

    def replace_variables(self, text: str, variables: dict[str, str]) -> str:
        """
        Replace placeholders in the text with the provided variable values.

        Args:
            text (str): The text containing placeholders.
            variables (dict): A dictionary with variable names as keys and their corresponding values.
            e.g. {"bot_name": "GPT-4", "max_tokens": "100"} will replace "{bot_name}" with "GPT-4" and
            "{max_tokens}" with "100".

        Returns:
            str: The text with placeholders replaced by their values.
        """
        for key, value in variables.items():
            placeholder = f"{{{key}}}"
            text = text.replace(placeholder, str(value))
        return text

    def construct_system_prompt(
        self, shared_prefix: str, bot_config: ChatbotConfigData
    ) -> str:
        """
        Construct the system prompt for a bot based on the shared prefix and bot configuration.

        Args:
            shared_prefix (str): The shared prefix for the system prompt.
            bot_config: The configuration for the bot.

        Returns:
            str: The constructed system prompt.
        """
        if bot_config.bot_params_opt.max_tokens is None:
            max_tokens = DEFAULT_MAX_TOKENS
        else:
            max_tokens = bot_config.bot_params_opt.max_tokens

        bot_system_prompt = shared_prefix + bot_config.bot_prompt
        bot_system_prompt = self.replace_variables(
            bot_system_prompt,
            {"bot_name": bot_config.bot_name, "max_tokens": str(max_tokens)},
        )

        return bot_system_prompt

    def system_prompt_add_suffix(
        self, bot: ChatbotBase, additional_prompt: str
    ) -> None:
        """
        Append additional text to the system prompt.

        Args:
            additional_prompt (str): The text to append.
        """
        if additional_prompt:
            bot.system_prompt += additional_prompt

    def system_prompt_remove_suffix(
        self, bot: ChatbotBase, text_to_remove: str
    ) -> None:
        """
        Remove specific text from the end of the system prompt.

        Args:
            text_to_remove (str): The text to remove.
        """
        if text_to_remove and bot.system_prompt.endswith(text_to_remove):
            bot.system_prompt = bot.system_prompt[: -len(text_to_remove)]
