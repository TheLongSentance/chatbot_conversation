"""
This module handles transcript management and saving conversation logs to file
using the TranscriptManager for the provided conversation configuration.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import List, Set, TextIO

from chatbot_conversation.conversation.loader import ConversationConfig
from chatbot_conversation.models.base import ConversationMessage
from chatbot_conversation.utils import (
    ErrorSeverity,
    SystemException,
    get_logger,
    get_output_dir,
)

# specific import path to avoid circular from package:
# from ..version import __version__
from chatbot_conversation.version import __version__

# Update constants
TRANSCRIPT_FILE_STUB: str = "transcript_"
FILE_IN_PROJECT_ROOT: str = "pyproject.toml"  # Same as in env.py

logger = get_logger("conversation")


def save_transcript(
    conversation: List[ConversationMessage],
    config: ConversationConfig,
    config_path: Path,
) -> Path:
    """Save conversation transcript to a file.

    Args:
        transcript_dir: Path to the transcript directory
        conversation: List of conversation messages
        author: Author of the conversation
        num_rounds: Total number of conversation rounds
        num_bots: Number of participating bots
        config_path: Path to the configuration file

    Returns:
        Path to the saved transcript file

    Raises:
        IOError: If file operations fail
    """
    # Get output directory
    output_dir = get_output_dir()

    # Generate timestamp for unique filename
    timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
    file_path = output_dir / f"{TRANSCRIPT_FILE_STUB}{timestamp}.md"

    # Extract metadata from the configuration
    num_rounds = config.rounds
    num_bots = len(config.bots)

    # Get set of hidden moderator message rounds
    hidden_moderator_rounds: Set[int] = {
        msg.round_number for msg in config.moderator_messages_opt if not msg.display_opt
    }

    try:
        with open(file_path, "w", encoding="utf-8") as file:
            # Write conversation title
            file.write(f"# {conversation[0]['content']}\n\n")

            # Write conversation content
            round_num = 1
            bot_round_count = 0
            announce_round = True
            for message in conversation[1:]:

                if announce_round:  # if new round
                    file.write(f"## Round {round_num} of {num_rounds}\n\n")
                    announce_round = False  # reset announcement flag

                if message["bot_index"] != 0:  # if a bot not moderator

                    # Write bot message
                    file.write(f"{message['content']}\n\n---\n\n")

                    bot_round_count += 1  # count bots in round so far
                    if bot_round_count == num_bots:  # if all bots have responded in round
                        round_num += 1  # increment round number
                        announce_round = True  # announce this new round in next iteration
                        bot_round_count = 0  # reset bot count

                # else must be a moderator message so immediately test whether to display
                elif round_num not in hidden_moderator_rounds:
                    file.write(f"{message['content']}\n\n---\n\n")

            # Write metadata
            _write_metadata(file, config, config_path)

        logger.info("Conversation saved to %s", file_path)
        return file_path

    except IOError as e:
        raise SystemException(
            message=f"Failed to write conversation transcript: {str(e)}",
            user_message=(
                "Unable to save the conversation. "
                "Please check if you have write permissions for the output directory."
            ),
            severity=ErrorSeverity.ERROR,
            original_error=e,
        ) from e


def _write_metadata(
    file: TextIO,
    config: ConversationConfig,
    config_path: Path,
) -> None:
    """Write conversation metadata to the transcript file.

    Args:
        file: Open file object for writing
        config: Conversation configuration
        config_path: Path to the configuration file
    """

    # Extract metadata from the configuration
    num_rounds = config.rounds
    num_bots = len(config.bots)
    author = config.author

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    file.write(
        f"## Conversation Finished - {num_rounds} Rounds With "
        f"{num_bots} Bots Completed!\n\n"
        f"## *Conversation Generated* : {now}\n\n"
        f"## *Software Version* : {__version__}\n\n"
        f"## *Configuration Author* : {author}\n\n"
        f"## *Configuration File* : {config_path}\n\n"
        f"```json\n{json.dumps(config.model_dump(), indent=4)}\n```\n"
    )
