"""
This module initializes and runs the chatbot conversation.
"""

import json
import logging
import os
import sys

from chatbot_conversation.conversation import ConversationManager
from chatbot_conversation.utils import APIConfig

logger = logging.getLogger("root")


def main() -> None:
    """
    Main function to set up environment variables, load configuration,
    initialize the conversation manager, and run the conversation.
    """
    try:
        # Set up environment variables for API access
        APIConfig.setup_env()
    except FileNotFoundError as e:
        error_message = f"Failed to set environment variables: {str(e)}"
        print(error_message)
        logger.error(error_message)
        sys.exit(1)

    # Get config path from command line or use default
    config_path = (
        sys.argv[1] if len(sys.argv) > 1 else os.path.join("config", "config.json")
    )

    try:
        # Load configuration and initialize conversation manager
        manager = ConversationManager(config_path)
    except FileNotFoundError as e:
        error_message = f"Configuration file not found: {str(e)}"
        print(error_message)
        logger.error(error_message)
        sys.exit(1)
    except json.JSONDecodeError as e:
        error_message = f"Error decoding JSON configuration: {str(e)}"
        print(error_message)
        logger.error(error_message)
        sys.exit(1)
    except ValueError as e:
        error_message = f"Value error loading configuration: {str(e)}"
        print(error_message)
        logger.error(error_message)
        sys.exit(1)
    except RuntimeError as e:
        error_message = f"Runtime error loading configuration: {str(e)}"
        print(error_message)
        logger.error(error_message)
        sys.exit(1)

    try:
        # Run conversation for configured number of rounds
        manager.run_conversation()
    except Exception as e:  # pylint: disable=broad-exception-caught
        error_message = f"Error during conversation: {str(e)}"
        print(error_message)
        logger.error(error_message)
        sys.exit(1)


if __name__ == "__main__":
    main()
