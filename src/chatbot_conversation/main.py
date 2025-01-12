"""
This module initializes and runs the chatbot conversation.
"""

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
        error_message = "Failed to set environment variables: %s" % str(e)
        print(error_message)
        logger.error(error_message)
        sys.exit(1)

    # Get config path from command line or use default
    config_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join("config", "config.json")

    try:
        # Load configuration and initialize conversation manager
        manager = ConversationManager(config_path)
    except Exception as e:
        error_message = "Failed to initialize conversation manager: %s" % str(e)
        print(error_message)
        logger.error(error_message)
        sys.exit(1)

    try:
        # Run conversation for configured number of rounds
        manager.run_conversation()
    except Exception as e:
        error_message = "Error during conversation: %s" % str(e)
        print(error_message)
        logger.error(error_message)
        sys.exit(1)


if __name__ == "__main__":
    main()
