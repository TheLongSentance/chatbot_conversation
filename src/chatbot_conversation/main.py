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
        logger.error(str(e))
        sys.exit(1)

    # Load configuration and initialize conversation manager
    manager = ConversationManager(os.path.join("config", "config.json"))

    # Run conversation for configured number of rounds
    manager.run_conversation()


if __name__ == "__main__":
    main()
