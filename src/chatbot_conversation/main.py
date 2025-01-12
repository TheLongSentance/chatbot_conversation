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
    initialize the conversation manager, and run the chatbot conversation.

    It handles any unexpected exceptions by logging the error and exiting the program.
    """
    try:
        # Set up environment variables for API access
        APIConfig.setup_env()

        # Load configuration and initialize conversation manager
        config_path = (
            sys.argv[1] if len(sys.argv) > 1 else os.path.join("config", "config.json")
        )
        manager = ConversationManager(config_path)

        # Run conversation
        manager.run_conversation()
    except Exception as e:  # pylint: disable=broad-exception-caught
        # Global error handling

        print("An unexpected error occurred. Please check the logs for details.")    
        
        logger.error("An unexpected error occurred: %s", str(e), exc_info=True)
                
        sys.exit(1)


if __name__ == "__main__":
    main()
