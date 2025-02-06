"""
This module initializes and runs the chatbot conversation.
"""

import logging
import os
import sys

from chatbot_conversation.conversation import ConversationManager
from chatbot_conversation.error import handle_error
from chatbot_conversation.utils import LOGNAME_ROOT, APIConfig

logger = logging.getLogger(LOGNAME_ROOT)


def main() -> None:
    """
    Main function to set up environment variables, load configuration,
    initialize the conversation manager, and run the chatbot conversation.

    It provides user-friendly error messages for various failure scenarios
    while ensuring all errors are properly logged for debugging.
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
        sys.exit(0)
    except Exception as e:  # pylint: disable=broad-except
        sys.exit(handle_error(e))


if __name__ == "__main__":
    main()
