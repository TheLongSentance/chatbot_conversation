"""
This module contains the APIConfig class, which is responsible for setting up
and validating environment variables for API keys.

The APIConfig class handles:
- Loading environment variables from a .env file
- Validating the presence of required API keys
- Logging the status of the API keys
"""

import os
from pathlib import Path

from dotenv import load_dotenv

from chatbot_conversation.utils.logging_util import LOGNAME_CONFIGURATION, get_logger

FILE_IN_PROJECT_ROOT = "pyproject.toml"
DEFAULT_CONFIG_DIR = "config"
CONFIG_DIR_ENV_VAR = "BOTCONV_CONFIG_DIR"

logger = get_logger(LOGNAME_CONFIGURATION)


class APIConfig:  # pylint: disable=too-few-public-methods
    """Class responsible for setting up and validating environment variables for API keys."""

    @staticmethod
    def setup_env() -> None:
        """Set up environment variables and validate keys."""
        APIConfig._load_config()

    @staticmethod
    def _load_config() -> None:
        """Initialize environment by loading .env file if present.

        Attempts to load .env file to supplement any environment variables.
        Does not enforce any specific keys as requirements depend on dynamic configuration.
        """

        current = Path.cwd()
        dotenv_path = current / ".env"

        if os.path.exists(dotenv_path):
            load_dotenv(dotenv_path=dotenv_path)
            logger.info("Loaded environment from: %s", dotenv_path)
        else:  # Only log info not debug message - environment could be set already
            logger.info("No .env file found in current directory with path: %s", dotenv_path)

        # Log available API-related environment variables without assuming which are required
        for env_var in os.environ:
            if env_var.endswith("_API_KEY"):
                logger.info("%s is set in environment", env_var)
