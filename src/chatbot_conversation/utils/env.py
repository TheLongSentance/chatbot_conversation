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

from chatbot_conversation.utils.dir_util import get_config_dir
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

        config_dir = get_config_dir()
        dotenv_path = config_dir / ".env"

        if os.path.exists(dotenv_path):
            load_dotenv(dotenv_path=dotenv_path)
            logger.info("Loaded environment from: %s", dotenv_path)
        else:
            logger.info("No .env file found in config directory with path: %s"
                        , dotenv_path)

        # Log available API-related environment variables without assuming which are required
        for env_var in os.environ:
            if env_var.endswith("_API_KEY"):
                logger.info("%s is set in environment", env_var)

    @staticmethod
    def get_config_dir_from_env() -> str | None:
        """Get the configuration directory path from environment or default.

        Returns:
            str: Path to configuration directory
        """
        return os.getenv(CONFIG_DIR_ENV_VAR)

    @staticmethod
    def get_default_config_dir() -> Path | None:
        """Get the default config directory relative to this module.

        Returns:
            Path: Path to default config directory, attempting to find project root's
                config dir, falling back to a relative path from this module if not found.
        """
        current = Path(__file__).resolve().parent
        # Try to find project root by walking up
        for parent in [current, *current.parents]:
            if (parent / FILE_IN_PROJECT_ROOT).exists():
                return parent / DEFAULT_CONFIG_DIR
        # None found so return None
        return None
