"""
This module contains the APIConfig class, which is responsible for setting up
and validating environment variables for API keys.

The APIConfig class handles:
- Loading environment variables from a .env file
- Validating the presence of required API keys
- Logging the status of the API keys
"""

import logging
import os
from typing import List, Optional, Tuple

from dotenv import load_dotenv


class APIConfig:  # pylint: disable=too-few-public-methods
    """Class responsible for setting up and validating environment variables for API keys."""

    @staticmethod
    def setup_env() -> None:
        """Set up environment variables and validate keys."""
        APIConfig._load_config()

    @staticmethod
    def _load_config() -> None:
        """Initialize and validate API keys."""
        dotenv_path = os.path.join(os.path.dirname(__file__), "../../../config/.env")

        if not os.path.exists(dotenv_path):
            raise FileNotFoundError(f".env file not found at path: {dotenv_path}")

        load_dotenv(dotenv_path=dotenv_path)

        keys = [
            ("OpenAI", os.getenv("OPENAI_API_KEY")),
            ("Anthropic", os.getenv("ANTHROPIC_API_KEY")),
            ("Google", os.getenv("GOOGLE_API_KEY")),
        ]
        APIConfig._validate_keys(keys)

    @staticmethod
    def _validate_keys(keys: List[Tuple[str, Optional[str]]]) -> None:
        """Validate presence of required API keys.

        Args:
            keys: List of tuples containing service name and API key
        """
        for service, key in keys:
            APIConfig._log_key_status(service, key)

    @staticmethod
    def _log_key_status(service: str, key: Optional[str]) -> None:
        """Log the status of the API key.

        Args:
            service: Name of the service
            key: API key for the service
        """
        if not key:
            logging.warning("%s API Key not set", service)
        else:
            logging.info("%s API Key exists and begins %s", service, key[:8])
