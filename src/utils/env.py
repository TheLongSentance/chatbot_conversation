from dotenv import load_dotenv
import os
import logging
from typing import List, Tuple, Optional

class APIConfig:
    @staticmethod
    def setup_env():
        """Set up environment variables and validate keys."""
        APIConfig._load_config()

    @staticmethod
    def _load_config():
        """Initialize and validate API keys"""
        load_dotenv()
        keys = [
            ("OpenAI", os.getenv('OPENAI_API_KEY')),
            ("Anthropic", os.getenv('ANTHROPIC_API_KEY')),
            ("Google", os.getenv('GOOGLE_API_KEY')),
        ]
        APIConfig._validate_keys(keys)

    @staticmethod
    def _validate_keys(keys: List[Tuple[str, Optional[str]]]):
        """Validate presence of required API keys"""
        for service, key in keys:
            APIConfig._log_key_status(service, key)

    @staticmethod
    def _log_key_status(service: str, key: Optional[str]):
        """Log the status of the API key"""
        if not key:
            logging.warning(f"{service} API Key not set")
        else:
            logging.info(f"{service} API Key exists and begins {key[:8]}")


# Usage:
# APIConfig.setup_env()