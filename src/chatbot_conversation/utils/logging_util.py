"""
Logging utility module for configuring and retrieving application loggers.

This module provides functionality to set up logging configuration
and retrieve named logger instances. It uses Python's built-in logging module.
"""

import logging
import logging.config
from typing import Any, Dict

LOGNAME_API = "api"
LOGNAME_CONFIGURATION = "configuration"
LOGNAME_CONVERSATION = "conversation"
LOGNAME_MODELS = "models"
LOGNAME_ROOT = "root"
LOGNAME_SYSTEM = "system"
LOGNAME_UTILS = "utils"
LOGNAME_VALIDATION = "validation"

LOGGING_CONFIG: Dict[str, Any] = {
    "version": 1,
    "formatters": {
        "defaultFormatter": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        }
    },
    "handlers": {
        "consoleHandler": {
            "class": "logging.StreamHandler",
            "level": "CRITICAL",
            "formatter": "defaultFormatter",
            "stream": "ext://sys.stdout",
        },
        "fileHandler": {
            "class": "logging.FileHandler",
            "level": "INFO",
            "formatter": "defaultFormatter",
            "filename": "chatbot_conversation.log",
            "mode": "a",
        },
    },
    "loggers": {
        LOGNAME_ROOT: {"level": "INFO", "handlers": ["consoleHandler", "fileHandler"]},
        LOGNAME_API: {
            "level": "INFO",
            "handlers": ["consoleHandler", "fileHandler"],
            "propagate": False,
        },
        LOGNAME_CONFIGURATION: {
            "level": "INFO",
            "handlers": ["consoleHandler", "fileHandler"],
            "propagate": False,
        },
        LOGNAME_CONVERSATION: {
            "level": "INFO",
            "handlers": ["consoleHandler", "fileHandler"],
            "propagate": False,
        },
        LOGNAME_MODELS: {
            "level": "INFO",
            "handlers": ["consoleHandler", "fileHandler"],
            "propagate": False,
        },
        LOGNAME_SYSTEM: {
            "level": "INFO",
            "handlers": ["consoleHandler", "fileHandler"],
            "propagate": False,
        },
        LOGNAME_UTILS: {
            "level": "INFO",
            "handlers": ["consoleHandler", "fileHandler"],
            "propagate": False,
        },
        LOGNAME_VALIDATION: {
            "level": "INFO",
            "handlers": ["consoleHandler", "fileHandler"],
            "propagate": False,
        },
    },
}

# Configure logging using dictionary config
logging.config.dictConfig(LOGGING_CONFIG)

def get_logger(name: str) -> logging.Logger:
    """
    Get a configured logger instance with the specified name.

    Args:
        name (str): The name of the logger to retrieve. Usually __name__ of the module.

    Returns:
        logging.Logger: A configured logger instance.

    Raises:
        ValueError: If the requested logger name is not configured.

    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("This is a log message")
    """
    # Check if this is a explicitly configured logger name
    if name not in LOGGING_CONFIG["loggers"]:
        raise ValueError(
            f"Logger '{name}' is not currently supported. "
            f"Must be one of: {', '.join(LOGGING_CONFIG['loggers'].keys())}"
        )

    return logging.getLogger(name)
