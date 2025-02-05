"""
Logging utility module for configuring and retrieving application loggers.

This module provides functionality to set up logging configuration from a file
and retrieve named logger instances. It uses Python's built-in logging module
and supports configuration via a logging.conf file.
"""

import configparser
import logging
import logging.config
import os

from chatbot_conversation.utils.exceptions import ConfigurationException, ErrorSeverity

LOGNAME_API = "api"
LOGNAME_CONFIG = "config"
LOGNAME_MODEL = "model"
LOGNAME_SYSTEM = "system"
LOGNAME_VALIDATION = "validation"
LOGNAME_ROOT = "root"


def _validate_logger_config(config_file_path: str) -> None:
    """Validate that all LOGNAME constants have corresponding config sections."""
    parser = configparser.ConfigParser()
    parser.read(config_file_path)

    configured_loggers = parser.get("loggers", "keys").split(",")
    logger_constants = [
        LOGNAME_API,
        LOGNAME_CONFIG,
        LOGNAME_MODEL,
        LOGNAME_SYSTEM,
        LOGNAME_VALIDATION,
        LOGNAME_ROOT,
    ]

    for logger_name in logger_constants:
        if logger_name not in configured_loggers:
            raise ConfigurationException(
                message=f"Logger {logger_name} not configured in logging.conf",
                user_message="System configuration error",
                severity=ErrorSeverity.FATAL,
            )


# Set up logging from config file
config_path = os.path.join(os.path.dirname(__file__), "../../../config/logging.conf")
_validate_logger_config(config_path)
logging.config.fileConfig(config_path)


def get_logger(name: str) -> logging.Logger:
    """
    Get a configured logger instance with the specified name.

    Args:
        name (str): The name of the logger to retrieve. Usually __name__ of the module.

    Returns:
        logging.Logger: A configured logger instance.

    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("This is a log message")
    """
    return logging.getLogger(name)
