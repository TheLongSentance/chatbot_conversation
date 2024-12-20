"""
Logging utility module for configuring and retrieving application loggers.

This module provides functionality to set up logging configuration from a file
and retrieve named logger instances. It uses Python's built-in logging module
and supports configuration via a logging.conf file.
"""

import logging
import logging.config
import os

# Set up logging from config file
config_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "logging.conf"
)

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
