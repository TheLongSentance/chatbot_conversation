"""Unit tests for logging utility module."""

import logging
from chatbot_conversation.utils.logging_util import get_logger


def test_get_logger(mock_logging_config: None) -> None:
    """Test getting a logger instance.
    
    Args:
        mock_logging_config: Fixture setting up mock logging configuration

    Returns:
        None
    """
    logger = get_logger("testLogger")
    assert isinstance(logger, logging.Logger)
    assert logger.name == "testLogger"


def test_logger_level(mock_logging_config: None) -> None:
    """Test logger level configuration.
    
    Args:
        mock_logging_config: Fixture setting up mock logging configuration

    Returns:
        None
    """
    logger = get_logger("testLogger")
    assert logger.level == logging.DEBUG


def test_logger_handler(mock_logging_config: None) -> None:
    """Test logger handler configuration.
    
    Args:
        mock_logging_config: Fixture setting up mock logging configuration

    Returns:
        None
    """
    logger = get_logger("testLogger")
    assert len(logger.handlers) == 1
    assert isinstance(logger.handlers[0], logging.StreamHandler)
