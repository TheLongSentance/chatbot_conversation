"""Unit tests for logging utility module."""

import logging

import pytest

from chatbot_conversation.utils.logging_util import (
    LOGNAME_API,
    LOGNAME_CONFIGURATION,
    LOGNAME_MODELS,
    get_logger,
)


def test_get_logger(mock_logging_config: None) -> None:
    """Test getting a valid logger instance.

    Args:
        mock_logging_config: Fixture setting up mock logging configuration

    Returns:
        None
    """
    logger = get_logger(LOGNAME_API)
    assert isinstance(logger, logging.Logger)
    assert logger.name == LOGNAME_API


def test_invalid_logger_name(mock_logging_config: None) -> None:
    """Test getting logger with invalid name raises ValueError.

    Args:
        mock_logging_config: Fixture setting up mock logging configuration

    Returns:
        None
    """
    with pytest.raises(ValueError) as exc_info:
        get_logger("invalid_logger_name")
    assert "is not currently supported" in str(exc_info.value)


def test_logger_propagation(mock_logging_config: None) -> None:
    """Test logger propagation configuration.

    Args:
        mock_logging_config: Fixture setting up mock logging configuration

    Returns:
        None
    """
    logger = get_logger(LOGNAME_MODELS)
    assert not logger.propagate


def test_logger_handlers(mock_logging_config: None) -> None:
    """Test logger handler configuration.

    Args:
        mock_logging_config: Fixture setting up mock logging configuration

    Returns:
        None
    """
    logger = get_logger(LOGNAME_CONFIGURATION)
    assert len(logger.handlers) > 0
    assert any(isinstance(handler, logging.StreamHandler) for handler in logger.handlers)
