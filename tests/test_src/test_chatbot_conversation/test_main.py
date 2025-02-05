"""Test module for the main application entry point.

This module contains test cases for the main application functionality, including:
- Configuration loading and initialization
- Command-line argument handling
- Exception handling and logging
- Integration with ConversationManager and APIConfig

The tests use mock objects to avoid actual file operations and API calls.
"""

import os
import sys
from unittest.mock import Mock

import pytest
from _pytest.logging import LogCaptureFixture
from pytest import MonkeyPatch

from chatbot_conversation.main import main

# Remove fixture definitions for mock_conversation_manager and mock_api_config
# as they are now in conftest.py


def test_main_with_default_config(
    mock_conversation_manager: Mock, mock_api_config: Mock, monkeypatch: MonkeyPatch
) -> None:
    """Test the main function using the default configuration path."""
    # Setup
    manager_instance: Mock = mock_conversation_manager.return_value
    monkeypatch.setattr(sys, "argv", ["script"])

    # Execute
    with pytest.raises(SystemExit) as exc_info:
        main()

    # Verify
    assert exc_info.value.code == 0  # Verify successful exit
    mock_api_config.setup_env.assert_called_once()
    mock_conversation_manager.assert_called_once_with(
        os.path.join("config", "config.json")
    )
    manager_instance.run_conversation.assert_called_once()


def test_main_with_custom_config(
    mock_conversation_manager: Mock, mock_api_config: Mock, monkeypatch: MonkeyPatch
) -> None:
    """Test the main function using a custom configuration path from command line.

    Verifies that the main function correctly:
    1. Sets up the environment
    2. Uses the provided command-line config path
    3. Initializes ConversationManager with custom path
    4. Runs the conversation
    5. Exits with success code 0

    Args:
        mock_conversation_manager: Mock object for ConversationManager
        mock_api_config: Mock object for APIConfig
        monkeypatch: Pytest fixture for modifying sys.argv

    Raises:
        AssertionError: If any of the expected method calls are not made
    """
    # Setup
    test_config: str = "custom_config.json"
    monkeypatch.setattr(sys, "argv", ["script", test_config])
    manager_instance: Mock = mock_conversation_manager.return_value

    # Execute and verify SystemExit
    with pytest.raises(SystemExit) as exc_info:
        main()

    # Verify exit code and mock calls
    assert exc_info.value.code == 0
    mock_api_config.setup_env.assert_called_once()
    mock_conversation_manager.assert_called_once_with(test_config)
    manager_instance.run_conversation.assert_called_once()


def test_main_handles_exceptions(
    mock_conversation_manager: Mock, mock_api_config: Mock, caplog: LogCaptureFixture
) -> None:
    """Test the main function's exception handling capabilities.

    Verifies that the main function correctly:
    1. Catches and logs unexpected exceptions
    2. Exits with status code 4
    3. Prints appropriate error message

    Args:
        mock_conversation_manager: Mock object for ConversationManager
        mock_api_config: Mock object for APIConfig
        caplog: Pytest fixture for capturing log output

    Raises:
        AssertionError: If the exception is not handled as expected
    """
    # Setup
    test_error_msg: str = "Test error"
    mock_conversation_manager.side_effect = Exception(test_error_msg)

    # Execute and verify SystemExit
    with pytest.raises(SystemExit) as exc_info:
        main()

    # Verify exit code, error logging and messages
    assert exc_info.value.code == 4  # Update expected exit code to 4
    assert "An unexpected error occurred" in caplog.text
    assert test_error_msg in caplog.text
    mock_api_config.setup_env.assert_called_once()
