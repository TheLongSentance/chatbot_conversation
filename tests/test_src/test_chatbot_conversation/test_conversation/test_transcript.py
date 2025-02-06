"""
Unit tests for the TranscriptManager class in the chatbot_conversation package.
"""

import json
from pathlib import Path
from unittest.mock import mock_open, patch

import pytest

from chatbot_conversation.conversation.loader import ConversationConfig
from chatbot_conversation.conversation.transcript import TranscriptManager
from chatbot_conversation.models import ConversationMessage
from chatbot_conversation.version import __version__


def test_save_transcript(
    sample_conversation_data: list[ConversationMessage],
    sample_conversation_config: ConversationConfig,
    tmp_path: Path,
) -> None:
    """
    Test saving a transcript to a file.

    Args:
        sample_conversation_data (list[ConversationMessage]): Sample conversation data.
        sample_conversation_config (ConversationConfig): Sample conversation configuration.
        tmp_path (Path): Temporary path for output files.
    """
    config_path = "test_config_path.json"
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    with patch("builtins.open", mock_open()) as mocked_file:
        with patch("pathlib.Path.mkdir"):
            file_path = TranscriptManager.save_transcript(
                conversation=sample_conversation_data,
                config=sample_conversation_config,
                config_path=config_path,
            )

            # Basic file checks
            assert file_path.parent.resolve() == Path("./output/").resolve()
            assert file_path.name.startswith("transcript_")
            assert file_path.suffix == ".md"

            # File opening check
            mocked_file.assert_called_once_with(file_path, "w", encoding="utf-8")

            # Collect all write calls
            handle = mocked_file()
            write_calls = [call[0][0] for call in handle.write.call_args_list]
            full_output = "".join(write_calls)

            # Content verification
            required_content = [
                "# Test seed message",
                "## Round 1 of 2",
                "Bot1 response",
                "Bot2 response",
                "## Conversation Finished - 2 Rounds With 2 Bots Completed!",
                "## *Configuration Author* : Test Author",
                f"## *Configuration File* : {config_path}",
            ]

            for content in required_content:
                assert content in full_output, f"Missing content: {content}"

            # Config JSON verification
            assert (
                json.dumps(sample_conversation_config.model_dump(), indent=4)
                in full_output
            )

def test_transcript_with_hidden_moderator(
    sample_conversation_data: list[ConversationMessage],
    sample_conversation_config: ConversationConfig,
    tmp_path: Path,
) -> None:
    """Test that hidden moderator messages are not included in transcript."""
    config_path = "test_config.json"
    
    # Add a hidden moderator message to conversation data
    hidden_msg = ConversationMessage(bot_index=0, content="Hidden message")
    conversation_data = sample_conversation_data + [hidden_msg]
    
    with patch("builtins.open", mock_open()) as mocked_file:
        TranscriptManager.save_transcript(
            conversation=conversation_data,
            config=sample_conversation_config,
            config_path=config_path,
        )
        
        handle = mocked_file()
        write_calls = [call[0][0] for call in handle.write.call_args_list]
        full_output = "".join(write_calls)
        
        assert "Hidden message" not in full_output

def test_get_transcript_dir_env(env_transcript_dir: Path) -> None:
    """Test transcript directory from environment variable."""
    dir_path = TranscriptManager.get_transcript_dir()
    assert dir_path == env_transcript_dir
    assert dir_path.exists()

def test_get_transcript_dir_project_root(
    mock_project_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test transcript directory in project root."""
    monkeypatch.chdir(mock_project_root)
    dir_path = TranscriptManager.get_transcript_dir()
    assert dir_path == mock_project_root / "output"
    assert dir_path.exists()

def test_get_transcript_dir_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test fallback to local output directory."""
    monkeypatch.chdir(tmp_path)
    dir_path = TranscriptManager.get_transcript_dir()
    assert dir_path == tmp_path / "output"
    assert dir_path.exists()
