"""
Unit tests for the TranscriptManager class in transcript.py.
"""

from pathlib import Path
from unittest.mock import patch, mock_open
from chatbot_conversation.conversation.transcript import TranscriptManager
from chatbot_conversation.models import ConversationMessage
from chatbot_conversation.conversation.loader import ConversationConfig


def test_save_transcript(
    sample_conversation_data: list[ConversationMessage],
    sample_conversation_config: ConversationConfig,
    tmp_path: Path,
) -> None:
    """
    Test saving a transcript to a file.
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

            # Check if the file was created in the correct directory
            assert file_path.parent == Path("./output/")
            assert file_path.name.startswith("transcript_")
            assert file_path.suffix == ".md"

            # Check if the file was written to
            mocked_file.assert_called_once_with(file_path, "w", encoding="utf-8")
            handle = mocked_file()
            handle.write.assert_any_call("# Test seed message\n\n")
            handle.write.assert_any_call("## Round 1 of 2\n\n")
            handle.write.assert_any_call("Bot1 response\n\n---\n\n")
            handle.write.assert_any_call("Bot2 response\n\n---\n\n")
