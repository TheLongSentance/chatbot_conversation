"""
Unit tests for the RichDisplayManager class in display.py.
"""

from typing import Generator
from _pytest.capture import CaptureFixture
from chatbot_conversation.conversation.display import RichDisplay

def test_show_text(display: RichDisplay, capsys: CaptureFixture[str]) -> None:
    """
    Test that show_text prints the expected text.
    """
    sample_text: str = "Hello, Test!"
    display.show_text(sample_text)
    captured = capsys.readouterr()
    assert sample_text in captured.out

def test_show_streaming_text(display: RichDisplay, capsys: CaptureFixture[str]) -> None:
    """
    Test that show_streaming_text returns the combined chunks and prints them.
    """
    chunks = ["Chunk1", "Chunk2", "Chunk3"]
    def gen() -> Generator[str, None, None]:
        for c in chunks:
            yield c

    result = display.show_streaming_text(gen())
    assert result == "".join(chunks)
    captured = capsys.readouterr()
    # Optionally check partial output, but here we ensure the first chunk is seen
    assert "Chunk1" in captured.out

