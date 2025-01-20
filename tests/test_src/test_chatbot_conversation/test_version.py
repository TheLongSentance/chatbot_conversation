from chatbot_conversation.version import __version__

def test_version() -> None:
    """Test that the version string is correctly defined."""
    assert __version__ == "1.0.0"