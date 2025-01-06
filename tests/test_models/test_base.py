"""Test module for the base classes and functionality of the chatbot conversation models.

This module contains test cases for the SystemPrompt class, verifying its initialization,
content management, and update tracking functionality.
"""

from chatbot_conversation.models.base import SystemPrompt


def test_system_prompt_initialization():
    """Test that SystemPrompt initializes with empty content and needs_update flag set."""
    prompt = SystemPrompt()
    assert prompt.content == ""
    assert prompt.needs_update is True


def test_system_prompt_content_property():
    """Test the content property getter and setter functionality."""
    prompt = SystemPrompt("Initial content")
    assert prompt.content == "Initial content"
    prompt.content = "Updated content"
    assert prompt.content == "Updated content"
    assert prompt.needs_update is True


def test_system_prompt_needs_update_property():
    """Test the needs_update property behavior when content is modified."""
    prompt = SystemPrompt("Initial content")
    assert prompt.needs_update is True
    prompt.mark_updated()
    assert prompt.needs_update is False


def test_system_prompt_mark_updated():
    """Test the mark_updated method for resetting the needs_update flag."""
    prompt = SystemPrompt("Initial content")
    prompt.mark_updated()
    assert prompt.needs_update is False


def test_system_prompt_add_suffix():
    """Test adding a suffix to the prompt content."""
    prompt = SystemPrompt("Initial content")
    prompt.add_suffix(" Suffix")
    assert prompt.content == "Initial content Suffix"
    assert prompt.needs_update is True


def test_system_prompt_remove_suffix():
    """Test removing a suffix from the prompt content."""
    prompt = SystemPrompt("Initial content Suffix")
    prompt.remove_suffix(" Suffix")
    assert prompt.content == "Initial content"
    assert prompt.needs_update is True


def test_system_prompt_add_prefix():
    """Test adding a prefix to the prompt content."""
    prompt = SystemPrompt("content")
    prompt.add_prefix("Prefix ")
    assert prompt.content == "Prefix content"
    assert prompt.needs_update is True


def test_system_prompt_remove_prefix():
    """Test removing a prefix from the prompt content."""
    prompt = SystemPrompt("Prefix content")
    prompt.remove_prefix("Prefix ")
    assert prompt.content == "content"
    assert prompt.needs_update is True
