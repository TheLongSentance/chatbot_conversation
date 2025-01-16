"""
This package contains the modules for the chatbot conversation.
"""

import importlib

from .version import __version__

# List of sub-packages to import
sub_packages = [
    "chatbot_conversation.conversation",
    "chatbot_conversation.models",
    "chatbot_conversation.utils",
]

# Dynamically import sub-packages and update __all__
__all__ = ["__version__"]
for package in sub_packages:
    module = importlib.import_module(package)
    __all__.extend(module.__all__)
