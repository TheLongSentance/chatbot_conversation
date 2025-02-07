"""
Subpackage for utilities used in the chatbot_conversation package.
"""

from chatbot_conversation.utils.dir_util import get_config_dir
from chatbot_conversation.utils.env import APIConfig
from chatbot_conversation.utils.exceptions import (
    APIException,
    ChatbotException,
    ConfigurationException,
    ErrorSeverity,
    ModelException,
    SystemException,
    ValidationException,
    handle_pydantic_validation_errors,
)
from chatbot_conversation.utils.logging_util import (
    LOGNAME_API,
    LOGNAME_CONFIGURATION,
    LOGNAME_CONVERSATION,
    LOGNAME_MODELS,
    LOGNAME_ROOT,
    LOGNAME_SYSTEM,
    LOGNAME_UTILS,
    LOGNAME_VALIDATION,
    get_logger,
)

__all__ = [
    "APIConfig",
    "get_logger",
    "LOGNAME_API",
    "LOGNAME_CONFIGURATION",
    "LOGNAME_CONVERSATION",
    "LOGNAME_MODELS",
    "LOGNAME_SYSTEM",
    "LOGNAME_UTILS",
    "LOGNAME_VALIDATION",
    "LOGNAME_ROOT",
    "APIException",
    "ChatbotException",
    "ConfigurationException",
    "ErrorSeverity",
    "ModelException",
    "SystemException",
    "ValidationException",
    "handle_pydantic_validation_errors",
    "get_config_dir",
]
