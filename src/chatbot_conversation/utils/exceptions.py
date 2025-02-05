"""
Custom exceptions for the chatbot conversation system.

This module provides a set of custom exceptions for handling various error 
conditions in the chatbot system. Each exception includes severity level 
classification and explicit retry settings.

Severity Levels:
    WARNING: Issues that don't prevent core functionality
    ERROR: Serious issues that impact functionality but aren't fatal
    FATAL: Critical issues that prevent system operation

Example:
    try:
        raise APIError(
            message="API request failed",
            user_message="Service temporarily unavailable",
            severity=ErrorSeverity.ERROR
        )
    except ChatbotError as e:
        print(f"Severity: {e.severity}")
        print(f"User message: {e.user_message}")
"""

from dataclasses import dataclass
from enum import Enum, auto

# Add after existing imports
from functools import wraps
from typing import Any, Callable, Optional, TypeVar

from pydantic import ValidationError

T = TypeVar("T")


# Add after ErrorSeverity class but before exception classes
def handle_pydantic_validation_errors(f: Callable[..., T]) -> Callable[..., T]:
    """Convert Pydantic ValidationErrors into ValidationException.

    Args:
        f: Function to wrap with validation error handling

    Returns:
        Wrapped function that converts ValidationError to ValidationException
    """

    @wraps(f)
    def wrapper(*args: Any, **kwargs: Any) -> T:
        try:
            return f(*args, **kwargs)
        except ValidationError as e:
            error_details: list[str] = []
            for error in e.errors():
                loc = " -> ".join(str(item) for item in error["loc"])
                msg = error["msg"]
                error_details.append(f"{loc}: {msg}")

            error_msg = "\n".join(error_details)

            raise ValidationException(
                message=f"Validation failed: {error_msg}",
                user_message="The provided data failed validation checks. Please verify all fields meet the requirements.",
                severity=ErrorSeverity.ERROR,
                original_error=e,
            ) from e

    return wrapper


class ErrorSeverity(Enum):
    """Classification of error severity levels."""

    WARNING = auto()  # Issue exists but operation can continue
    ERROR = auto()  # Serious issue impacting functionality
    FATAL = auto()  # Critical issue preventing operation


@dataclass
class ChatbotException(Exception):
    """Base class for all chatbot-related exceptions.

    Attributes:
        message: Technical error message for logging
        user_message: User-friendly error message
        severity: Classification of error severity
        original_error: Original exception that caused this error
    """

    message: str
    user_message: str
    severity: ErrorSeverity
    original_error: Optional[Exception] = None

    def __str__(self) -> str:
        """Return the technical error message with traceback info if available."""
        if self.original_error:
            return f"{self.message} (caused by: {self.original_error})"
        return self.message


@dataclass
class APIException(ChatbotException):
    """Errors related to external API communication.

    Examples:
        - Network connectivity issues
        - API timeouts
        - Rate limiting
        - Service unavailable
    """

    def __init__(
        self,
        message: str,
        user_message: str = (
            "There was a problem communicating with the AI service. "
            "Please try again in a few moments."
        ),
        severity: ErrorSeverity = ErrorSeverity.ERROR,
        original_error: Optional[Exception] = None,
    ):
        super().__init__(message, user_message, severity, original_error)


@dataclass
class ConfigurationException(ChatbotException):
    """Errors related to system configuration.

    Examples:
        - Missing or invalid configuration values
        - Incompatible settings
        - Resource allocation issues
    """

    def __init__(
        self,
        message: str,
        user_message: str = "There is a problem with the system configuration.",
        severity: ErrorSeverity = ErrorSeverity.FATAL,
        original_error: Optional[Exception] = None,
    ):
        super().__init__(message, user_message, severity, original_error)


@dataclass
class ModelException(ChatbotException):
    """Errors specific to AI model operation.

    Examples:
        - Token limit exceeded
        - Content filtering triggered
        - Model-specific limitations
    """

    def __init__(
        self,
        message: str,
        user_message: str = "The AI model encountered a limitation. Try simplifying your request.",
        severity: ErrorSeverity = ErrorSeverity.ERROR,
        original_error: Optional[Exception] = None,
    ):
        super().__init__(message, user_message, severity, original_error)


@dataclass
class SystemException(ChatbotException):
    """Critical system-level errors.

    Examples:
        - Unrecoverable system state
        - Security violations
        - Resource exhaustion
    """

    def __init__(
        self,
        message: str,
        user_message: str = "A critical system error has occurred.",
        severity: ErrorSeverity = ErrorSeverity.FATAL,
        original_error: Optional[Exception] = None,
    ):
        super().__init__(message, user_message, severity, original_error)


@dataclass
class ValidationException(ChatbotException):
    """Errors related to data validation.

    Examples:
        - Invalid input formats
        - Data constraint violations
        - Schema validation failures
    """

    def __init__(
        self,
        message: str,
        user_message: str = "The provided data does not meet the required format or constraints.",
        severity: ErrorSeverity = ErrorSeverity.ERROR,
        original_error: Optional[Exception] = None,
    ):
        super().__init__(message, user_message, severity, original_error)
