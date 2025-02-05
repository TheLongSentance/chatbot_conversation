from typing import Dict, Type

from chatbot_conversation.utils import (
    LOGNAME_API,
    LOGNAME_CONFIG,
    LOGNAME_MODEL,
    LOGNAME_ROOT,
    LOGNAME_SYSTEM,
    LOGNAME_VALIDATION,
    APIException,
    ChatbotException,
    ConfigurationException,
    ErrorSeverity,
    ModelException,
    SystemException,
    ValidationException,
    get_logger,
)

LOGGER_MAPPING: Dict[Type[Exception], str] = {
    APIException: LOGNAME_API,
    ConfigurationException: LOGNAME_CONFIG,
    ModelException: LOGNAME_MODEL,
    SystemException: LOGNAME_SYSTEM,
    ValidationException: LOGNAME_VALIDATION,
}

def handle_error(error: Exception) -> int:
    """
    Handle errors by logging them and providing user-friendly messages.

    This function processes exceptions, logs appropriate messages, and determines
    the exit status code based on the error severity.

    Args:
        error (Exception): The exception to be handled. Can be a ChatbotException 
            or any other Exception type.

    Returns:
        int: Exit status code indicating the severity of the error:
            - 0: Success (no error)
            - 1: Warning level errors (recoverable)
            - 2: Error level errors (potentially recoverable)
            - 3: Fatal errors (non-recoverable)
            - 4: Unexpected errors (system/unknown errors)

    Note:
        For ChatbotException types, the function logs both technical details and 
        user-friendly messages. For unexpected errors, it logs full stack traces
        and provides a generic user message.
    """

    # Safe lookup with default to "root"
    logger_name = LOGGER_MAPPING.get(type(error), LOGNAME_ROOT)
    logger = get_logger(logger_name)

    if isinstance(error, ChatbotException):
        # Log the technical message
        logger.error(
            "Error occurred: %s", error.message, exc_info=error.original_error or error
        )

        # # Display user-friendly message with retry information
        # retry_msg = " You may try the operation again." if error.retry_allowed else ""
        # print(f"\nError: {error.user_message}{retry_msg}")
        print(f"\nError: {error.user_message}")

        # Exit with appropriate code based on severity
        exit_codes = {
            ErrorSeverity.WARNING: 1,
            ErrorSeverity.ERROR: 2,
            ErrorSeverity.FATAL: 3,
        }
        return exit_codes[error.severity]
    else:
        # Unexpected error - log full details
        logger.error("An unexpected error occurred: %s", str(error), exc_info=True)
        print(
            "\nAn unexpected error occurred please review the application "
            "logs for more information."
        )
        return 4
