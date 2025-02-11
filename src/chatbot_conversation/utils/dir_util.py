"""
Utility module for managing configuration and output directories.

This module provides functions to determine appropriate directories for storing
configuration files and output data. It follows a priority-based approach to
locate these directories, considering environment variables, project structure,
and current working directory.
"""

import os
from pathlib import Path

from chatbot_conversation.utils.logging_util import LOGNAME_UTILS, get_logger

logger = get_logger(LOGNAME_UTILS)


CONFIG_DIR_ENV_VAR: str = "BOTCONV_CONFIG_DIR"
DEFAULT_CONFIG_DIR: str = "config"
OUTPUT_DIR_ENV_VAR: str = "BOTCONV_OUTPUT_DIR"
DEFAULT_OUTPUT_DIR: str = "output"
FILE_IN_PROJECT_ROOT: str = "pyproject.toml"


def get_config_dir() -> Path:
    """Determine and return the configuration directory path.

    This function follows a priority-based search to locate the configuration directory:
    1. Uses BOTCONV_CONFIG_DIR environment variable if set
    2. Uses 'config' directory in the project root if found
    3. Falls back to current working directory if above options fail

    Returns:
        Path: Directory path for configuration files, created if necessary
    """
    # creat_dir set to false since used to search for files that should already
    # exist in the config directory
    return _get_dir(CONFIG_DIR_ENV_VAR, FILE_IN_PROJECT_ROOT, DEFAULT_CONFIG_DIR, False)


def get_output_dir() -> Path:
    """Determine and return the output directory path.

    This function follows a priority-based search to locate the output directory:
    1. Uses BOTCONV_OUTPUT_DIR environment variable if set
    2. Creates/uses 'output' directory in the project root if found
    3. Falls back to current working directory if above options fail

    Returns:
        Path: Directory path for output files, created if necessary
    """
    return _get_dir(OUTPUT_DIR_ENV_VAR, FILE_IN_PROJECT_ROOT, DEFAULT_OUTPUT_DIR)


def _get_dir(
    env_var: str, target_file: str, default_dir: str, create_dir: bool = True
) -> Path:
    """Locate or create a directory based on priority rules.

    Args:
        env_var (str): Environment variable name to check for directory path
        target_file (str): Filename to look for when identifying project root
        default_dir (str): Default directory name to create under project root
        create_dir (bool): If True, creates directories if they don't exist.
                          If False, only uses existing directories. Defaults to True.

    Returns:
        Path: Directory path based on the following priority:
            1. Path from environment variable (if set and exists/creatable)
            2. Default directory under project root (if root found and exists/creatable)
            3. Current working directory (as fallback)

    Note:
        When create_dir is False, directories must exist to be used.
    """
    # First priority: Check environment variable
    env_dir = os.getenv(env_var)
    if env_dir is not None:
        dir_path = Path(env_dir)
        if dir_path.exists():
            logger.info(
                "Using existing environment variable setting %s with directory value: %s",
                env_var,
                dir_path,
            )
            return dir_path
        if create_dir:
            dir_path.mkdir(parents=True, exist_ok=True)
            logger.info(
                "Created directory from environment variable %s: %s",
                env_var,
                dir_path,
            )
            return dir_path
        logger.warning(
            "Environment variable %s points to non-existent path: %s",
            env_var,
            dir_path,
        )

    # Second priority: Try to find project root and use/create directory there
    current = Path.cwd()
    for parent in [current, *current.parents]:
        if (parent / target_file).exists():
            root_output = parent / default_dir
            if root_output.exists():
                logger.info(
                    "Using existing directory under project root: %s", root_output
                )
                return root_output
            if create_dir:
                root_output.mkdir(parents=True, exist_ok=True)
                logger.info("Created directory under project root: %s", root_output)
                return root_output
            logger.warning(
                "Project root directory %s does not contain %s", parent, default_dir
            )

    # Third priority: Use current directory
    logger.info("Using current directory: %s", current)
    return current


def path_is_simple_filename(filename: str) -> bool:
    """
    Check if the given filename is a simple filename (not a directory) without any path components.
    Had to take in filename as str since as a Path object, it would normalise the path giving
    a parent of '.' for a filename like 'file.txt' which is not what we want.

    Args:
        filename (str): The filename to check.

    Returns:
        bool: True if it is a simple filename, False otherwise.
    """
    # Reject empty filenames or special directory references
    if not filename or filename in {".", ".."}:
        return False

    # Reject filenames containing path separators
    if "/" in filename or "\\" in filename:
        return False

    # Reject filenames with Windows-style drive letters (e.g., "C:file.txt")
    if len(filename) > 2 and filename[1] == ":":
        return False  # Ensures it only matches cases like "C:file.txt"

    return True
