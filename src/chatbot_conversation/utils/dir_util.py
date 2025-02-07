"""
This module contains utility functions for finding config and output directories.
"""

import os
from pathlib import Path

from chatbot_conversation.utils.logging_util import LOGNAME_UTILS, get_logger

logger = get_logger(LOGNAME_UTILS)


CONFIG_DIR_ENV_VAR: str = "BOTCONV_CONFIG_DIR"
DEFAULT_CONFIG_DIR: str = "config"
FILE_IN_PROJECT_ROOT: str = "pyproject.toml"


def get_config_dir() -> Path:
    """Get the directory to search for config files.

    Tries the following locations in order:
    1. Directory specified in BOTCONV_CONFIG_DIR environment variable (creates if needed)
    2. 'config' directory under project root (creates if project root found)
    3. Current directory as fallback

    Returns:
        Path: Directory path where configuration files should be found
    """
    # First priority: Check environment variable
    env_dir = os.getenv(CONFIG_DIR_ENV_VAR)
    if env_dir is not None:
        dir_path = Path(env_dir)
        dir_path.mkdir(parents=True, exist_ok=True)
        logger.info(
            "Using environment variable setting %s as config directory: %s",
            CONFIG_DIR_ENV_VAR, dir_path
        )
        return dir_path

    # Second priority: Try to find project root and use/create config directory there
    current = Path.cwd()
    for parent in [current, *current.parents]:
        if (parent / FILE_IN_PROJECT_ROOT).exists():
            root_config = parent / DEFAULT_CONFIG_DIR
            root_config.mkdir(parents=True, exist_ok=True)
            logger.info(
                "Using /config/ directory under project root%s", root_config
            )
            return root_config

    # Third priority: Use current directory
    logger.info("Using current directory: %s", current)
    return current
