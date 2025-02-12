"""
Configuration Editor for Chatbot Settings.

This module provides functionality to update bot configuration settings in JSON files.
It allows batch updates of bot_type and bot_version for all bots defined in a
configuration file.

Example:
    To update bot configurations from command line:
        $ python edit_config.py config.json "new_type" "v2.0"
"""

import json
import sys


def update_bot_config(
    config_file: str, new_bot_type: str, new_bot_version: str
) -> None:
    """
    Update all bot_type and bot_version values in a config file.

    This function reads a JSON configuration file, updates the bot_type and
    bot_version for all bots defined in the configuration, and writes the
    changes back to the file.

    Args:
        config_file (str): Path to the config JSON file.
        new_bot_type (str): New bot type to set for all bots.
        new_bot_version (str): New bot version to set for all bots.

    Returns:
        None

    Raises:
        FileNotFoundError: If the specified config file doesn't exist.
        json.JSONDecodeError: If the config file contains invalid JSON.
        KeyError: If the config file is missing the required 'bots' key.
        Exception: For other unexpected errors during processing.

    Example:
        >>> update_bot_config("config.json", "gpt", "4.0")
    """
    try:
        # Read the config file
        with open(config_file, "r") as f:
            config = json.load(f)

        # Update all bots
        for bot in config["bots"]:
            bot["bot_type"] = new_bot_type
            bot["bot_version"] = new_bot_version

        # Write the updated config back to file
        with open(config_file, "w") as f:
            json.dump(config, f, indent=4)

        print(f"Successfully updated {config_file}")
        print(f"All bots now have type: {new_bot_type} and version: {new_bot_version}")

    except FileNotFoundError:
        print(f"Error: Config file '{config_file}' not found")
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in '{config_file}'")
    except KeyError:
        print(f"Error: Missing required 'bots' key in config file")
    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print(
            "Usage: python update_bot_config.py <config_file> <new_bot_type> <new_bot_version>"
        )
        sys.exit(1)

    config_file = sys.argv[1]
    new_bot_type = sys.argv[2]
    new_bot_version = sys.argv[3]

    update_bot_config(config_file, new_bot_type, new_bot_version)
