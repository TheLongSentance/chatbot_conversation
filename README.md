# Chatbot Conversation

An extensible Python application that facilitates conversations between multiple AI chatbots using different language models (existing chatbot support includes GPT, Claude, Gemini, and Ollama but more providers can easily be added).

## Features

- Existing support for multiple AI models:
  - OpenAI GPT
  - Anthropic Claude
  - Google Gemini
  - Ollama (local models)
- Configurable conversation settings via JSON
- Extensible drop-in architecture for adding new models without modifying core project files.
  - See `dummy_bot.py` for an example of how to add a new model
- Type-safe implementation
- Comprehensive logging
- Environment-based configuration

## Project Structure

```
chatbot_conversation/
├── tests/
│   ├── fixtures/
│   │   ├── test_config.json
│   │   └── test_config_empty.json
│   ├── test_conversation/
│   │   ├── __init__.py
│   │   └── test_manager.py
│   ├── test_models/
│   │   ├── __init__.py
│   │   ├── conftest.py
│   │   ├── test_base.py
│   │   ├── test_claude_bot.py
│   │   ├── test_factory.py
│   │   ├── test_gemini_bot.py
│   │   ├── test_ollama_bot.py
│   │   └── test_openai_bot.py
│   ├── __init__.py
│   └── conftest.py
├── src/
│   └── chatbot_conversation/
│       ├── conversation/
│       │   ├── __init__.py
│       │   ├── loader.py
│       │   └── manager.py
│       ├── models/
│       │   ├── bots/
│       │   │   ├── __init__.py
│       │   │   ├── claude_bot.py
│       │   │   ├── dummy_bot.py
│       │   │   ├── gemini_bot.py
│       │   │   ├── ollama_bot.py
│       │   │   └── openai_bot.py
│       │   ├── __init__.py
│       │   ├── base.py
│       │   ├── bot_registry.py
│       │   └── factory.py
│       ├── utils/
│       │   ├── __init__.py
│       │   ├── env.py
│       │   └── logging_util.py
│       ├── __init__.py
│       └── main.py
├── config/
│   ├── examples/
│   │   ├── brexit.config.json
│   │   ├── churchill.config.json
│   │   ├── dummy.config.json
│   │   └── tennis.config.json
│   ├── .env.example
│   ├── config.json
│   └── logging.conf
├── .gitignore
├── environment.yml
├── LICENSE
├── pyproject.toml
├── README.md
├── requirements.txt
└── requirements-dev.txt
```

The project is organized into the following components:

- `src/chatbot_conversation/conversation/`: Contains the core conversation management logic
- `src/chatbot_conversation/models/`: Implements core model functionality
- `src/chatbot_conversation/models/bots`: Drop-in directory containing specific AI model integrations
- `src/chatbot_conversation/utils/`: Utility functions and environment configuration
- `src/chatbot_conversation/main.py`: Application entry point
- `config/`: Contains runtime configuration files
- `config/config.json`: Configuration file for conversation settings
- `config/examples/`: Example configuration files for use in the config directory
- `config/.env.example`: Example .env file format for storage of private keys for AI APIs
- `config/logging.conf`: Configuration file for logging 
- `tests/`: Test infrastructure based on pytest
- `.gitignore`: Git ignore file
- `environment.yml`: Conda environment configuration
- `LICENSE`: License file
- `pyproject.toml`: Development tools configuration
- `requirements.txt`: Pip requirements file
- `requirements-dev.txt`: Pip requirements file for development environment

## Installation

### Prerequisites

If you plan to use Ollama models, you need to:
1. Install Ollama from https://ollama.com/
2. Pull the models you want to use:
```bash
# Example for pulling the Llama 2 model
ollama pull llama2
```

### Using pip (Recommended)
```bash
# Clone the repository
git clone https://github.com/TheLongSentance/chatbot_conversation.git
cd chatbot_conversation

# Create and activate virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install build requirements
pip install hatch hatchling

# Install the package in editable mode with development dependencies
pip install -e ".[test]"  # Include [test] for development dependencies
```

This will:
- Install all runtime dependencies specified in `pyproject.toml`
- Install all development dependencies (pytest, black, etc.) if you include `[test]`
- Install the package in editable mode for development
- Set up the CLI command specified in `pyproject.toml` so you can just enter `chatbot_conversation` at the command line instead of `python.exe ./src/chatbot_conversation/main.py`

### Alternative: Using Conda

You have two options when using Conda:

#### Option 1: Conda + pip (Recommended for Conda users)
```bash
# Clone the repository
git clone https://github.com/TheLongSentance/chatbot_conversation.git
cd chatbot_conversation

# Create and activate conda environment 
conda env create -f environment.yml # this also installs hatch and hatchling
conda activate botconv

# Install the package in editable mode with development dependencies
pip install -e ".[test]"  # Include [test] for development dependencies
```

This approach:
- Uses Conda to manage the Python environment
- Uses pip/Hatch to handle package dependencies and development setup
- Gives you access to all development tools configured in `pyproject.toml`

#### Option 2: Pure Conda Development
```bash
# Clone the repository
git clone https://github.com/TheLongSentance/chatbot_conversation.git
cd chatbot_conversation

# Create and activate conda environment
conda env create -f environment.yml
conda activate botconv

# Register the package for development
conda develop ./src
```

Note about Option 2:
- Only registers the package for development
- Requires all dependencies to be listed in `environment.yml`
- By default installs all development dependencies
- Doesn't use the build configuration from `pyproject.toml`
- May miss out on development tools unless manually installed

We recommend Option 1 (Conda + pip) as it gives you the best of both worlds: Conda's environment management and modern Python packaging tools.

### Alternative: Manual Setup Without Build Tools

If you prefer not to use Hatch/pip's editable install, then when you try to import a Python module, Python searches for it in a list of directories. By default, this includes:
- The directory containing the input script (or current directory when no file is specified)
- The Python standard library
- The site-packages directory where pip installs packages

For a project with a `src` directory layout like this one, Python won't automatically find your package unless:
1. You install it properly using `pip install -e .` (recommended - see above), OR
2. You add the `src` directory to Python's search path

#### Checking Python's Search Path

You can check what directories Python is searching with this command:
```bash
python -c "import sys; print('\n'.join(sys.path))"
```

What to look for in the output:
- If using a virtual environment: The path should start with your virtual environment's site-packages
- If using `pip install -e .`: You should see your project's `src` directory or `.egg-link` in the virtual environment
- If setting `PYTHONPATH` manually: You should see your project's `src` directory in the list

For example, a correctly configured environment might show:
```
/Users/username/projects/chatbot_conversation/venv/lib/python3.8/site-packages
/Users/username/projects/chatbot_conversation/src
/Users/username/projects/chatbot_conversation/venv/lib/python3.8/lib-dynload
/usr/local/lib/python3.8
/usr/local/lib/python3.8/lib-dynload
/Users/username/projects/chatbot_conversation/venv/lib/python3.8/site-packages
```

You can also check just the `PYTHONPATH` environment variable:
```bash
python -c "import os; print(os.environ.get('PYTHONPATH', 'PYTHONPATH is not set'))"
```

#### Terminal Setup
On Linux/Mac:
```bash
export PYTHONPATH="${PYTHONPATH}:/path/to/your/project/src"
```

On Windows (Command Prompt):
```cmd
set PYTHONPATH=%PYTHONPATH%;C:\path\to\your\project\src
```

On Windows (PowerShell):
```powershell
$env:PYTHONPATH = "$env:PYTHONPATH;C:\path\to\your\project\src"
```

#### VS Code Setup
Create or modify `.vscode/settings.json` in your project:
```json
{
    "python.analysis.extraPaths": ["./src"],
    "terminal.integrated.env.linux": {
        "PYTHONPATH": "${workspaceFolder}/src"
    },
    "terminal.integrated.env.osx": {
        "PYTHONPATH": "${workspaceFolder}/src"
    },
    "terminal.integrated.env.windows": {
        "PYTHONPATH": "${workspaceFolder}/src"
    }
}
```

#### Troubleshooting Manual Setup

If you can't import your package, verify:
1. You're in the right virtual environment
2. Your project's `src` directory is in Python's search path (check using commands above)

## Environment Setup

Create a `.env` file in the `./config/` directory with your API keys:

```
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key
GOOGLE_API_KEY=your_google_key
```

## Configuration

Edit `/config/config.json` to customize the conversation. Example configuration for a tennis discussion:

```json
{
    "conversation_seed": "I think Roger Federer is the GOAT!",
    "rounds": 2,
    "shared_prefix": "You are about to take part in a conversation...",
    "first_round_postfix": "This is the first round of the conversation. Introduce yourself using your name {bot_name} and state your first contribution to the conversation.",
    "last_round_postfix": "This is now the last round of the conversation. This is your last response that will be added to the conversation. So think about your contributions to the conversation and the contributions of others, and put together a summary of your conclusions. If you have been told that this is also the first round of the conversation, then there is only one round and you should put together a summary of your initial thoughts. Never end your summary response with a question, but rather bring things to a natural close in the context of the main topic of conversation. ",
    "bots": [
        {
            "bot_name": "RogerFan",
            "bot_type": "GPT",
            "bot_version": "gpt-4-mini",
            "bot_prompt": "You are an expert on modern professional tennis..."
        }
    ]
}
```

Configuration parameters:
- `conversation_seed`: The initial prompt to start the discussion.
- `rounds`: Number of conversation rounds.
- `shared_prefix`: Common instructions provided to all bots about conversation structure which forms part of the system prompt for each bot.
  - Supports template variable `{bot_name}` which gets replaced with each bot's name from their configuration
  - Example: "You are {bot_name}" becomes "You are RogerFan" for the RogerFan bot
- `first_round_postfix`: Instructions for the first round of the conversation which get appended to each bot's system prompt only for the first round of the conversation. This could be used to ask each bot to introduce themselves for example. This will be removed from each bot's system prompt at the end of the first round and the system prompt re-applied.
- `last_round_postfix`: Instructions for the last round of the conversation which get appended to each bot's system prompt only for the last round of the conversation. This could be used to ask each bot to draw conclusions from the conversation for example.
- `bots`: Array of bot configurations:
  - `bot_name`: Display name for the bot (also used in shared system prompt templating)
  - `bot_type`: Model type (GPT, CLAUDE, GEMINI, OLLAMA)
  - `bot_version`: Specific model version to use
  - `bot_prompt`: Role-specific instructions for each bot

### Template Variables

The `shared_prefix`, `first_round_postfix`, `last_round_postfix` and each `bot_prompt` all support the following template variable:
- `{bot_name}`: Replaced with the bot's name from its configuration
  - Used for example to suggest in `first_round_postfix` that each bot should introduce itself by name in their first contribution to the conversation.
  - Used for making the system prompt more personalized to each bot at all stages of the conversation.

## Usage

1. Set up environment variables in `/config/.env`
2. Configure/check logging in `/config/logging.conf`
3. Configure your bots and conversation in `/config/config.json`
4. Run the conversation:

```bash
python /src/chatbot_conversation/main.py
```

The bots will engage in a multi-round discussion based on the conversation seed, with each bot maintaining its configured personality and expertise.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
