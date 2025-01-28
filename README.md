# Chatbot Conversation

An extensible Python application that facilitates conversations between multiple AI chatbots using different language models.

## Features

- Configurable conversation settings via JSON
- Real-time streaming of bot responses with live Markdown rendering
- Existing support for multiple AI models:
  - OpenAI GPT
  - Anthropic Claude
  - Google Gemini
  - Ollama (local models)
- Extensible drop-in architecture for adding new models without modifying core project files.
  - See `dummy_bot.py` for an example of how to add a new model
- Type-safe implementation
- Comprehensive logging
- Environment-based configuration

## Project Structure

```text
chatbot_conversation/
├── tests/
│   ├── config/
│   │   ├── test_config.json
│   │   └── test_config_empty.json
│   ├── output/
│   ├── test_src/
│   │   └── test_chatbot_conversation/
│   │       ├── test_conversation/
│   │       │   ├── test_display/
│   │       │   │   ├── conftest.py
│   │       │   │   └── test_display.py
│   │       │   ├── conftest.py
│   │       │   ├── test_bots_initializer.py
│   │       │   ├── test_loader.py
│   │       │   ├── test_manager.py
│   │       │   ├── test_prompt.py
│   │       │   └── test_transcript.py
│   │       ├── test_models/
│   │       │   ├── testbots/
│   │       │   │   ├── conftest.py
│   │       │   │   ├── test_bots_shared.py
│   │       │   │   ├── test_claude_bot.py
│   │       │   │   ├── test_dummy_bot.py
│   │       │   │   ├── test_gemini_bot.py
│   │       │   │   ├── test_gpt_bot.py
│   │       │   │   └── test_ollama_bot.py
│   │       │   ├── conftest.py
│   │       │   ├── test_base.py
│   │       │   ├── test_bot_registry.py
│   │       │   └── test_factory.py
│   │       ├── test_utils/
│   │       │   ├── conftest.py
│   │       │   ├── test_env.py
│   │       │   └── test_logging_util.py
│   │       ├── conftest.py
│   │       ├── test_main.py
│   │       └── test_version.py
│   └── conftest.py
├── src/
│   └── chatbot_conversation/
│       ├── conversation/
│       │   ├── display/
│       │   │   ├── __init__.py
│       │   │   ├── abstract_display.py
│       │   │   └── rich_display.py
│       │   ├── __init__.py
│       │   ├── bots_initializer.py
│       │   ├── loader.py
│       │   ├── manager.py
│       │   ├── prompt.py
│       │   └── transcript.py
│       ├── models/
│       │   ├── bots/
│       │   │   ├── __init__.py
│       │   │   ├── claude_bot.py
│       │   │   ├── dummy_bot.py
│       │   │   ├── gemini_bot.py
│       │   │   ├── gpt_bot.py
│       │   │   └── ollama_bot.py
│       │   ├── __init__.py
│       │   ├── base.py
│       │   ├── bot_registry.py
│       │   └── factory.py
│       ├── utils/
│       │   ├── __init__.py
│       │   ├── env.py
│       │   └── logging_util.py
│       ├── __init__.py
│       ├── main.py
│       └── version.py
├── config/
│   ├── examples/
│   │   ├── churchill.config.json
│   │   ├── dummy.config.json
│   │   └── tennis.config.json
│   ├── .env.example
│   ├── config.json
│   └── logging.conf
├── output/
│   ├── examples/
│   │   ├── churchill.transcript_250112_110658.md
│   │   ├── dummy.transcript_250112_111856.md
│   │   └── tennis.transcript_250112_111705.md
│   └── transcript_<yymmdd>_<hhmmss>.md
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
- `src/chatbot_conversation/version.py`: Package software version number
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
- `output/`: Directory containing conversation transcripts with date and time in their name in the pattern *transcript_\<yymmdd>_\<hhmmss>.md*

## Installation

### Prerequisites

If you plan to use Ollama models, you need to:

1. Install Ollama from <https://ollama.com/>
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

```text
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

```text
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key
GOOGLE_API_KEY=your_google_key
```

## Configuration

Edit `/config/config.json` to customize the conversation. Example configuration for a tennis discussion:

```json
{
    "author": "Brian Sentance",
    "conversation_seed": "I think Roger Federer is the GOAT!",
    "rounds": 2,
    "shared_prefix": "You are about to take part in a conversation...",
    "first_round_postfix": "This is the first round of the conversation...",
    "last_round_postfix": "This is now the last round of the conversation...",
    "bots": [
        {
            "bot_name": "RogerFan",
            "bot_type": "GPT",
            "bot_version": "gpt-4-mini",
            "bot_prompt": "You are a fan of tennis and Roger Federer...",
            "bot_params_opt": {
                "temperature": 0.7,
                "max_tokens": 500
            }
        },
        {
            "bot_name": "RafaFan",
            "bot_type": "CLAUDE",
            "bot_version": "claude-3-haiku-20240307",
            "bot_prompt": "You are a fan of tennis and Rafael Nadal...",
            "bot_params_opt": {
                "temperature": 0.5
            }
        },
        {
            "bot_name": "NovakFan",
            "bot_type": "GEMINI",
            "bot_version": "gemini-1.5-flash",
            "bot_prompt": "You are a fan of tennis and Novak Djokovic..."
        }
    ]
}
```

Configuration parameters:

- `author`: The author of the conversation configuration. Used in transcript.md for record keeping.
- `conversation_seed`: The initial prompt to start the discussion.
- `rounds`: Number of conversation rounds.
- `shared_prefix`: Common instructions provided to all bots about conversation structure.
- `first_round_postfix`: Instructions for first round only (e.g., bot introductions).
- `last_round_postfix`: Instructions for last round only (e.g., concluding thoughts).
  - Prefix and postfix all support template variable names:
    - Supports template variable `{bot_name}` which gets replaced with each bot's name
    - Example: "You are {bot_name}. " becomes "You are RogerFan. " for the RogerFan bot
    - Supports template variable `{max_tokens}` which gets replaced with the setting of max_tokens passed to the model
    - Example: "Keep responses under {max_tokens} tokens. " becomes "Keep responses under 500 tokens. " for a max_tokens parameter set to 500
- `bots`: Array of bot configurations:
  - `bot_name`: Display name for the bot (see template variable `{bot_name}` above)
  - `bot_type`: Model type (GPT, CLAUDE, GEMINI, OLLAMA)
  - `bot_version`: Specific model version to use
  - `bot_prompt`: Role-specific instructions for each bot
  - `bot_params_opt`: Optional model parameters (may vary by model type):
    - `temperature`: Controls randomness in responses
      - Range of values typically 0.0 to 2.0
      - Some models override this range (e.g. Ollama 0.0 to 1.0)
      - Lower values: More focused, deterministic responses
      - Higher values: More creative, varied responses
      - Defaults to model-specific values if not specified
    - `max_tokens`: Maximum length of generated responses
      - Defaults to 300 if not specified
      - Higher values allow longer responses but may use more API tokens
      - Supported in prompting with the template variable `{max_tokens}` (see above)

Important validation rules:

- Bot names must be unique within a configuration
- Bot names are alphanumeric and cannot contain special/punctuation charactors
- Bot names can however use underscore within their name e.g. "Fredbot_42"
- But names cannot start or end with an underscore

### Template Variables

The following strings support the `{bot_name}` and `{max_tokens}` template variables:

- `shared_prefix`
- `first_round_postfix`
- `last_round_postfix`
- `bot_prompt`

These template variables allow for personalized system prompts and instructions for each bot at different stages of the conversation.

## Usage

1. Set up environment variables in `/config/.env`
2. Configure/check logging in `/config/logging.conf`
3. Configure your bots and conversation by either:
   - Editing the default `/config/config.json`, or
   - Creating a custom configuration file (see examples in `/config/examples/`)
4. Run the conversation using either:

   ```bash
   # Using default config.json
   python /src/chatbot_conversation/main.py
   
   # Or specifying a custom config file
   python /src/chatbot_conversation/main.py /config/examples/tennis.config.json
   ```

The bots will engage in a multi-round discussion based on the conversation seed, with each bot maintaining its configured personality and expertise. The conversation will be saved to the `./output` directory with a filename that includes the date and time in the pattern *transcript_\<yymmdd>_\<hhmmss>.md* for example *"transcript_250112_095235.md"*

The transcript file will also include the configuration data used to generate the conversation and the name of the configuration file that was used.

### Example `transcript_<yymmdd>_<hhmmss>.md`

Here is an excerpt from the start of a sample `transcript_250111_172128.md`:

```markdown
# Is Novak Djokovic the GOAT of tennis?

## Round 1 of 3

**RogerFan**: Hello everyone, I'm RogerFan. The debate over who the greatest of all time (GOAT) in tennis is certainly heated! While many argue in favor of Novak Djokovic due to his incredible achievements and consistency, I firmly believe that **Roger Federer** holds that title.

// ...more content...

---

**RafaFan**: Hi everyone, I'm RafaFan! I appreciate RogerFan's points about Federer, but I have to respectfully disagree. For me, **Rafael Nadal** is the true GOAT of tennis, and here's why:

// ...more content and another round of conversation...

---

## Round 3 of 3

**RogerFan**: Thank you for understanding, RafaFan! It's always refreshing to discuss these players, especially their impact on tennis. Each of the 'Big Three' has indeed shaped the game in unique ways, and I'd like to elaborate on their contributions:

// ...more content

---

**RafaFan**: I really appreciate your insights, RogerFan! You’ve captured the essence of each player's unique contributions to tennis beautifully. 

// ...more content

---

## Conversation Finished - 3 Rounds With 2 Bots Completed!

## *Conversation Generated* : 2025-01-11 17:21:28

## *Software Version* : 1.0.0

## *Configuration Author* : Brian Sentance

## *Configuration File* : config\config.json


// ...config.json data
```

Following the Configuration Data title, the config.json data is of the form outlined in the [Configuration](#configuration) section above. This keeps both the conversation transcript and the configuration data used to generate the conversation together in the transcript file located in the `./output` directory.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
