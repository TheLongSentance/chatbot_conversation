# Chatbot Conversation

A Python application that facilitates conversations between multiple AI chatbots using different language models (GPT, Claude, Gemini, and Ollama).

## Features

- Support for multiple AI models:
  - OpenAI GPT
  - Anthropic Claude
  - Google Gemini
  - Ollama (local models)
- Configurable conversation settings via JSON
- Extensible architecture for adding new models
- Type-safe implementation
- Comprehensive logging
- Environment-based configuration

## Project Structure

```
chatbot_conversation/
├── tests/
│   ├── __init__.py
│   ├── test_conversation/
│   │   └── test_manager.py
│   └── test_models/
│       └── test_base.py
├── src/
│   ├── chatbot_conversation/
│   │   ├── conversation/
|   |   |   ├── __init__.py
|   |   |   ├── loader.py
|   |   |   └── manager.py
│   │   ├── models/
|   |   |   ├── __init__.py
|   |   |   ├── base.py
|   |   |   ├── claude_bot.py
|   |   |   ├── gemini_bot.py
|   |   |   ├── ollama.py
|   |   |   ├── openai.py
|   |   |   └── factory.py
│   │   ├── utils/
|   |   |   ├── __init__.py
|   |   |   ├── env.py
|   |   |   └── logging_util.py
│   │   ├── __init__.py
│   │   └── main.py
│   └── __init__.py
├── config/
│   ├── examples/
│   │   ├── brexit.config.json
│   │   ├── churchill.config.json
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
- `src/chatbot_conversation/models/`: Implements different AI model integrations
- `src/chatbot_conversation/utils/`: Utility functions and environment configuration
- `src/chatbot_conversation/main.py`: Application entry point
- `config/`: Contains runtime configuration files
- `config/config.json`: Configuration file for conversation settings
- `config/examples/`: Example configuration files for use in config directory
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

You can set up the project using either Conda (recommended) or pip:

### Using Conda
```bash
# Clone the repository
git clone https://github.com/TheLongSentance/chatbot_conversation.git
cd chatbot_conversation

# Create and activate environment using environment.yml
conda env create -f environment.yml
conda activate chatbots
```

### Alternative: Using pip
```bash
# Clone the repository
git clone https://github.com/TheLongSentance/chatbot_conversation.git
cd chatbot_conversation

# Create and activate virtual environment (optional)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

## Environment Setup

Create a `.env` file in the ./config/ directory with your API keys:

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
    "shared_system_prompt_prefix": "You are about to take part in a conversation...",
    "bots": [
        {
            "bot_name": "RogerFan",
            "bot_type": "GPT",
            "bot_model_version": "gpt-4-mini",
            "bot_specific_system_prompt": "You are an expert on modern professional tennis..."
        }
    ]
}
```

Configuration parameters:
- `conversation_seed`: The initial prompt to start the discussion
- `rounds`: Number of conversation rounds
- `shared_system_prompt_prefix`: Common instructions provided to all bots about conversation structure
  - Supports template variable `{bot_name}` which gets replaced with each bot's name from their configuration
  - Example: "You are {bot_name}" becomes "You are RogerFan" for the RogerFan bot
- `bots`: Array of bot configurations
  - `bot_name`: Display name for the bot (also used in shared system prompt templating)
  - `bot_type`: Model type (GPT, CLAUDE, GEMINI, OLLAMA)
  - `bot_model_version`: Specific model version to use
  - `bot_specific_system_prompt`: Role-specific instructions for each bot

### Template Variables

The `shared_system_prompt_prefix` supports the following template variables:
- `{bot_name}`: Replaced with the bot's name from its configuration
  - This allows the shared prompt to reference each bot's specific identity
  - Used for making the system prompt more personalized to each bot

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
