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
├── src/
│   ├── conversation/
│   │   ├── __init__.py
│   │   ├── loader.py
│   │   └── manager.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── claude.py
│   │   ├── gemini.py
│   │   ├── gpt.py
│   │   └── ollama.py
│   └── utils/
│       ├── __init__.py
│       └── env.py
├── config.json
├── main.py
└── README.md
```

The project is organized into the following components:

- `src/conversation/`: Contains the core conversation management logic
- `src/models/`: Implements different AI model integrations
- `src/utils/`: Utility functions and environment configuration
- `config.json`: Configuration file for conversation settings
- `main.py`: Application entry point

## Installation

```bash
git clone https://github.com/yourusername/chatbot_conversation.git
cd chatbot_conversation
pip install -r requirements.txt
```

## Environment Setup

Create a `.env` file in the root directory with your API keys:

```
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key
GOOGLE_API_KEY=your_google_key
```

## Configuration

Edit `config.json` to customize the conversation. Example configuration for a tennis discussion:

```json
{
    "conversation_seed": "What makes a tennis player the GOAT?",
    "rounds": 3,
    "bots": [
        {
            "name": "TennisBot1",
            "bot_type": "GPT",
            "model_version": "gpt-4",
            "system_prompt": "You are an expert tennis analyst..."
        }
    ]
}
```

Configuration parameters:
- `conversation_seed`: The initial prompt to start the discussion
- `rounds`: Number of conversation rounds
- `bots`: Array of bot configurations
  - `name`: Display name for the bot
  - `bot_type`: Model type (GPT, CLAUDE, GEMINI, OLLAMA)
  - `model_version`: Specific model version to use
  - `system_prompt`: Instructions that define the bot's personality and knowledge

## Usage

1. Configure your bots and conversation in `config.json`
2. Set up environment variables in `.env`
3. Run the conversation:

```bash
python main.py
```

The bots will engage in a multi-round discussion based on the conversation seed, with each bot maintaining its configured personality and expertise.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This code is not licensed
