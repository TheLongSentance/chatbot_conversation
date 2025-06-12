#!/usr/bin/env bash
set -euo pipefail

# Example: read values defined as Codex secrets
# (Replace the names if your secrets use different identifiers)

OPENAI_API_KEY="$(cat "$OPENAI_API_KEY_FILE")"
ANTHROPIC_API_KEY="$(cat "$ANTHROPIC_API_KEY_FILE")"
GOOGLE_API_KEY="$(cat "$GOOGLE_API_KEY_FILE")"

export OPENAI_API_KEY
export ANTHROPIC_API_KEY
export GOOGLE_API_KEY

# Optionally create/update a .env file for APIConfig.setup_env()
cat > .env <<EOF
OPENAI_API_KEY=$OPENAI_API_KEY
ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY
GOOGLE_API_KEY=$GOOGLE_API_KEY
EOF
