#!/usr/bin/env bash
set -euo pipefail

# This assumes secrets were added in the Codex UI as env vars
: "${OPENAI_API_KEY:?OPENAI_API_KEY not set}"
: "${ANTHROPIC_API_KEY:?ANTHROPIC_API_KEY not set}"
: "${GOOGLE_API_KEY:?GOOGLE_API_KEY not set}"

# Export is unnecessary if already in env, but harmless
export OPENAI_API_KEY
export ANTHROPIC_API_KEY
export GOOGLE_API_KEY

# Optionally create a .env file for tools like dotenv
cat > .env <<EOF
OPENAI_API_KEY=$OPENAI_API_KEY
ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY
GOOGLE_API_KEY=$GOOGLE_API_KEY
EOF

echo ".env created with API keys"