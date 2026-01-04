#!/bin/bash
# Setup script for Docker secrets
# Creates secrets directory and sets proper permissions

set -e

SECRETS_DIR="secrets"

echo "🔐 Setting up Docker secrets directory..."

# Create secrets directory if it doesn't exist
if [ ! -d "$SECRETS_DIR" ]; then
    mkdir -p "$SECRETS_DIR"
    echo "✓ Created secrets directory: $SECRETS_DIR"
else
    echo "✓ Secrets directory already exists: $SECRETS_DIR"
fi

# Set secure permissions (owner read/write/execute only)
chmod 700 "$SECRETS_DIR"
echo "✓ Set permissions on secrets directory (700)"

# Create placeholder secret files if they don't exist
if [ ! -f "$SECRETS_DIR/alpha_vantage_api_key.txt" ]; then
    echo "# Placeholder - Replace with your actual Alpha Vantage API key" > "$SECRETS_DIR/alpha_vantage_api_key.txt"
    echo "⚠ Created placeholder file: $SECRETS_DIR/alpha_vantage_api_key.txt"
    echo "   Please update with your actual API key!"
fi

if [ ! -f "$SECRETS_DIR/finnhub_api_key.txt" ]; then
    echo "# Placeholder - Replace with your actual Finnhub API key" > "$SECRETS_DIR/finnhub_api_key.txt"
    echo "⚠ Created placeholder file: $SECRETS_DIR/finnhub_api_key.txt"
    echo "   Please update with your actual API key!"
fi

# Set secure permissions on secret files (owner read/write only)
chmod 600 "$SECRETS_DIR"/*.txt 2>/dev/null || true
echo "✓ Set permissions on secret files (600)"

echo ""
echo "✅ Secrets directory setup complete!"
echo ""
echo "📝 Next steps:"
echo "   1. Edit $SECRETS_DIR/alpha_vantage_api_key.txt with your Alpha Vantage API key"
echo "   2. Edit $SECRETS_DIR/finnhub_api_key.txt with your Finnhub API key"
echo "   3. Verify files are not tracked by git: git status --ignored | grep secrets"
echo ""
echo "⚠️  IMPORTANT: Never commit secrets to git!"
echo "   The secrets/ directory is already in .gitignore"

