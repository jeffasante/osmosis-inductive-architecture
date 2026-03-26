#!/bin/bash

# Load .env if it exists
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

DEFAULT_ARC_API_KEY="sk-gclcqdsvtehcbgjti3xf56ejklkizvdq"
export ARC_API_KEY="${ARC_API_KEY:-$DEFAULT_ARC_API_KEY}"

echo "Using ARC_API_KEY: $ARC_API_KEY"

# Ensure virtual environment exists
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment and installing arc-agi..."
    python3 -m venv .venv
    source .venv/bin/activate
    pip install arc-agi
else
    source .venv/bin/activate
fi

# Run the data fetching script
python3 get_arc_data.py

# Display summary
if [ -f "arc_games_summary.md" ]; then
    cat arc_games_summary.md
fi
