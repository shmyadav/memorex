#!/bin/bash

# MemoRex Setup Script
# This script creates a virtual environment, installs dependencies,
# and assists in setting up the .env file.

set -e

echo "--------------------------------------------------"
echo "Setting up MemoRex environment..."
echo "--------------------------------------------------"

# Check for Python 3
if ! command -v python3 &> /dev/null
then
    echo "Error: Python 3 is not installed. Please install Python 3."
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
else
    echo "Virtual environment already exists."
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "Installing dependencies from requirements.txt..."
pip install --upgrade pip
pip install -r requirements.txt

# Initializing .env file
if [ ! -f ".env" ]; then
    if [ -f ".env.example" ]; then
        echo "Creating .env from .env.example..."
        cp .env.example .env
        echo "Successfully created .env. Please edit it with your OpenAI and Neo4j credentials."
    else
        echo "Error: .env.example file not found. Please create a .env file manualy."
    fi
else
    echo ".env file already exists."
fi

echo "--------------------------------------------------"
echo "Setup complete!"
echo "To start working, run:"
echo "source venv/bin/activate"
echo "python3 main.py"
echo "--------------------------------------------------"
