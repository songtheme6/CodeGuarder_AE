#!/bin/bash

# Define environment and Python version
ENV_NAME="CodeGuarder"
PYTHON_VERSION="3.10"

# --- 1. Check if Conda is installed ---
if ! command -v conda &> /dev/null
then
    echo "Conda is not found. Please install Miniconda or Anaconda first."
    exit 1
fi

# --- 2. Check if jq is installed ---
# This part checks for the 'jq' command. It's a useful tool for parsing JSON.
if ! command -v jq &> /dev/null
then
    echo "jq is not found. Please install it using your system's package manager."
    echo "For Debian/Ubuntu: sudo apt-get install jq"
    echo "For Fedora/CentOS: sudo yum install jq"
    echo "For macOS (with Homebrew): brew install jq"
    exit 1
fi

# --- 3. Check if the environment already exists ---
if conda info --envs | grep -q "$ENV_NAME"
then
    echo "Environment '$ENV_NAME' already exists. Activating it..."
    conda activate "$ENV_NAME"
else
    echo "Creating environment '$ENV_NAME' with Python $PYTHON_VERSION..."
    conda create --name "$ENV_NAME" python="$PYTHON_VERSION" -y
    if [ $? -ne 0 ]; then
        echo "Failed to create the Conda environment. Exiting."
        exit 1
    fi
    conda activate "$ENV_NAME"
fi

# --- 4. Install dependencies from requirements.txt ---
echo "Installing dependencies from requirements.txt..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
else
    echo "Error: 'requirements.txt' file not found."
    exit 1
fi

echo "Environment setup complete!"
echo "To activate it in the future, just run: conda activate $ENV_NAME"

