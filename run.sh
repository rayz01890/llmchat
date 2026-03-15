#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"

# Install dependencies if needed
if ! python3 -m pip show streamlit &>/dev/null; then
    echo "Installing dependencies..."
    python3 -m pip install -r requirements.txt
fi

python3 -m streamlit run streamlit_app.py
