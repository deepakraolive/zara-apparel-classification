#!/bin/bash
# -----------------------------------------------------------------------------
# Zara Retail Intelligence & ML Platform - Streamlit App Launcher
# INT234 (Predictive Analytics) - Deepak Rao (12401110)
# -----------------------------------------------------------------------------
cd "$(dirname "$0")"

echo "================================================================="
echo "   Launching Zara Predictive Analytics Platform (Streamlit)      "
echo "================================================================="
echo "Activating virtual environment (.venv)..."

if [ -f "./.venv/bin/streamlit" ]; then
    echo "Starting server on http://localhost:8501 ..."
    ./.venv/bin/streamlit run app.py
else
    echo "Error: Virtual environment not found at ./.venv"
    echo "Please ensure you are in the project directory."
fi
