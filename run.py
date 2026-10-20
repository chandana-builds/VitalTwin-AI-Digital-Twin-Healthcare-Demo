"""
VitalTwin AI - Direct Dashboard Launcher
Applies Windows path normalizations and launches the Streamlit app.
Usage:
    python run.py
"""

import sys
import os

# Add project root to sys.path
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import sitecustomize

from streamlit.web import cli as stcli

if __name__ == "__main__":
    app_path = os.path.join(PROJECT_ROOT, "app.py")
    port = os.environ.get("PORT", "8501")
    sys.argv = [
        "streamlit",
        "run",
        app_path,
        "--server.port",
        str(port),
        "--server.address",
        "0.0.0.0",
        "--server.headless",
        "true",
    ]
    sys.exit(stcli.main())
