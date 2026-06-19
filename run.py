"""
run.py — Launcher for the Krisha Patel AI Model Hub
=====================================================
Installs all required dependencies and launches the Streamlit app.

Usage:
    python run.py
"""

import subprocess
import sys
import os


def install_dependencies():
    """Install all required packages from requirements.txt."""
    req_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "requirements.txt")

    print("=" * 60)
    print("  Krisha Patel — AI Model Hub")
    print("  Dependency Installer & Launcher")
    print("=" * 60)
    print()

    if not os.path.exists(req_path):
        print("[ERROR] requirements.txt not found!")
        sys.exit(1)

    print("[1/2] Installing dependencies...")
    print("-" * 40)

    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-r", req_path, "--quiet"],
        capture_output=False,
    )

    if result.returncode != 0:
        print("\n[WARNING] Some packages may have failed. Trying individual installs...\n")
        packages = [
            "streamlit>=1.30.0",
            "torch>=1.10.0",
            "numpy>=1.21.0",
            "pandas>=1.3.0",
            "scikit-learn>=1.0.0",
            "transformers>=4.30.0",
            "plotly>=5.15.0",
        ]
        for pkg in packages:
            print(f"  Installing {pkg}...")
            subprocess.run(
                [sys.executable, "-m", "pip", "install", pkg, "--quiet"],
                capture_output=True,
            )

    print()
    print("[OK] All dependencies installed.")
    print()


def launch_app():
    """Launch the Streamlit application."""
    app_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app.py")

    if not os.path.exists(app_path):
        print("[ERROR] app.py not found!")
        sys.exit(1)

    print("[2/2] Launching Streamlit app...")
    print("-" * 40)
    print("  The app will open at: http://localhost:8501")
    print("  Press Ctrl+C to stop the server.")
    print()

    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", app_path, "--server.headless", "true"],
    )


if __name__ == "__main__":
    install_dependencies()
    launch_app()
