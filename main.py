import os
import sys
import subprocess

if __name__ == "__main__":
    # Hugging Face Spaces routes port 7860 to the public web
    port = os.environ.get("PORT", "7860")
    
    cmd = [
        sys.executable,
        "-m",
        "marimo",
        "run",
        "app.py",
        "--host",
        "0.0.0.0",
        "--port",
        str(port),
        "--no-token",
        "--headless",
    ]
    print(f"Starting Marimo server on 0.0.0.0:{port}...")
    sys.exit(subprocess.call(cmd))
