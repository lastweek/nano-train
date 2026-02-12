#!/usr/bin/env python3
"""
Simple HTTP server to view TensorBoard logs without installing tensorflow.

Usage:
    python scripts/view_logs.py

Then open http://localhost:8000 in your browser.
"""

import http.server
import os
import sys
import webbrowser
from pathlib import Path

def main():
    log_dir = Path("logs")
    if not log_dir.exists():
        print(f"Error: {log_dir} directory not found")
        print("Run training first to generate logs")
        sys.exit(1)

    port = 8000
    print(f"\n🔍 Starting log viewer server...")
    print(f"Log directory: {log_dir.absolute()}")
    print(f"URL: http://localhost:{port}")
    print("\nPress Ctrl+C to stop\n")

    # Try to open browser automatically
    try:
        webbrowser.open(f"http://localhost:{port}")
    except:
        pass

    # Change to logs directory and serve
    os.chdir(log_dir)
    http.server.HTTPServer(('', port).serve_forever()

if __name__ == "__main__":
    main()
