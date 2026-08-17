"""
main.py

Project Root Entrypoint for UAV Localization Evaluation.
Delegates to scripts/main.py.
"""

import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_MAIN = os.path.join(SCRIPT_DIR, "scripts", "main.py")

if __name__ == "__main__":
    import subprocess
    cmd = [sys.executable, SCRIPTS_MAIN] + sys.argv[1:]
    sys.exit(subprocess.call(cmd))
