"""
Backward-compatibility alias for question_decomposition.py.
Redirects execution to question_decomposition.py.
"""
import sys
import subprocess

if __name__ == "__main__":
    cmd = [sys.executable, "question_decomposition.py"] + sys.argv[1:]
    sys.exit(subprocess.call(cmd))
