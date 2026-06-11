import sys
import os

# Add the project root to sys.path so `app` is importable from the function.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.main import app  # noqa: F401  — Vercel detects the ASGI app object
