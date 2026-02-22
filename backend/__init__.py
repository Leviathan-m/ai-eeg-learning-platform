"""Backend package initialization."""

from __future__ import annotations

import os
import sys

# Keep backward-compatible absolute imports used across the backend package.
_BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)
