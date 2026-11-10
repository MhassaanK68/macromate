"""Compatibility for historical ``python -m model.*`` commands.

The maintained implementation lives in ``models/`` for Deliverable 2.
"""

from pathlib import Path

__path__ = [str(Path(__file__).resolve().parents[1] / "models")]
