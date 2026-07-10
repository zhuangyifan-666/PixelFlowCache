from __future__ import annotations

import sys
import inspect
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Tag subprocess-based command-line tests without broad file-level markers."""

    for item in items:
        try:
            source = inspect.getsource(item.obj)
        except (OSError, TypeError):
            continue
        if "subprocess.run" in source:
            item.add_marker(pytest.mark.cli)

