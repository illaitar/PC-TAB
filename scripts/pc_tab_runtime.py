"""Import helper for the hyphenated `pc-tab` implementation directory."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType


REPO_ROOT = Path(__file__).resolve().parents[1]
IMPL_DIR = REPO_ROOT / "pc-tab"


def load_pc_tab_impl() -> ModuleType:
    existing = sys.modules.get("pc_tab_impl")
    if existing is not None:
        return existing

    init_py = IMPL_DIR / "__init__.py"
    if not init_py.exists():
        raise FileNotFoundError(init_py)

    spec = importlib.util.spec_from_file_location(
        "pc_tab_impl",
        init_py,
        submodule_search_locations=[str(IMPL_DIR)],
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import PC-TAB implementation from {init_py}")

    module = importlib.util.module_from_spec(spec)
    sys.modules["pc_tab_impl"] = module
    spec.loader.exec_module(module)
    return module
