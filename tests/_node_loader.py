"""Load a node module without a running ComfyUI installation.

Node modules use package-relative imports (from ...utils._file_ops import),
so loading one through spec_from_file_location with a bare top-level name
fails with "attempted relative import with no known parent package". This
helper builds a throwaway package hierarchy that mirrors the repository
layout and exposes the two helpers node modules import from utils/__init__.

It deliberately does not execute the real utils/__init__.py: that module
starts the workflow monitor and imports watchdog, which is unnecessary for
node unit tests.
"""

from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]


def _first_torch_tensor(value):
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (list, tuple)):
        for item in value:
            tensor = _first_torch_tensor(item)
            if tensor is not None:
                return tensor
    return None


def _make_ui_text(text):
    return {"text": [str(text)]}


def install_package(package_name, categories):
    """Create a synthetic package with nodes/<category> and utils subpackages."""
    modules = {}

    top = types.ModuleType(package_name)
    top.__path__ = [str(ROOT)]
    modules[package_name] = top

    nodes_pkg = types.ModuleType(package_name + ".nodes")
    nodes_pkg.__path__ = [str(ROOT / "nodes")]
    modules[nodes_pkg.__name__] = nodes_pkg

    for category in categories:
        package = types.ModuleType(f"{package_name}.nodes.{category}")
        package.__path__ = [str(ROOT / "nodes" / category)]
        modules[package.__name__] = package

    utils_pkg = types.ModuleType(package_name + ".utils")
    utils_pkg.__path__ = [str(ROOT / "utils")]
    utils_pkg.first_torch_tensor = _first_torch_tensor
    utils_pkg.make_ui_text = _make_ui_text
    modules[utils_pkg.__name__] = utils_pkg

    sys.modules.update(modules)
    return modules


def load_node(package_name, relative_path, categories=None):
    """Import relative_path (e.g. nodes/io/save_audio.py) as a module."""
    rel = Path(relative_path)
    if categories is None:
        categories = [rel.parts[1]]
    install_package(package_name, categories)
    module_name = package_name + "." + ".".join(rel.with_suffix("").parts)
    # Reload on every call: tests capture module-level state (e.g. saved
    # results) in closures that differ per test instance.
    sys.modules.pop(module_name, None)
    return importlib.import_module(module_name)
