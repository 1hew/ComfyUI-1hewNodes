"""Test-suite bootstrap for running node tests outside ComfyUI.

The repository root is a Python package whose ``__init__`` imports ``comfy_api``.
When the suite runs in a bare interpreter (no ComfyUI on ``sys.path``) that import
fails before any test can be collected. Install a minimal stub only when the real
module is unavailable, so tests still work both inside and outside ComfyUI.
"""

import sys
import types

try:
    import comfy_api  # noqa: F401
except ImportError:
    io = types.SimpleNamespace(
        ComfyNode=object,
        Schema=lambda **kwargs: kwargs,
        NodeOutput=lambda *values, **kwargs: values,
    )
    latest = types.ModuleType("comfy_api.latest")
    latest.io = io
    latest.ui = types.SimpleNamespace()
    latest.ComfyExtension = object
    package = types.ModuleType("comfy_api")
    package.latest = latest
    sys.modules["comfy_api"] = package
    sys.modules["comfy_api.latest"] = latest

try:
    import aiohttp  # noqa: F401
except ImportError:
    # Load Text registers an aiohttp route at import time. Provide only the
    # surface that node module touches so it can be imported in a bare
    # interpreter; no route is ever called by the tests.
    aiohttp = types.ModuleType("aiohttp")
    web = types.ModuleType("aiohttp.web")

    def _json_response(data=None, status=200, **kwargs):
        return {"data": data, "status": status, **kwargs}

    web.json_response = _json_response
    aiohttp.web = web
    sys.modules["aiohttp"] = aiohttp
    sys.modules["aiohttp.web"] = web

try:
    import server  # noqa: F401
except ImportError:
    server = types.ModuleType("server")

    class _Routes:
        def get(self, _path):
            return lambda handler: handler

    class _PromptServer:
        instance = types.SimpleNamespace(routes=_Routes())

    server.PromptServer = _PromptServer
    sys.modules["server"] = server
