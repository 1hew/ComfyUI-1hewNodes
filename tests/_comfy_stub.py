"""Minimal comfy_api stub shared by the dict/list node tests.

The suite runs without ComfyUI, and some test modules install their own
minimal stub. Replacing it here keeps these tests independent of collection
order and provides every port type the new nodes declare.
"""

import sys
import types


class PortType:
    def __init__(self, type_name):
        self.type_name = type_name

    def Input(self, name, **kwargs):
        return {"kind": "input", "type": self.type_name, "id": name, **kwargs}

    def Output(self, **kwargs):
        return {"kind": "output", "type": self.type_name, **kwargs}


class _AutogrowTemplateNames:
    def __init__(self, input, names, min=1):
        self.input = input
        self.names = list(names)
        self.min = min

    def ports(self):
        return [
            {"kind": "input", "type": self.input["type"], "id": name}
            for name in self.names
        ]


class _Autogrow:
    """Just enough of io.Autogrow for a schema to declare growing ports."""

    TemplateNames = _AutogrowTemplateNames

    class Input:
        def __init__(self, id, template, **kwargs):
            self.id = id
            self.template = template

        def as_port(self):
            return {
                "kind": "input",
                "type": "AUTOGROW",
                "id": self.id,
                "template": self.template,
            }


def install():
    io = types.SimpleNamespace(
        ComfyNode=object,
        Schema=lambda **kwargs: kwargs,
        NodeOutput=lambda *values, **kwargs: values,
        Custom=PortType,
        String=PortType("STRING"),
        Combo=PortType("COMBO"),
        Boolean=PortType("BOOLEAN"),
        Int=PortType("INT"),
        Float=PortType("FLOAT"),
        Image=PortType("IMAGE"),
        Mask=PortType("MASK"),
        Video=PortType("VIDEO"),
        Audio=PortType("AUDIO"),
        Autogrow=_Autogrow,
    )
    latest = types.ModuleType("comfy_api.latest")
    latest.io = io
    latest.ui = types.SimpleNamespace()
    latest.ComfyExtension = object
    package = types.ModuleType("comfy_api")
    package.latest = latest
    sys.modules["comfy_api"] = package
    sys.modules["comfy_api.latest"] = latest
