"""Counting semantics for the Int Mask / Audio / Text Count nodes.

The presence rule must stay identical to Any Empty Bool (utils/_logic_ops.py):
None, empty/all-zero tensors, blank strings, 0/False scalars and empty
containers are absent. These tests pin that parity plus the dynamic-port and
list-input behaviour.
"""

import asyncio
import sys
import types

import torch

from _node_loader import load_node


def _install_comfy_stub():
    class _PortType:
        def __init__(self, name):
            self.name = name

        def Input(self, id, *args, **kwargs):
            return {"kind": "input", "type": self.name, "id": id, **kwargs}

        def Output(self, *args, **kwargs):
            return {"kind": "output", "type": self.name, **kwargs}

    io = types.SimpleNamespace(
        ComfyNode=object,
        Schema=lambda **kwargs: kwargs,
        NodeOutput=lambda *values, **kwargs: values,
    )
    for name, io_type in {
        "Image": "IMAGE",
        "Mask": "MASK",
        "Video": "VIDEO",
        "Audio": "AUDIO",
        "Int": "INT",
        "Float": "FLOAT",
        "Boolean": "BOOLEAN",
        "String": "STRING",
        "Combo": "COMBO",
    }.items():
        setattr(io, name, _PortType(io_type))
    io.Custom = lambda name: _PortType(name)

    latest = types.ModuleType("comfy_api.latest")
    latest.io = io
    latest.ui = types.SimpleNamespace()
    latest.ComfyExtension = object
    package = types.ModuleType("comfy_api")
    package.latest = latest
    sys.modules["comfy_api"] = package
    sys.modules["comfy_api.latest"] = latest


_install_comfy_stub()

MASK = load_node("int_count_under_test", "nodes/int/int_mask_count.py").IntMaskCount
AUDIO = load_node(
    "int_count_under_test", "nodes/int/int_audio_count.py"
).IntAudioCount
TEXT = load_node("int_count_under_test", "nodes/int/int_text_count.py").IntTextCount
IMAGE = load_node(
    "int_count_under_test", "nodes/int/int_image_count.py"
).IntImageCount
VIDEO = load_node(
    "int_count_under_test", "nodes/int/int_video_count.py"
).IntVideoCount


def _run(node, **kwargs):
    return asyncio.run(node.execute(**kwargs))


# --------------------------------------------------------------------- schema


def test_schema_declares_one_dynamic_wildcard_family_per_node():
    mask = MASK.define_schema()
    assert mask["node_id"] == "1hew_IntMaskCount"
    assert mask["display_name"] == "Int Mask Count"
    assert mask["is_input_list"] is True
    assert mask["accept_all_inputs"] is True
    assert mask["inputs"][0]["id"] == "mask_1"
    assert mask["inputs"][0]["type"] == "MASK"

    audio = AUDIO.define_schema()
    assert audio["node_id"] == "1hew_IntAudioCount"
    assert audio["display_name"] == "Int Audio Count"
    assert audio["inputs"][0]["id"] == "audio_1"
    assert audio["inputs"][0]["type"] == "AUDIO"

    text = TEXT.define_schema()
    assert text["node_id"] == "1hew_IntTextCount"
    assert text["display_name"] == "Int Text Count"
    assert text["inputs"][0]["id"] == "text_1"
    assert text["inputs"][0]["type"] == "*"


# ----------------------------------------------------------------------- mask


def test_mask_batch_counts_valid_frames_only():
    batch = torch.ones(3, 8, 8)
    batch[1] = 0.0
    assert _run(MASK, mask_1=[batch])[0] == 2


def test_single_mask_and_empty_mask():
    assert _run(MASK, mask_1=[torch.ones(8, 8)])[0] == 1
    assert _run(MASK, mask_1=[torch.zeros(8, 8)])[0] == 0
    assert _run(MASK, mask_1=[torch.zeros(0, 8, 8)])[0] == 0


def test_mask_list_counts_each_element():
    masks = [torch.ones(1, 8, 8), torch.zeros(1, 8, 8), torch.ones(2, 8, 8)]
    assert _run(MASK, mask_1=masks)[0] == 3


def test_mask_dynamic_ports_and_unconnected_first():
    assert _run(
        MASK,
        mask_2=[torch.ones(1, 8, 8)],
        mask_3=[torch.ones(1, 8, 8)],
    )[0] == 2
    assert _run(MASK)[0] == 0


def test_mask_ignores_other_kwargs():
    out = _run(
        MASK,
        mask_1=[torch.ones(1, 8, 8)],
        image_2=[torch.ones(1, 8, 8, 3)],
    )
    assert out[0] == 1


# ---------------------------------------------------------------------- audio


def _audio():
    return {"waveform": torch.zeros(1, 2, 4410), "sample_rate": 44100}


def test_audio_emptiness_matches_any_empty_bool():
    # Any Empty Bool only checks dict length, so a silent audio dict is present
    # while None / {} are absent.
    assert _run(AUDIO, audio_1=[_audio()])[0] == 1
    assert _run(AUDIO, audio_1=[{}])[0] == 0
    assert _run(AUDIO, audio_1=[None])[0] == 0


def test_audio_list_and_dynamic_ports():
    assert _run(AUDIO, audio_1=[_audio(), _audio()])[0] == 2
    assert _run(AUDIO, audio_2=[_audio()])[0] == 1
    assert _run(AUDIO)[0] == 0


# ----------------------------------------------------------------------- text


def test_text_empty_rules_match_any_empty_bool():
    cases = [
        ("", 0),
        ("   ", 0),
        ("hello", 1),
        (0, 0),
        (0.0, 0),
        (False, 0),
        (1, 1),
        (0.5, 1),
        (True, 1),
    ]
    for value, expected in cases:
        assert _run(TEXT, text_1=[value])[0] == expected, value


def test_text_list_counts_non_empty_items():
    assert _run(TEXT, text_1=["a", "", " ", "b"])[0] == 2


def test_text_ignores_non_text_values():
    out = _run(
        TEXT,
        text_1=[torch.ones(2, 2), {"a": 1}, None],
        text_2=["x"],
    )
    assert out[0] == 1


def test_text_dynamic_ports_and_unconnected_first():
    assert _run(TEXT, text_2=[1], text_3=["x"])[0] == 2
    assert _run(TEXT)[0] == 0


# ------------------------------------------- shared helper (existing counters)


def test_image_count_keeps_frame_and_dynamic_port_semantics():
    batch = torch.ones(3, 8, 8, 3)
    batch[1] = 0.0
    # No INPUT_IS_LIST on this node, so runtime passes the raw tensor.
    assert _run(IMAGE, image_1=batch)[0] == 2
    assert _run(IMAGE, image_1=None, image_2=torch.ones(2, 8, 8, 3))[0] == 2
    assert _run(IMAGE)[0] == 0


def test_video_count_counts_present_objects():
    assert _run(VIDEO, video_1=object())[0] == 1
    assert _run(VIDEO, video_1=None)[0] == 0
    assert _run(VIDEO, video_2=object(), video_3=object())[0] == 2
