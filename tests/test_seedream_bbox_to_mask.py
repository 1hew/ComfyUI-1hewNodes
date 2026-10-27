"""Seedream bbox 转遮罩的行为锁定。

坐标只有 normalized（千分位 0-1000）与 absolute（像素）两种单位，由 coord_mode
显式指定；格式一律 xyxy（right / bottom 是边界）。节点只处理 bbox，<point> 属于
另一类标记，会被忽略；输出只有 mask 与 count。
"""

import sys
import types

import pytest
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
        "String": "STRING",
        "Combo": "COMBO",
        "Int": "INT",
        "Float": "FLOAT",
        "Boolean": "BOOLEAN",
        "Image": "IMAGE",
        "Mask": "MASK",
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

NODE = load_node(
    "seedream_bbox_under_test", "nodes/conversion/seedream_bbox_to_mask.py"
).SeedreamBBoxToMask

WIDTH, HEIGHT = 640, 480


def _image(batch=1, height=HEIGHT, width=WIDTH):
    return torch.zeros((batch, height, width, 3), dtype=torch.float32)


def _run(text, **kwargs):
    options = {
        "image": _image(),
        "coordinates_string": text,
        "coord_mode": "normalized",
        "output_mode": "separate",
    }
    options.update(kwargs)
    return NODE.execute(**options)


def _mask_box(mask_2d):
    """从遮罩反推像素框（右下开区间），用来替代已移除的 bboxes 输出。"""
    ys, xs = (mask_2d > 0.5).nonzero(as_tuple=True)
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def _boxes_of(masks):
    return [_mask_box(masks[index]) for index in range(int(masks.shape[0]))]


def _slug_mask(text, **kwargs):
    top = _run(text, output_mode="merge", **kwargs)[0]
    return top[0].reshape(HEIGHT, WIDTH)


# --------------------------------------------------------------------- schema


def test_schema_contract():
    schema = NODE.define_schema()
    assert schema["node_id"] == "1hew_SeedreamBBoxToMask"
    assert schema["display_name"] == "Seedream BBox to Mask"
    assert schema["category"] == "1hewNodes/conversion"

    assert [port["id"] for port in schema["inputs"]] == [
        "image",
        "coordinates_string",
        "coord_mode",
        "output_mode",
    ]
    assert schema["inputs"][1]["multiline"] is True
    assert schema["inputs"][2]["default"] == "normalized"
    assert schema["inputs"][2]["options"] == ["normalized", "absolute"]
    assert schema["inputs"][3]["default"] == "separate"

    assert [port["type"] for port in schema["outputs"]] == ["MASK", "INT"]
    assert [port["display_name"] for port in schema["outputs"]] == ["mask", "count"]


# --------------------------------------------------------- normalized 0-1000


def test_official_bbox_reverse_maps_to_pixels():
    mask = _slug_mask("<bbox>120 180 640 760</bbox>")
    # 120/1000*640=76.8->77, 180/1000*480=86.4->86
    # 640/1000*640=409.6->410, 760/1000*480=364.8->365
    assert _mask_box(mask) == [77, 86, 410, 365]
    assert int(mask[:, :77].sum()) == 0
    assert int(mask[:86, :].sum()) == 0
    assert int(mask[:, 410:].sum()) == 0
    assert int(mask[365:, :].sum()) == 0
    assert int(mask.sum()) == (410 - 77) * (365 - 86)


def test_doc_prompt_line_with_image_prefix_keeps_one_box():
    mask = _slug_mask("2 图1<bbox>120 180 640 760</bbox> 区域替换成花园")
    assert int(mask.sum()) == (410 - 77) * (365 - 86)


def test_multiple_boxes_produce_mask_batch():
    text = (
        "1 图1<bbox>120 180 640 760</bbox>\n"
        "2 图1<bbox>520 460 999 999</bbox>\n"
    )
    masks, count = _run(text)
    assert count == 2
    assert tuple(masks.shape) == (2, HEIGHT, WIDTH)
    assert _boxes_of(masks) == [[77, 86, 410, 365], [333, 221, 639, 480]]
    assert float(masks[0].sum()) == 333 * 279
    assert float(masks[1].sum()) == 306 * 259


def test_merge_unions_all_boxes():
    mask = _slug_mask("<bbox>0 0 100 100</bbox>\n<bbox>500 400 999 999</bbox>")
    assert _mask_box(mask) == [0, 0, 639, 480]
    assert float(mask.sum()) == 64 * 48 + (639 - 320) * (480 - 192)


# ---------------------------------------------------------------- point 忽略


def test_point_marker_is_ignored():
    empty, count = _run("<point>520 460</point>")
    assert count == 0
    assert tuple(empty.shape) == (0, HEIGHT, WIDTH)


def test_point_alongside_bbox_only_yields_the_bbox():
    masks, count = _run("<point>520 460</point>\n<bbox>120 180 640 760</bbox>")
    assert count == 1
    assert tuple(masks.shape) == (1, HEIGHT, WIDTH)
    assert _boxes_of(masks) == [[77, 86, 410, 365]]


def test_two_number_line_is_not_a_box():
    masks, count = _run("<point>1 2</point>\n10 20 110 70", coord_mode="absolute")
    assert count == 1
    assert _boxes_of(masks) == [[10, 20, 110, 70]]


# ------------------------------------------------------------------- 注释


def test_comment_lines_are_ignored():
    text = (
        "# xyxy, normalized 0-1000\n"
        "// 例: <bbox>120 180 640 760</bbox>\n"
        "-- 下面才是真正的框\n"
        "120 180 640 760\n"
    )
    masks, count = _run(text)
    assert count == 1
    assert _boxes_of(masks) == [[77, 86, 410, 365]]


def test_comment_only_input_yields_nothing():
    masks, count = _run("# 只有注释\n// xyxy\n")
    assert count == 0
    assert tuple(masks.shape) == (0, HEIGHT, WIDTH)


def test_label_prefix_on_a_line_is_ignored():
    masks, count = _run("xyxy: 120 180 640 760")
    assert count == 1
    assert _boxes_of(masks) == [[77, 86, 410, 365]]


# ------------------------------------------------------------- text formats


def test_bare_lines_in_absolute_mode():
    masks, count = _run("10,20,110,70\n300 200 400 320", coord_mode="absolute")
    assert count == 2
    assert tuple(masks.shape) == (2, HEIGHT, WIDTH)
    assert _boxes_of(masks) == [[10, 20, 110, 70], [300, 200, 400, 320]]


def test_consecutive_bare_boxes_on_one_line():
    masks, count = _run("10 20 110 70 300 200 400 320", coord_mode="absolute")
    assert count == 2
    assert _boxes_of(masks) == [[10, 20, 110, 70], [300, 200, 400, 320]]


def test_leading_line_index_is_dropped():
    masks, count = _run("1. 10 20 110 70", coord_mode="absolute")
    assert count == 1
    assert _boxes_of(masks) == [[10, 20, 110, 70]]


def test_json_array_uses_normalized_units_by_default():
    masks, count = _run("[[187,59,808,188]]")
    assert count == 1
    assert _boxes_of(masks) == [[120, 28, 517, 90]]


def test_json_object_keys():
    masks, count = _run('{"bbox": [120, 180, 640, 760]}')
    assert count == 1
    assert _boxes_of(masks) == [[77, 86, 410, 365]]

    masks, count = _run(
        '[{"x1": 1, "y1": 2, "x2": 30, "y2": 40}, {"x": 5, "y": 6, "w": 10, "h": 20}]',
        coord_mode="absolute",
    )
    assert count == 2
    assert _boxes_of(masks) == [[1, 2, 30, 40], [5, 6, 15, 26]]


def test_function_style_and_unclosed_tag():
    assert _boxes_of(_run("bbox(120, 180, 640, 760)")[0]) == [[77, 86, 410, 365]]
    assert _boxes_of(_run("<bbox>120 180 640 760")[0]) == [[77, 86, 410, 365]]


def test_fullwidth_punctuation_is_normalised():
    assert _boxes_of(_run("＜bbox＞１２０，１８０，６４０，７６０＜／bbox＞")[0]) == [
        [77, 86, 410, 365]
    ]


# ------------------------------------------------------- layer bounding_box

LAYER_JSON = """{
  "model": "doubao-seedream-5-0-pro-260628",
  "data": [
    {"url": "https://x/0.png", "size": "2048x2048", "z_index": 0},
    {"url": "https://x/1.png", "size": "1273x265", "z_index": 1,
     "bounding_box": {"absolute": [383, 120, 1655, 384], "normalized": [187, 59, 808, 188]},
     "name": "Seedream标题文字"},
    {"url": "https://x/2.png", "size": "492x98", "z_index": 2,
     "bounding_box": {"absolute": [140, 451, 631, 548], "normalized": [68, 220, 308, 268]},
     "name": "左上角标语文字"}
  ]
}"""


def test_layer_json_normalized_field_keeps_rounding_error():
    masks, count = _run(LAYER_JSON, image=_image(width=2048, height=2048))
    # 只取 normalized：每层一个框，乘回 2048 时官方提示的 ±1px 取整误差确实存在
    assert count == 2
    assert _boxes_of(masks) == [[383, 121, 1655, 385], [139, 451, 631, 549]]


def test_layer_json_absolute_field_is_exact():
    masks, count = _run(
        LAYER_JSON, image=_image(width=2048, height=2048), coord_mode="absolute"
    )
    assert count == 2
    assert _boxes_of(masks) == [[383, 120, 1655, 384], [140, 451, 631, 548]]


def test_layer_json_falls_back_to_the_only_field_present():
    only_normalized = '{"bounding_box": {"normalized": [187, 59, 808, 188]}}'
    masks, _ = _run(
        only_normalized, image=_image(width=2048, height=2048), coord_mode="absolute"
    )
    assert _boxes_of(masks) == [[383, 121, 1655, 385]]

    only_absolute = '{"bounding_box": {"absolute": [383, 120, 1655, 384]}}'
    masks, _ = _run(
        only_absolute, image=_image(width=2048, height=2048), coord_mode="normalized"
    )
    assert _boxes_of(masks) == [[383, 120, 1655, 384]]


# ------------------------------------------------------------------- guards


def test_pixel_values_in_normalized_mode_are_rejected():
    with pytest.raises(ValueError, match="coord_mode"):
        _run("<bbox>1200 300 1800 900</bbox>")


def test_out_of_range_absolute_coordinates_are_clamped():
    masks, _ = _run("<bbox>900 900 1200 1200</bbox>", coord_mode="absolute")
    assert _boxes_of(masks) == [[639, 479, 640, 480]]


# ------------------------------------------------------------------- empties


def test_empty_input_shapes():
    empty, count = _run("")
    assert tuple(empty.shape) == (0, HEIGHT, WIDTH)
    assert count == 0

    merged, count = _run("", output_mode="merge")
    assert tuple(merged.shape) == (1, HEIGHT, WIDTH)
    assert float(merged.sum()) == 0.0
    assert count == 0


def test_batch_image_keeps_mask_batch():
    masks, count = _run("<bbox>0 0 100 100</bbox>", image=_image(batch=3))
    assert count == 1
    assert tuple(masks.shape) == (1, HEIGHT, WIDTH)
    assert _boxes_of(masks) == [[0, 0, 64, 48]]
