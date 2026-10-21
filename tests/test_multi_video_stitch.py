"""Regression tests for Multi Video Stitch frame/audio alignment logic."""

import sys
import types

import torch

from _node_loader import load_node


def _load_node():
    try:
        from comfy_api.latest import io  # noqa: F401
    except Exception:
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
    return load_node(
        "multi_video_stitch_under_test",
        "nodes/multi/multi_video_stitch.py",
    )


MVS = _load_node().MultiVideoStitch


class _FakeVideo:
    def __init__(self, width, height):
        self._width = width
        self._height = height

    def get_dimensions(self):
        return self._width, self._height


def test_collect_ordered_videos_ignores_non_suffixed_inputs():
    first, second = object(), object()
    kwargs = {
        "video_2": second,
        "get_video_size": object(),
        "video_1": first,
        "video_x": object(),
    }
    ordered = MVS._collect_ordered_videos(kwargs)
    assert [index for index, _ in ordered] == [1, 2]
    assert ordered[0][1] is first
    assert ordered[1][1] is second


def test_collect_ordered_videos_skips_unconnected():
    ordered = MVS._collect_ordered_videos({"video_1": None, "video_2": object()})
    assert [index for index, _ in ordered] == [2]


def test_resolve_target_fps_by_video_reference():
    decoded = [{"index": 1, "frame_rate": 30.0}, {"index": 2, "frame_rate": 24.0}]
    assert MVS._resolve_target_fps("video_1", decoded) == 30.0
    assert MVS._resolve_target_fps("video_2", decoded) == 24.0
    assert MVS._resolve_target_fps("video2", decoded) == 24.0
    assert MVS._resolve_target_fps("v2", decoded) == 24.0
    assert MVS._resolve_target_fps("V_2", decoded) == 24.0
    assert MVS._resolve_target_fps("video 2", decoded) == 24.0
    assert MVS._resolve_target_fps("video_3", decoded) == 30.0
    assert (
        MVS._resolve_target_fps(
            "video_2",
            [{"index": 1, "frame_rate": 30.0}, {"index": 2, "frame_rate": 0.0}],
        )
        == 30.0
    )


def test_resolve_target_fps_by_number():
    decoded = [{"index": 1, "frame_rate": 30.0}, {"index": 2, "frame_rate": 24.0}]
    assert MVS._resolve_target_fps("29.5", decoded) == 29.5
    assert MVS._resolve_target_fps("8", decoded) == 8.0
    assert MVS._resolve_target_fps("24", decoded) == 24.0


def test_resolve_target_fps_auto_alias_and_fallback():
    decoded = [{"index": 1, "frame_rate": 30.0}, {"index": 2, "frame_rate": 24.0}]
    assert MVS._resolve_target_fps("auto", decoded) == 30.0
    assert MVS._resolve_target_fps("", decoded) == 30.0
    assert MVS._resolve_target_fps("0", decoded) == 30.0
    assert MVS._resolve_target_fps("abc", decoded) == 30.0
    assert MVS._resolve_target_fps("auto", [{"index": 1, "frame_rate": 0.0}]) == 8.0


def test_resolve_target_size_prefers_reference_video():
    decoded = [{"images": torch.zeros(2, 10, 20, 3)}]
    assert MVS._resolve_target_size(None, decoded) == (10, 20)
    assert MVS._resolve_target_size(_FakeVideo(64, 48), decoded) == (48, 64)


def test_fit_frames_pad_uses_target_size_and_color():
    frames = torch.rand(3, 10, 20, 3)
    out = MVS._fit_frames(frames, 20, 20, "pad", (1.0, 0.0, 0.0), 3)
    assert tuple(out.shape) == (3, 20, 20, 3)
    assert torch.allclose(out[:, 0, 0, :], torch.tensor([1.0, 0.0, 0.0]).expand(3, 3))


def test_fit_frames_stretch_and_crop():
    frames = torch.rand(2, 8, 12, 3)
    assert tuple(MVS._fit_frames(frames, 16, 16, "stretch", (1.0, 1.0, 1.0), 3).shape) == (2, 16, 16, 3)
    assert tuple(MVS._fit_frames(frames, 16, 16, "crop", (1.0, 1.0, 1.0), 3).shape) == (2, 16, 16, 3)


def test_fit_frames_promotes_to_rgba():
    frames = torch.rand(2, 4, 4, 3)
    out = MVS._fit_frames(frames, 4, 4, "pad", (1.0, 1.0, 1.0), 4)
    assert tuple(out.shape) == (2, 4, 4, 4)
    assert torch.allclose(out[..., 3], torch.ones(2, 4, 4))


def test_resample_frames_keeps_duration():
    frames = torch.arange(30, dtype=torch.float32).view(30, 1, 1, 1)
    assert tuple(MVS._resample_frames(frames, 30.0, 15.0).shape) == (15, 1, 1, 1)
    assert tuple(MVS._resample_frames(frames, 30.0, 30.0).shape) == (30, 1, 1, 1)
    assert tuple(MVS._resample_frames(frames, 0.0, 24.0).shape) == (30, 1, 1, 1)


def test_concat_audio_pads_silence_for_missing_tracks():
    audio = {"waveform": torch.ones(1, 2, 100), "sample_rate": 100}
    merged = MVS._concat_audio([audio, None], [1.0, 2.0])
    assert merged is not None
    assert merged["sample_rate"] == 100
    assert tuple(merged["waveform"].shape) == (1, 2, 300)
    assert torch.allclose(merged["waveform"][0, :, :100], torch.ones(2, 100))
    assert torch.allclose(merged["waveform"][0, :, 100:], torch.zeros(2, 200))


def test_concat_audio_returns_none_without_any_track():
    assert MVS._concat_audio([None, None], [1.0, 1.0]) is None
