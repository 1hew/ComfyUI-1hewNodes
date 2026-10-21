"""Save Video must keep the source container instead of always writing MP4.

ComfyUI's file-backed videos (VideoFromFile) expose the original stream via
'get_stream_source()' and have no 'path'/'source_path' attribute. Save Video
used to fall back to a '.mp4' name for every such input, so a WebM/VP8 source
was remuxed into an MP4 container and FFmpeg failed with
"'mp4' format does not support 'vp8' codec".
"""

import asyncio
import io
import os
import sys
import types
from pathlib import Path

import av
import numpy as np
import pytest

from _node_loader import load_node

ROOT = Path(__file__).parents[1]


class _NodeOutput:
    def __init__(self, *values, ui=None, **kwargs):
        self.values = values
        self.ui = ui


class _Port:
    def __init__(self, name):
        self.name = name

    def Input(self, *args, **kwargs):
        return ("input", self.name, args, kwargs)

    def Output(self, *args, **kwargs):
        return ("output", self.name, args, kwargs)


class _VideoInput:
    """Stand-in for comfy_api.input.VideoInput and its default methods."""

    def get_stream_source(self):
        raise NotImplementedError

    def get_container_format(self):
        raise NotImplementedError


class _VideoContainer:
    AUTO = "auto"
    MP4 = "mp4"
    MKV = "mkv"
    WEBM = "webm"

    @classmethod
    def get_extension(cls, value):
        return {
            "auto": "mp4",
            "mp4": "mp4",
            "mkv": "mkv",
            "webm": "webm",
        }.get(value, "")


def _load_node(directory):
    folders = types.ModuleType("folder_paths")
    folders.get_output_directory = lambda: directory
    folders.get_temp_directory = lambda: directory
    folders.get_save_image_path = lambda prefix, *args, **kwargs: (
        directory,
        os.path.basename(prefix) or "ComfyUI",
        1,
        "",
        prefix,
    )

    cli = types.ModuleType("comfy.cli_args")
    # Skips the hidden prompt/extra_pnginfo access, which the stub class lacks.
    cli.args = types.SimpleNamespace(disable_metadata=True)

    io_module = types.SimpleNamespace(
        ComfyNode=object,
        Schema=lambda **kwargs: kwargs,
        NodeOutput=_NodeOutput,
        FolderType=types.SimpleNamespace(output="output", temp="temp"),
        Hidden=types.SimpleNamespace(prompt="prompt", extra_pnginfo="extra_pnginfo"),
        Video=_Port("VIDEO"),
        String=_Port("STRING"),
        Boolean=_Port("BOOLEAN"),
    )
    ui = types.SimpleNamespace(
        PreviewVideo=lambda values: values,
        SavedResult=lambda *values: values,
    )
    latest = types.ModuleType("comfy_api.latest")
    latest.io = io_module
    latest.ui = ui
    inputs = types.ModuleType("comfy_api.input")
    inputs.VideoInput = _VideoInput
    util = types.ModuleType("comfy_api.util")
    util.VideoContainer = _VideoContainer

    managed = [
        "folder_paths",
        "comfy",
        "comfy.cli_args",
        "comfy_api",
        "comfy_api.latest",
        "comfy_api.input",
        "comfy_api.util",
    ]
    previous = {name: sys.modules.get(name) for name in managed}
    sys.modules["folder_paths"] = folders
    sys.modules["comfy"] = types.ModuleType("comfy")
    sys.modules["comfy.cli_args"] = cli
    sys.modules["comfy_api"] = types.ModuleType("comfy_api")
    sys.modules["comfy_api.latest"] = latest
    sys.modules["comfy_api.input"] = inputs
    sys.modules["comfy_api.util"] = util
    try:
        module = load_node("save_video_container_test", "nodes/io/save_video.py")
    finally:
        for name, value in previous.items():
            if value is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value
    return module


def _write_video(path, codec="libvpx", container_format=None):
    """Encode a tiny 16x16 clip so the node can probe real stream metadata."""
    try:
        av.codec.Codec(codec, "w")
    except Exception:  # pragma: no cover - encoder missing in this build
        pytest.skip(f"{codec} encoder is unavailable")
    with av.open(str(path), "w", format=container_format) as output:
        stream = output.add_stream(codec, rate=10)
        stream.width = 16
        stream.height = 16
        stream.pix_fmt = "yuv420p"
        for _ in range(3):
            frame = av.VideoFrame.from_ndarray(
                np.zeros((16, 16, 3), dtype=np.uint8), format="rgb24"
            )
            for packet in stream.encode(frame.reformat(format="yuv420p")):
                output.mux(packet)
        for packet in stream.encode(None):
            output.mux(packet)


class _FileVideo(_VideoInput):
    """Mimics ComfyUI's VideoFromFile: a source stream, but no path attribute."""

    def __init__(self, path, width=16, height=16, trim=(0.0, 0.0)):
        # Deliberately no 'path' attribute: ComfyUI's VideoFromFile has none.
        self.source = Path(path)
        self.width = width
        self.height = height
        self.trim = trim
        self.saved = []

    def get_dimensions(self):
        return self.width, self.height

    def get_stream_source(self):
        return str(self.source)

    def get_container_format(self):
        with av.open(str(self.source)) as container:
            return container.format.name

    def get_active_trim_window(self):
        return self.trim

    def save_to(self, path, format="auto", codec="auto", metadata=None):
        self.saved.append((path, format, codec))
        with open(path, "wb") as handle:
            handle.write(b"saved")


class _ComponentVideo(_VideoInput):
    """Mimics VideoFromComponents: streaming would encode the whole video."""

    def __init__(self):
        self.stream_source_calls = 0
        self.saved = []

    def get_dimensions(self):
        return 16, 16

    def get_stream_source(self):
        self.stream_source_calls += 1
        raise AssertionError("component videos must not be streamed for probing")

    def save_to(self, path, format="auto", codec="auto", metadata=None):
        self.saved.append((path, format, codec))
        with open(path, "wb") as handle:
            handle.write(b"saved")


def test_webm_source_is_saved_with_webm_extension(tmp_path):
    _write_video(tmp_path / "source.webm", "libvpx", "webm")
    module = _load_node(str(tmp_path))
    video = _FileVideo(tmp_path / "source.webm")

    result = asyncio.run(
        module.SaveVideo.execute(
            video=video,
            filename="video/ComfyUI",
            auto_increment=True,
            save_output=True,
            save_metadata=False,
        )
    )

    saved_path = result.values[0]
    assert saved_path.endswith(".webm")
    assert video.saved and video.saved[0][0].endswith(".webm")


def test_mp4_source_keeps_mp4_extension(tmp_path):
    _write_video(tmp_path / "source.mp4", "libx264")
    module = _load_node(str(tmp_path))
    video = _FileVideo(tmp_path / "source.mp4")

    result = asyncio.run(
        module.SaveVideo.execute(
            video=video,
            filename="video/ComfyUI",
            auto_increment=True,
            save_output=True,
            save_metadata=False,
        )
    )

    assert result.values[0].endswith(".mp4")


def test_extension_falls_back_to_matroska_for_unsupported_codecs(tmp_path):
    module = _load_node(str(tmp_path))
    probe = module._StreamProbe(
        container_names=("mov", "mp4", "m4a", "3gp", "3g2", "mj2"),
        video_codecs=("prores",),
        audio_codecs=(),
        subtitle_codecs=(),
        width=16,
        height=16,
    )

    assert module._probe_extension(probe) == "mkv"


def test_component_video_is_not_probed(tmp_path):
    module = _load_node(str(tmp_path))
    video = _ComponentVideo()

    assert module.SaveVideo._stream_source(video) is None
    extension = asyncio.run(
        module.SaveVideo._resolve_extension(video, 16, 16, None, None)
    )

    assert extension == "mp4"
    assert video.stream_source_calls == 0


def test_trimmed_source_uses_mp4_because_it_is_re_encoded(tmp_path):
    _write_video(tmp_path / "source.webm", "libvpx", "webm")
    module = _load_node(str(tmp_path))
    video = _FileVideo(tmp_path / "source.webm", trim=(0.0, 0.5))

    extension = asyncio.run(
        module.SaveVideo._resolve_extension(
            video, 16, 16, None, video.get_stream_source()
        )
    )

    assert extension == "mp4"


def test_cropped_source_uses_mp4_because_it_is_re_encoded(tmp_path):
    _write_video(tmp_path / "source.webm", "libvpx", "webm")
    module = _load_node(str(tmp_path))
    video = _FileVideo(tmp_path / "source.webm", width=8, height=8)

    extension = asyncio.run(
        module.SaveVideo._resolve_extension(
            video, 8, 8, None, video.get_stream_source()
        )
    )

    assert extension == "mp4"


def test_path_backed_video_keeps_its_extension(tmp_path):
    _write_video(tmp_path / "source.webm", "libvpx", "webm")
    module = _load_node(str(tmp_path))
    video = _FileVideo(tmp_path / "source.webm")

    extension = asyncio.run(
        module.SaveVideo._resolve_extension(
            video, 16, 16, str(video.source), video.get_stream_source()
        )
    )

    assert extension == "webm"


def test_alpha_probe_accepts_a_stream(tmp_path):
    _write_video(tmp_path / "source.webm", "libvpx", "webm")
    module = _load_node(str(tmp_path))
    stream = io.BytesIO((tmp_path / "source.webm").read_bytes())

    assert asyncio.run(module.SaveVideo.check_has_alpha(stream)) is False
    assert stream.tell() == 0
