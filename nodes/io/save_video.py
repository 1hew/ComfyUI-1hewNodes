# 保存路径说明：
# - 默认文件名 "video/ComfyUI"（相对路径）会将视频保存到 ComfyUI 配置的 output 目录下。
# - output 目录由 folder_paths.get_output_directory() 决定：
#   * 无 --output-directory 参数 → 默认 <comfyui_root>/output
#   * 带 --output-directory <路径> 参数 → 自定义路径
# - 输入绝对路径时会跳过 output 目录，直接保存到指定位置。
from __future__ import annotations

import asyncio
import json
import os
import shutil
from dataclasses import dataclass
from typing import Optional

import av
import folder_paths
import numpy as np
from comfy.cli_args import args
from comfy_api.input import VideoInput
from comfy_api.latest import io, ui
from comfy_api.util import VideoContainer

from ...utils._file_ops import (
    is_absolute_like,
    sanitize_filename_prefix,
    split_prefix_to_dir_and_stem,
    remove_file_with_retries,
)

try:
    from comfy.utils import ProgressBar
except Exception:
    ProgressBar = None


_PATH_LOCK: Optional[asyncio.Lock] = None


def _relative_subfolder(path: str, base_dir: str) -> str:
    try:
        rel = os.path.relpath(os.path.dirname(path), base_dir)
    except ValueError:
        return ""
    if rel in {".", ""}:
        return ""
    return rel


def _new_progress_bar(total: int):
    if ProgressBar is None:
        return None
    if int(total or 0) <= 0:
        return None
    try:
        return ProgressBar(int(total))
    except Exception:
        return None


async def _replace_file_with_retries(
    src: str,
    dst: str,
    *,
    attempts: int = 20,
    base_delay_s: float = 0.05,
) -> None:
    for attempt in range(attempts):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if attempt >= attempts - 1:
                raise
            await asyncio.sleep(base_delay_s * (attempt + 1))


# Containers PyAV can mux every stream of, keyed by output container. Matroska
# is the fallback container because it accepts virtually any codec, which keeps
# the fast remux path available for formats such as AVI.
_CONTAINER_CODECS = {
    "webm": {
        "video": frozenset({"vp8", "vp9", "av1"}),
        "audio": frozenset({"opus", "vorbis"}),
        "subtitle": frozenset({"webvtt"}),
    },
    "mp4": {
        "video": frozenset({"h264", "hevc", "av1", "mpeg4", "vp9", "mjpeg"}),
        "audio": frozenset({"aac", "mp3", "opus", "ac3", "eac3", "flac", "alac"}),
        "subtitle": frozenset({"mov_text"}),
    },
}

# Source extensions that map onto a container this node can write. Anything
# else (avi, ts, ...) is remuxed into Matroska instead of being mislabelled.
_CONTAINER_EXTENSIONS = {
    "mp4": "mp4",
    "m4v": "mp4",
    "mov": "mp4",
    "mkv": "mkv",
    "webm": "webm",
}


@dataclass(frozen=True)
class _StreamProbe:
    """Container, codecs and frame size of a source, as reported by FFmpeg."""

    container_names: tuple[str, ...]
    video_codecs: tuple[str, ...]
    audio_codecs: tuple[str, ...]
    subtitle_codecs: tuple[str, ...]
    width: int = 0
    height: int = 0

    def fits(self, container: str) -> bool:
        allowed = _CONTAINER_CODECS.get(container)
        if allowed is None:
            return True
        return (
            all(codec in allowed["video"] for codec in self.video_codecs)
            and all(codec in allowed["audio"] for codec in self.audio_codecs)
            and all(codec in allowed["subtitle"] for codec in self.subtitle_codecs)
        )


def _rewind(source) -> None:
    """Reset a file-like source so probing and muxing start at the beginning."""
    seek = getattr(source, "seek", None)
    if seek is None:
        return
    try:
        seek(0)
    except (OSError, ValueError):
        pass


def _add_stream_from_template(container, template):
    """Copy a source stream into a new container without re-encoding it.

    PyAV 14+ removed the ``add_stream(template=...)`` keyword in favour of
    ``add_stream_from_template()``, which is what current ComfyUI PyAV builds
    expose; keep the old call as a fallback.
    """
    add_from_template = getattr(container, "add_stream_from_template", None)
    if add_from_template is not None:
        try:
            return add_from_template(template=template, opaque=True)
        except TypeError:
            return add_from_template(template=template)
    return container.add_stream(template=template)


def _source_extension(path) -> str:
    if not isinstance(path, str):
        return ""
    extension = os.path.splitext(path)[1].lstrip(".").lower()
    return _CONTAINER_EXTENSIONS.get(extension, "")


def _probe_extension(probe: _StreamProbe) -> str:
    if "webm" in probe.container_names and probe.fits("webm"):
        return "webm"
    if "mp4" in probe.container_names and probe.fits("mp4"):
        return "mp4"
    return "mkv"


class SaveVideo(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="1hew_SaveVideo",
            display_name="Save Video",
            category="1hewNodes/io",
            inputs=[
                io.Video.Input("video", optional=True, tooltip="要保存的视频；为空时节点直接通过。"),
                io.String.Input("filename", default="video/ComfyUI", tooltip="保存文件名或路径前缀；支持格式占位符（如 %date:yyyy-MM-dd%）。"),
                io.Boolean.Input(
                    "auto_increment",
                    default=True,
                    tooltip="为 True 时自动编号；为 False 时固定文件名并覆盖。",
                ),
                io.Boolean.Input("save_output", default=True, tooltip="是否保存到输出目录；若为 False 则保存到临时目录。"),
                io.Boolean.Input("save_metadata", default=True, tooltip="是否写入 prompt/workflow 元数据。"),
            ],
            outputs=[
                io.String.Output(display_name="file_path"),
            ],
            hidden=[io.Hidden.prompt, io.Hidden.extra_pnginfo],
            is_output_node=True,
        )

    @classmethod
    async def execute(
        cls,
        video: Optional[VideoInput] = None,
        filename: str = "video/ComfyUI",
        auto_increment: bool = True,
        save_output: bool = True,
        save_metadata: bool = True,
    ) -> io.NodeOutput:
        if video is None:
            return io.NodeOutput()

        output_dir = folder_paths.get_output_directory()
        output_type = io.FolderType.output
        if not save_output:
            output_dir = folder_paths.get_temp_directory()
            output_type = io.FolderType.temp

        width, height = video.get_dimensions()

        path_attr = cls._coerce_path(getattr(video, "path", None))
        source_path = path_attr
        if not isinstance(source_path, str):
            source_path = cls._coerce_path(getattr(video, "source_path", None))
        if not isinstance(source_path, str) or not os.path.isfile(source_path):
            source_path = None

        # ComfyUI's file-backed videos (VideoFromFile) expose the original
        # stream through get_stream_source() rather than a path attribute.
        # That stream is only inspected here (container, codecs, alpha); the
        # write below still goes through save_to so trims, crops and colour
        # changes survive.
        source_stream = source_path
        if source_stream is None:
            source_stream = cls._stream_source(video)

        has_alpha = False
        if source_stream is not None:
            has_alpha = await cls.check_has_alpha(source_stream)

        extension = await cls._resolve_extension(
            video, width, height, source_path, source_stream
        )

        global _PATH_LOCK
        if _PATH_LOCK is None:
            _PATH_LOCK = asyncio.Lock()

        raw_prefix = str(filename or "").strip()
        safe_filename_prefix = sanitize_filename_prefix(filename, "video")
        preview_file = ""
        preview_path = ""
        preview_subfolder = ""

        # auto lets save_to() pick the muxer and encoder from the extension.
        format = "auto"
        codec = "auto"

        async with _PATH_LOCK:
            if auto_increment:
                if is_absolute_like(raw_prefix):
                    target_dir_raw, filename = split_prefix_to_dir_and_stem(raw_prefix)
                    full_output_folder = (
                        os.path.abspath(target_dir_raw)
                        if target_dir_raw
                        else os.path.abspath(output_dir)
                    )
                    os.makedirs(full_output_folder, exist_ok=True)
                    subfolder = ""
                    counter = 1
                    while os.path.exists(
                        os.path.join(
                            full_output_folder,
                            f"{filename}_{counter:05}_.{extension}",
                        )
                    ):
                        counter += 1
                else:
                    (
                        full_output_folder,
                        filename,
                        counter,
                        subfolder,
                        _resolved_prefix,
                    ) = folder_paths.get_save_image_path(
                        safe_filename_prefix, output_dir, width, height
                    )

                output_file = f"{filename}_{counter:05}_.{extension}"
                path = os.path.join(full_output_folder, output_file)

                preview_file = output_file
                preview_path = path
                preview_subfolder = subfolder

                if has_alpha:
                    preview_dir = os.path.join(
                        folder_paths.get_temp_directory(),
                        subfolder,
                    )
                    os.makedirs(preview_dir, exist_ok=True)
                    preview_file = f"{filename}_{counter:05}_.webm"
                    preview_path = os.path.join(preview_dir, preview_file)
                    preview_subfolder = subfolder
            else:
                if is_absolute_like(raw_prefix):
                    target_dir_raw, filename = split_prefix_to_dir_and_stem(raw_prefix)
                    full_output_folder = (
                        os.path.abspath(target_dir_raw)
                        if target_dir_raw
                        else os.path.abspath(output_dir)
                    )
                else:
                    rel_dir_raw, filename = split_prefix_to_dir_and_stem(
                        safe_filename_prefix
                    )
                    rel_dir = rel_dir_raw.replace("/", os.sep) if rel_dir_raw else ""
                    full_output_folder = os.path.abspath(
                        os.path.join(output_dir, rel_dir)
                    )

                os.makedirs(full_output_folder, exist_ok=True)
                output_file = f"{filename}.{extension}"
                path = os.path.join(full_output_folder, output_file)
                preview_file = output_file
                preview_path = path
                preview_subfolder = _relative_subfolder(path, output_dir)

                if has_alpha:
                    preview_dir = os.path.join(
                        folder_paths.get_temp_directory(),
                        preview_subfolder,
                    )
                    os.makedirs(preview_dir, exist_ok=True)
                    preview_file = f"{filename}.webm"
                    preview_path = os.path.join(preview_dir, preview_file)
                    preview_subfolder = _relative_subfolder(
                        preview_path, folder_paths.get_temp_directory()
                    )

            if not os.path.exists(path):
                with open(path, "wb"):
                    pass

            if has_alpha and preview_path != path:
                if not os.path.exists(preview_path):
                    with open(preview_path, "wb"):
                        pass

        metadata_comment = None
        if save_metadata and not args.disable_metadata:
            metadata: dict[str, object] = {}
            if cls.hidden.extra_pnginfo is not None:
                metadata.update(cls.hidden.extra_pnginfo)
            if cls.hidden.prompt is not None:
                metadata["prompt"] = cls.hidden.prompt
            if metadata:
                metadata_comment = json.dumps(
                    metadata,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )

        progress_steps = 1
        if has_alpha and preview_path != path:
            progress_steps += 1
        progress_bar = _new_progress_bar(progress_steps)

        # ComfyUI video implementations may expose the original file as
        # either ``path`` or ``source_path``. Prefer the original stream when
        # available so alpha-capable MOV/ProRes files are not unnecessarily
        # decoded and re-encoded as yuv420p.
        if source_path is not None:
            # The source already sits at the target path when a file is saved
            # over itself (fixed filename + same directory); leave it alone.
            if os.path.abspath(source_path) != os.path.abspath(path):
                has_audio = await cls._has_audio_stream(source_path)
                if has_audio:
                    await asyncio.to_thread(shutil.copy2, source_path, path)
                else:
                    try:
                        await cls._write_silent_audio_video(
                            input_path=source_path,
                            output_path=path,
                        )
                    except Exception as e:
                        # The target container cannot carry the silent track:
                        # keep the original file instead of failing the save.
                        print(f"Info: Silent audio skipped: {e}")
                        await remove_file_with_retries(path)
                        await asyncio.to_thread(shutil.copy2, source_path, path)
        else:
            try:
                await asyncio.to_thread(
                    video.save_to,
                    path,
                    format=format,
                    codec=codec,
                    metadata=None,
                )
            except Exception:
                cls._strip_video_audio(video)
                await asyncio.to_thread(
                    video.save_to,
                    path,
                    format=format,
                    codec=codec,
                    metadata=None,
                )
        if progress_bar is not None:
            try:
                progress_bar.update(1)
            except Exception:
                pass

        await cls._remux_metadata(
            input_path=path,
            metadata_comment=metadata_comment,
        )

        if has_alpha and preview_path != path:
            await cls.generate_preview(
                path,
                preview_path,
                strip_metadata=not save_metadata,
            )
            if progress_bar is not None:
                try:
                    progress_bar.update(1)
                except Exception:
                    pass

        file_path = os.path.abspath(path)

        return io.NodeOutput(
            file_path,
            ui=ui.PreviewVideo(
                [
                    ui.SavedResult(
                        preview_file,
                        preview_subfolder,
                        io.FolderType.temp if has_alpha else output_type,
                    )
                ]
            )
        )

    @staticmethod
    async def _remux_metadata(
        input_path: str,
        metadata_comment: Optional[str],
    ):
        if not os.path.isfile(input_path):
            return

        base, ext = os.path.splitext(input_path)
        tmp_path = f"{base}.tmp{ext}"

        def _remux():
            with av.open(input_path) as inp:
                with av.open(tmp_path, "w") as out:
                    stream_map = {}
                    for s in inp.streams:
                        stream_map[s.index] = _add_stream_from_template(out, s)

                    if metadata_comment:
                        out.metadata["comment"] = metadata_comment

                    for packet in inp.demux():
                        if packet.dts is None:
                            continue
                        packet.stream = stream_map[packet.stream.index]
                        out.mux(packet)

        try:
            await asyncio.to_thread(_remux)
        except Exception as e:
            if os.path.exists(tmp_path):
                await remove_file_with_retries(tmp_path)
            print(f"Info: Metadata remux skipped: {e}")
            return

        await _replace_file_with_retries(tmp_path, input_path)

    @staticmethod
    async def check_has_alpha(source) -> bool:
        try:
            def _check():
                _rewind(source)
                try:
                    with av.open(source) as container:
                        if not container.streams.video:
                            return False
                        stream = container.streams.video[0]
                        codec_context = stream.codec_context
                        pix_fmt = getattr(codec_context, "pix_fmt", "") or ""
                        context_format = getattr(codec_context, "format", None)
                        context_format_name = getattr(context_format, "name", "") or ""
                        stream_format = getattr(getattr(stream, "format", None), "name", "") or ""
                        probe = f"{pix_fmt} {context_format_name} {stream_format}".lower()
                        alpha_formats = [
                            "yuva", "rgba", "argb", "abgr", "bgra",
                            "gbrap", "ya8", "ya16", "ayuv",
                        ]
                        return any(fmt in probe for fmt in alpha_formats)
                finally:
                    _rewind(source)
            return await asyncio.to_thread(_check)
        except Exception:
            return False

    @staticmethod
    def _coerce_path(value) -> Optional[str]:
        if value is None:
            return None
        try:
            return os.fspath(value)
        except TypeError:
            return None

    @classmethod
    def _stream_source(cls, video: object):
        """Return an openable source (a path or file object) for video, if cheap.

        File-backed videos such as ComfyUI's VideoFromFile override both
        get_stream_source() and get_container_format(); component-based videos
        only encode themselves inside get_stream_source(), so they are skipped
        to avoid encoding the whole video just to inspect it.
        """
        stream_method = getattr(type(video), "get_stream_source", None)
        container_method = getattr(type(video), "get_container_format", None)
        base_stream = getattr(VideoInput, "get_stream_source", None)
        base_container = getattr(VideoInput, "get_container_format", None)
        if (
            stream_method is None
            or stream_method is base_stream
            or container_method is None
            or container_method is base_container
        ):
            return None

        try:
            source = video.get_stream_source()
        except Exception:
            return None

        if isinstance(source, str):
            return source if os.path.isfile(source) else None
        if hasattr(source, "read"):
            return source
        return None

    @classmethod
    async def _probe_streams(cls, source) -> Optional[_StreamProbe]:
        """Inspect the source container and codecs without encoding anything."""
        if source is None:
            return None

        def _probe() -> _StreamProbe:
            _rewind(source)
            with av.open(source) as container:
                names = tuple(
                    name
                    for name in str(container.format.name or "").split(",")
                    if name
                )
                codecs: dict[str, list[str]] = {
                    "video": [],
                    "audio": [],
                    "subtitle": [],
                }
                width = 0
                height = 0
                for stream in container.streams:
                    try:
                        codec = stream.codec.canonical_name
                    except Exception:
                        codec = ""
                    if codec and stream.type in codecs:
                        codecs[stream.type].append(codec)
                    if stream.type == "video" and not width:
                        width = int(getattr(stream, "width", 0) or 0)
                        height = int(getattr(stream, "height", 0) or 0)
                return _StreamProbe(
                    container_names=names,
                    video_codecs=tuple(codecs["video"]),
                    audio_codecs=tuple(codecs["audio"]),
                    subtitle_codecs=tuple(codecs["subtitle"]),
                    width=width,
                    height=height,
                )

        try:
            probe = await asyncio.to_thread(_probe)
        except Exception:
            return None
        finally:
            _rewind(source)
        return probe

    @classmethod
    async def _resolve_extension(cls, video, width, height, source_path, source_stream) -> str:
        """Pick an extension whose container can hold the source streams.

        Falling back to MP4 for a WebM/VP8 source makes save_to() remux VP8
        into MP4, which FFmpeg rejects with "'mp4' format does not support
        'vp8' codec".
        """
        if source_path is not None:
            # Path-backed videos are copied verbatim, so keep their extension.
            extension = os.path.splitext(source_path)[1].lstrip(".").lower()
            if extension:
                return extension
        probe = await cls._probe_streams(source_stream)
        if probe is not None and cls._requires_reencode(video, probe, width, height):
            # Trimmed/cropped sources are re-encoded by save_to(); MP4/H.264 is
            # far cheaper than the AV1 encode a WebM container would force.
            return VideoContainer.get_extension(VideoContainer.AUTO)
        extension = _source_extension(source_stream)
        if extension and (probe is None or probe.fits(extension)):
            return extension
        if probe is not None:
            return _probe_extension(probe)
        return VideoContainer.get_extension(VideoContainer.AUTO)

    @staticmethod
    def _requires_reencode(video, probe: _StreamProbe, width: int, height: int) -> bool:
        """True when save_to() has to decode the source instead of remuxing it."""
        try:
            start_time, duration = video.get_active_trim_window()
        except Exception:
            start_time, duration = 0.0, 0.0
        if start_time or duration:
            return True
        # Cropped videos report the crop size while the source keeps its own.
        if probe.width and probe.height and (probe.width, probe.height) != (width, height):
            return True
        return False

    @staticmethod
    def _strip_video_audio(video: object) -> None:
        attr_names = [
            "_VideoInput__components",
            "_ComfyVideo__components",
        ]
        for attr in attr_names:
            try:
                components = getattr(video, attr)
                if hasattr(components, "audio"):
                    setattr(components, "audio", None)
            except Exception:
                pass

        for attr in dir(video):
            if "__components" not in attr and "components" not in attr.lower():
                continue
            try:
                components = getattr(video, attr)
                if hasattr(components, "audio"):
                    setattr(components, "audio", None)
            except Exception:
                pass

    @staticmethod
    async def _has_audio_stream(path: str) -> bool:
        try:
            def _check():
                with av.open(path) as container:
                    return len(container.streams.audio) > 0
            return await asyncio.to_thread(_check)
        except Exception:
            return False

    @staticmethod
    async def _write_silent_audio_video(
        input_path: str,
        output_path: str,
    ) -> None:
        _, ext = os.path.splitext(output_path)
        ext = ext.lower()
        audio_codec = "aac"
        if ext in {".webm", ".mkv"}:
            audio_codec = "libopus"

        def _mux():
            # movflags only belongs to the ISO-BMFF muxers; other containers
            # reject it and log an unused-option warning.
            container_options = {}
            if ext in {".mp4", ".m4v", ".mov"}:
                container_options["movflags"] = "+faststart"

            with av.open(input_path) as inp:
                duration_sec = max((inp.duration or 0) / 1_000_000.0, 0.1)

                with av.open(output_path, "w", options=container_options) as out:
                    in_video = inp.streams.video[0]
                    out_video = _add_stream_from_template(out, in_video)

                    sample_rate = 48000 if audio_codec == "libopus" else 44100
                    out_audio = out.add_stream(audio_codec, rate=sample_rate, layout="stereo")
                    if audio_codec == "aac":
                        out_audio.bit_rate = 128_000

                    for packet in inp.demux(in_video):
                        if packet.dts is None:
                            continue
                        packet.stream = out_video
                        out.mux(packet)

                    samples_per_frame = 1024
                    total_samples = int(sample_rate * duration_sec)
                    pts = 0
                    while pts < total_samples:
                        n = min(samples_per_frame, total_samples - pts)
                        silence = np.zeros((2, n), dtype=np.float32)
                        frame = av.AudioFrame.from_ndarray(silence, format="fltp", layout="stereo")
                        frame.sample_rate = sample_rate
                        frame.pts = pts
                        for pkt in out_audio.encode(frame):
                            out.mux(pkt)
                        pts += n

                    for pkt in out_audio.encode(None):
                        out.mux(pkt)

        await asyncio.to_thread(_mux)

    @staticmethod
    async def generate_preview(
        input_path: str,
        output_path: str,
        strip_metadata: bool = False,
    ):
        try:
            def _generate():
                with av.open(input_path) as inp:
                    in_video = inp.streams.video[0]

                    with av.open(output_path, "w") as out:
                        out_video = out.add_stream("libvpx-vp9")
                        out_video.width = in_video.width
                        out_video.height = in_video.height
                        out_video.pix_fmt = "yuva420p"
                        out_video.rate = in_video.average_rate or 30
                        out_video.options = {"crf": "30", "auto-alt-ref": "0"}

                        out_audio = None
                        if inp.streams.audio:
                            in_audio = inp.streams.audio[0]
                            out_audio = out.add_stream("libopus", rate=in_audio.rate or 48000)

                        if not strip_metadata:
                            out.metadata.update(inp.metadata)

                        for packet in inp.demux():
                            if packet.stream.type == "video":
                                for frame in packet.decode():
                                    frame = frame.reformat(format="yuva420p")
                                    for out_pkt in out_video.encode(frame):
                                        out.mux(out_pkt)
                            elif packet.stream.type == "audio" and out_audio:
                                for frame in packet.decode():
                                    frame.pts = None
                                    for out_pkt in out_audio.encode(frame):
                                        out.mux(out_pkt)

                        for pkt in out_video.encode(None):
                            out.mux(pkt)
                        if out_audio:
                            for pkt in out_audio.encode(None):
                                out.mux(pkt)

            await asyncio.to_thread(_generate)
        except Exception as e:
            print(f"Info: Preview generation skipped: {e}")
