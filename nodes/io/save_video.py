from __future__ import annotations

import asyncio
import json
import os
import shutil
from typing import Optional

import av
import folder_paths
import numpy as np
from comfy.cli_args import args
from comfy_api.input import VideoInput
from comfy_api.latest import io, ui
from comfy_api.util import VideoContainer

try:
    from comfy.utils import ProgressBar
except Exception:
    ProgressBar = None


_PATH_LOCK: Optional[asyncio.Lock] = None


def _sanitize_filename_prefix(filename: str) -> str:
    if not filename:
        return "video/ComfyUI"

    normalized = filename.replace("\\", "/").strip()
    drive, tail = os.path.splitdrive(normalized)
    if os.path.isabs(normalized) or drive:
        if drive:
            drive_token = drive.rstrip(":").replace("/", "_").replace("\\", "_")
            tail = tail.lstrip("/\\")
            normalized = f"{drive_token}/{tail}" if tail else drive_token
        else:
            normalized = normalized.lstrip("/\\")

    parts: list[str] = []
    for part in normalized.split("/"):
        if not part or part in {".", ".."}:
            continue
        parts.append(part)

    return "/".join(parts) if parts else "video/ComfyUI"


def _is_absolute_like(path_value: str) -> bool:
    raw = str(path_value or "").strip().replace("\\", "/")
    if not raw:
        return False
    drive, _ = os.path.splitdrive(raw)
    return bool(drive) or raw.startswith("/")


def _sanitize_filename_stem(name: str) -> str:
    stem = str(name or "").strip()
    if not stem:
        return "ComfyUI"
    for ch in '<>:"/\\|?*':
        stem = stem.replace(ch, "_")
    stem = stem.rstrip(". ")
    return stem if stem else "ComfyUI"


def _split_prefix_to_dir_and_stem(prefix: str) -> tuple[str, str]:
    raw = str(prefix or "").strip().replace("\\", "/")
    if not raw:
        return "", "ComfyUI"

    is_trailing_sep = raw.endswith("/")
    cleaned = raw.rstrip("/")
    if not cleaned:
        return "", "ComfyUI"

    if is_trailing_sep:
        return cleaned, "ComfyUI"
    if "/" not in cleaned:
        stem = os.path.splitext(cleaned)[0]
        return "", _sanitize_filename_stem(stem)

    dir_part, stem_part = cleaned.rsplit("/", 1)
    stem = os.path.splitext(stem_part)[0]
    return dir_part, _sanitize_filename_stem(stem)


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


async def _remove_file_with_retries(
    path: str,
    *,
    attempts: int = 20,
    base_delay_s: float = 0.05,
) -> None:
    for attempt in range(attempts):
        try:
            os.remove(path)
            return
        except FileNotFoundError:
            return
        except PermissionError:
            if attempt >= attempts - 1:
                return
            await asyncio.sleep(base_delay_s * (attempt + 1))


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
        video: Optional[VideoInput],
        filename: str,
        auto_increment: bool,
        save_output: bool,
        save_metadata: bool,
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

        has_alpha = False
        if isinstance(source_path, str) and os.path.exists(source_path):
            has_alpha = await cls.check_has_alpha(source_path)

        global _PATH_LOCK
        if _PATH_LOCK is None:
            _PATH_LOCK = asyncio.Lock()

        raw_prefix = str(filename or "").strip()
        safe_filename_prefix = _sanitize_filename_prefix(filename)
        preview_file = ""
        preview_path = ""
        preview_subfolder = ""

        async with _PATH_LOCK:
            format = "auto"
            codec = "auto"
            extension = VideoContainer.get_extension(format)

            # Try to use source extension if available
            if isinstance(source_path, str):
                _, source_ext = os.path.splitext(source_path)
                if source_ext:
                    extension = source_ext.lstrip(".")

            if auto_increment:
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
                if _is_absolute_like(raw_prefix):
                    target_dir_raw, filename = _split_prefix_to_dir_and_stem(raw_prefix)
                    full_output_folder = (
                        os.path.abspath(target_dir_raw)
                        if target_dir_raw
                        else os.path.abspath(output_dir)
                    )
                else:
                    rel_dir_raw, filename = _split_prefix_to_dir_and_stem(
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

        if isinstance(path_attr, str) and os.path.isfile(path_attr):
            has_audio = await cls._has_audio_stream(source_path)
            if has_audio:
                await asyncio.to_thread(shutil.copy2, path_attr, path)
            else:
                await cls._write_silent_audio_video(
                    input_path=path_attr,
                    output_path=path,
                )
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
                        stream_map[s.index] = out.add_stream(template=s)

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
                await _remove_file_with_retries(tmp_path)
            print(f"Info: Metadata remux skipped: {e}")
            return

        await _replace_file_with_retries(tmp_path, input_path)

    @staticmethod
    async def check_has_alpha(path: str) -> bool:
        try:
            def _check():
                with av.open(path) as container:
                    if not container.streams.video:
                        return False
                    pix_fmt = container.streams.video[0].codec_context.pix_fmt or ""
                    alpha_formats = [
                        "yuva", "rgba", "argb", "abgr", "bgra",
                        "gbrap", "ya8", "ya16", "ayuv",
                    ]
                    return any(fmt in pix_fmt for fmt in alpha_formats)
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
            container_options = {}
            if audio_codec == "aac":
                container_options["movflags"] = "+faststart"

            with av.open(input_path) as inp:
                duration_sec = max((inp.duration or 0) / 1_000_000.0, 0.1)

                with av.open(output_path, "w", options=container_options) as out:
                    in_video = inp.streams.video[0]
                    out_video = out.add_stream(template=in_video)

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
