from __future__ import annotations

import asyncio
from fractions import Fraction
import hashlib
from io import BytesIO
import os
import subprocess
import time
from typing import Optional

from aiohttp import web
import av
import folder_paths
import numpy as np
import torch
from comfy_api.latest import io

from ...utils._video_io import (
    VALID_VIDEO_EXTENSIONS,
    VideoComponents,
    VideoFromFile,
    apply_frame_subset,
    force_frame_rate,
    get_video_paths,
    resolve_video_path,
)
from server import PromptServer

try:
    from comfy_api.input_impl.video_types import VideoFromFile as ComfyVideoFromFile
except Exception:
    ComfyVideoFromFile = None

try:
    from comfy.utils import ProgressBar
except Exception:
    ProgressBar = None




def _get_filename_stem(path: str) -> str:
    base_name = os.path.basename(str(path or "").strip())
    stem, _ext = os.path.splitext(base_name)
    return stem or "video"






class VideoFromFileWithSettings:
    def __init__(
        self,
        path: str,
        start_skip: int,
        end_skip: int,
        fps: float,
        frame_limit: int,
        format: str,
    ):
        self.path = path
        self._video = VideoFromFile(path)
        self._start_skip = int(start_skip or 0)
        self._end_skip = int(end_skip or 0)
        self._fps = float(fps or 0.0)
        self._frame_limit = int(frame_limit or 0)
        self._format = format or "4n+1"

    def get_dimensions(self):
        return self._video.get_dimensions()

    def save_to(self, path, format="auto", codec="auto", metadata=None):
        info = VideoFromFile._probe_video_info_with_ffprobe(self.path)
        source_fps = float(info.get("fps") or 0.0)
        
        # Optimization: If no settings change the video, just copy
        is_default_format = (not self._format) or (self._format.lower() in ("n", "", "default"))
        if (
            self._start_skip == 0
            and self._end_skip == 0
            and self._frame_limit == 0
            and (self._fps <= 0 or (source_fps > 0 and abs(self._fps - source_fps) < 0.01))
            and is_default_format
        ):
            return self._video.save_to(path, format, codec, metadata)

        components = self.get_components()
        if components is None or components.images is None or components.images.shape[0] == 0:
            return path

        images = components.images
        audio = components.audio
        fps = float(components.frame_rate or 0.0)
        if not (0.0 < fps < 1000.0):
            fps = 1.0

        if images.device.type != "cpu":
            images = images.cpu()
            
        images_np = (images * 255.0).clamp(0, 255).byte().numpy()
        
        T, H, W, C = images_np.shape
        has_alpha = C == 4
        input_fmt = "rgba" if has_alpha else "rgb24"

        _, path_ext = os.path.splitext(path)
        output_ext = path_ext.lower()
        use_alpha = has_alpha or output_ext == ".webm"

        out_audio = None
        audio_np = None
        a_sr = 44100
        ch = 2
        layout = "stereo"
        audio_codec_name = "libopus" if output_ext == ".webm" else "aac"

        if audio is not None and audio.get("waveform") is not None:
            waveform = audio["waveform"]
            a_sr = int(audio.get("sample_rate", 44100))
            if waveform.dim() == 2:
                waveform = waveform.unsqueeze(0)
            if waveform.device.type != "cpu":
                waveform = waveform.cpu()
            waveform_np = waveform.numpy()
            if waveform_np.shape[0] > 0:
                audio_np = waveform_np[0]
                if audio_np.shape[0] > 2:
                    audio_np = audio_np[:2]
                ch = audio_np.shape[0]
                layout = "stereo" if ch >= 2 else "mono"

        with av.open(path, "w") as output:
            if has_alpha and output_ext in {".mov", ".qt"}:
                out_video = output.add_stream("prores_ks")
                out_video.pix_fmt = "yuva444p10le"
                out_video.options = {"profile": "4"}
            elif use_alpha:
                out_video = output.add_stream("libvpx-vp9")
                out_video.pix_fmt = "yuva420p"
                out_video.options = {"auto-alt-ref": "0", "b": "0", "crf": "23"}
            elif codec not in ("auto", None):
                out_video = output.add_stream(codec)
                out_video.pix_fmt = "yuv420p"
            else:
                out_video = output.add_stream("libx264")
                out_video.pix_fmt = "yuv420p"
                out_video.options = {"preset": "fast", "crf": "23"}
            out_video.width = W
            out_video.height = H
            out_video.rate = Fraction(fps).limit_denominator(10001)

            if audio_np is not None:
                out_audio = output.add_stream(audio_codec_name, rate=a_sr, layout=layout)
                if audio_codec_name == "aac":
                    out_audio.bit_rate = 128_000

            for i in range(T):
                frame = av.VideoFrame.from_ndarray(images_np[i], format=input_fmt)
                frame = frame.reformat(format=out_video.pix_fmt)
                for pkt in out_video.encode(frame):
                    output.mux(pkt)

            for pkt in out_video.encode(None):
                output.mux(pkt)

            if out_audio is not None and audio_np is not None:
                video_dur = T / fps if fps > 0 else 0.0
                max_s = int(video_dur * a_sr) if video_dur > 0 else audio_np.shape[1]
                total_s = min(audio_np.shape[1], max_s)

                pts = 0
                while pts < total_s:
                    n = min(1024, total_s - pts)
                    chunk = audio_np[:ch, pts:pts + n].copy().astype(np.float32)
                    af = av.AudioFrame.from_ndarray(chunk, format="fltp", layout=layout)
                    af.sample_rate = a_sr
                    af.pts = pts
                    for pkt in out_audio.encode(af):
                        output.mux(pkt)
                    pts += n

                for pkt in out_audio.encode(None):
                    output.mux(pkt)

        return path

    def get_components(self) -> Optional[VideoComponents]:
        components = self._video.get_components()
        if components is None:
            return None

        images = components.images
        audio = components.audio
        source_fps = float(components.frame_rate or 0.0)
        if images is not None:
            images, audio, fps = LoadVideo.apply_video_settings(
                images=images,
                audio=audio,
                fps=self._fps,
                source_fps=source_fps,
                frame_limit=self._frame_limit,
                start_skip=self._start_skip,
                end_skip=self._end_skip,
                format=self._format,
            )
        else:
            fps = source_fps

        frame_count = images.shape[0] if images is not None else 0
        if frame_count == 0:
            return VideoComponents(images=None, audio=None, frame_rate=0.0)

        return VideoComponents(images=images, audio=audio, frame_rate=fps)


def _processed_video_cache_dir() -> str:
    return os.path.join(folder_paths.get_temp_directory(), "1hew_processed_videos")


def _encode_components_to_video_path(components: VideoComponents, out_path: str) -> str:
    if components is None or components.images is None:
        return out_path

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    base, ext = os.path.splitext(out_path)
    tmp_path = f"{base}.{time.time_ns()}.tmp{ext}"

    images = components.images
    audio = components.audio
    fps = float(components.frame_rate or 0.0)
    if not (0.0 < fps < 1000.0):
        fps = 1.0

    if images.device.type != "cpu":
        images = images.cpu()

    images_np = (images * 255.0).clamp(0, 255).byte().numpy()

    _, height, width, channels = images_np.shape
    has_alpha = int(channels) == 4
    input_pix_fmt = "rgba" if has_alpha else "rgb24"

    out_ext = ext.lower()

    out_audio_obj = None
    audio_np = None
    a_sr = 44100
    ch = 2
    layout = "stereo"

    if has_alpha or out_ext == ".webm":
        audio_codec_name = "libopus"
    else:
        audio_codec_name = "aac"

    if audio is not None and audio.get("waveform") is not None:
        waveform = audio["waveform"]
        a_sr = int(audio.get("sample_rate", 44100))
        if waveform.dim() == 2:
            waveform = waveform.unsqueeze(0)
        if waveform.device.type != "cpu":
            waveform = waveform.cpu()
        waveform_np = waveform.numpy()
        if waveform_np.shape[0] > 0:
            audio_np = waveform_np[0]
            if audio_np.shape[0] > 2:
                audio_np = audio_np[:2]
            ch = audio_np.shape[0]
            layout = "stereo" if ch >= 2 else "mono"

    with av.open(tmp_path, "w") as output:
        if has_alpha or out_ext == ".webm":
            out_video = output.add_stream("libvpx-vp9")
            out_video.pix_fmt = "yuva420p"
            out_video.options = {
                "auto-alt-ref": "0",
                "b": "0",
                "crf": "33",
                "deadline": "realtime",
                "cpu-used": "5",
            }
        else:
            out_video = output.add_stream("libx264")
            out_video.pix_fmt = "yuv420p"
            out_video.options = {"preset": "fast", "crf": "23"}

        out_video.width = width
        out_video.height = height
        out_video.rate = Fraction(fps).limit_denominator(10001)

        if audio_np is not None:
            out_audio_obj = output.add_stream(audio_codec_name, rate=a_sr, layout=layout)
            if audio_codec_name == "aac":
                out_audio_obj.bit_rate = 128_000

        T = images_np.shape[0]
        for i in range(T):
            frame = av.VideoFrame.from_ndarray(images_np[i], format=input_pix_fmt)
            frame = frame.reformat(format=out_video.pix_fmt)
            for pkt in out_video.encode(frame):
                output.mux(pkt)

        for pkt in out_video.encode(None):
            output.mux(pkt)

        if out_audio_obj is not None and audio_np is not None:
            video_dur = T / fps if fps > 0 else 0.0
            max_s = int(video_dur * a_sr) if video_dur > 0 else audio_np.shape[1]
            total_s = min(audio_np.shape[1], max_s)

            pts = 0
            while pts < total_s:
                n = min(1024, total_s - pts)
                chunk = audio_np[:ch, pts:pts + n].copy().astype(np.float32)
                af = av.AudioFrame.from_ndarray(chunk, format="fltp", layout=layout)
                af.sample_rate = a_sr
                af.pts = pts
                for pkt in out_audio_obj.encode(af):
                    output.mux(pkt)
                pts += n

            for pkt in out_audio_obj.encode(None):
                output.mux(pkt)

    try:
        os.replace(tmp_path, out_path)
    except OSError:
        return out_path

    return out_path


class LoadVideo(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        format_options = ["n", "2n+1", "4n+1", "6n+1", "8n+1"]
        return io.Schema(
            node_id="1hew_LoadVideo",
            display_name="Load Video",
            category="1hewNodes/io",
            inputs=[
                io.String.Input("file", default=""),
                io.Int.Input("frame_limit", default=0, min=0, max=100000, step=1),
                io.Float.Input("fps", default=0.0, min=0.0, max=120.0, step=1.0),
                io.Int.Input("start_skip", default=0, min=0, max=100000, step=1),
                io.Int.Input("end_skip", default=0, min=0, max=100000, step=1),
                io.Combo.Input("format", options=format_options, default="4n+1"),
                io.Int.Input("video_index", default=0, min=-8192, max=8192, step=1),
                io.Boolean.Input("include_subdir", default=False),
            ],
            outputs=[
                io.Video.Output(display_name="video"),
                io.String.Output(display_name="filename"),
            ],
        )

    @classmethod
    async def execute(
        cls,
        file: str,
        start_skip: int,
        end_skip: int,
        fps: float,
        frame_limit: int,
        format: str,
        video_index: int,
        include_subdir: bool,
    ) -> io.NodeOutput:
        file = (file or "").strip().strip('"').strip("'")
        video_paths = get_video_paths(file, include_subdir)
        count = len(video_paths)

        if count == 0:
            return io.NodeOutput(None, "")

        idx = video_index % count
        selected = video_paths[idx]
        filename = _get_filename_stem(selected)

        start_skip = int(start_skip or 0)
        end_skip = int(end_skip or 0)
        frame_limit = int(frame_limit or 0)
        fps = float(fps or 0.0)
        format = (format or "4n+1").strip()

        is_default_format = (not format) or (
            format.strip().lower() in {"n", "", "default"}
        )
        needs_fps_change = fps > 0.0
        if needs_fps_change:
            info = VideoFromFile._probe_video_info_with_ffprobe(selected)
            source_fps = float(info.get("fps") or 0.0)
            if source_fps <= 0.0 or abs(fps - source_fps) >= 0.01:
                needs_fps_change = True
            else:
                needs_fps_change = False

        if (
            start_skip == 0
            and end_skip == 0
            and frame_limit == 0
            and not needs_fps_change
            and is_default_format
        ):
            if ComfyVideoFromFile is not None:
                return io.NodeOutput(ComfyVideoFromFile(selected), filename)
            return io.NodeOutput(VideoFromFile(selected), filename)

        video = VideoFromFileWithSettings(
            path=selected,
            start_skip=start_skip,
            end_skip=end_skip,
            fps=fps,
            frame_limit=frame_limit,
            format=format,
        )

        components = video.get_components()
        if components is None or components.images is None:
            return io.NodeOutput(None, filename)

        settings_key = (
            f"video_index:{int(video_index)}|start_skip:{int(start_skip)}"
            f"|end_skip:{int(end_skip)}|fps:{float(fps)}"
            f"|frame_limit:{int(frame_limit)}|format:{(format or '4n+1').strip()}"
        )
        try:
            mtime = os.path.getmtime(selected)
        except OSError:
            mtime = 0.0

        key = hashlib.sha256(
            f"{os.path.abspath(selected)}:{mtime}:{settings_key}".encode()
        ).hexdigest()
        cache_dir = _processed_video_cache_dir()
        os.makedirs(cache_dir, exist_ok=True)

        out_ext = ".webm" if int(components.images.shape[-1]) == 4 else ".mp4"
        out_path = os.path.join(cache_dir, f"{key}{out_ext}")

        if not os.path.isfile(out_path):
            await asyncio.to_thread(_encode_components_to_video_path, components, out_path)

        if ComfyVideoFromFile is not None:
            return io.NodeOutput(ComfyVideoFromFile(out_path), filename)
        return io.NodeOutput(VideoFromFile(out_path), filename)

    @staticmethod
    def apply_video_settings(
        images: torch.Tensor,
        audio: Optional[dict],
        source_fps: float,
        fps: float,
        frame_limit: int,
        start_skip: int,
        end_skip: int,
        format: str,
    ) -> tuple[torch.Tensor, Optional[dict], float]:
        images = apply_frame_subset(
            images=images,
            start_skip=int(start_skip or 0),
            end_skip=int(end_skip or 0),
            frame_limit=0,
        )

        # Handle Audio Start Skip
        if audio is not None and start_skip > 0 and source_fps > 0:
            try:
                sample_rate = audio.get("sample_rate", 44100)
                waveform = audio.get("waveform")
                if waveform is not None:
                    start_sec = float(start_skip) / float(source_fps)
                    start_sample = int(start_sec * sample_rate)
                    if start_sample < waveform.shape[-1]:
                        audio["waveform"] = waveform[..., start_sample:]
                    else:
                        audio["waveform"] = waveform[..., :0]
            except Exception:
                pass

        images = force_frame_rate(
            images=images,
            source_fps=source_fps,
            target_fps=float(fps or 0.0),
        )

        images = LoadVideo._apply_format_constraint(
            images=images,
            format=format or "4n+1",
        )

        limit = int(frame_limit or 0)
        if limit > 0 and images is not None:
            images = images[:limit]

        fps = LoadVideo._compute_output_fps(
            source_fps=float(source_fps or 0.0),
            fps=float(fps or 0.0),
            format=format or "Default",
        )

        # Handle Audio End Trimming (Match final video duration)
        if audio is not None and images is not None and fps > 0:
            try:
                frame_count = images.shape[0]
                duration_sec = float(frame_count) / float(fps)
                sample_rate = audio.get("sample_rate", 44100)
                waveform = audio.get("waveform")
                
                if waveform is not None:
                    target_samples = int(duration_sec * sample_rate)
                    if waveform.shape[-1] > target_samples:
                         audio["waveform"] = waveform[..., :target_samples]
            except Exception:
                pass

        return images, audio, fps



    @staticmethod
    def _apply_format_constraint(images: torch.Tensor, format: str) -> torch.Tensor:
        if images is None or images.shape[0] == 0:
            return images

        f = (format or "").strip().lower()
        if f in ("n", ""):
            return images
        if f == "default":
            return images

        step = 1
        mod = 0

        if "n" in f:
            parts = f.split("n")
            try:
                if parts[0]:
                    step = int(parts[0])
                if len(parts) > 1 and parts[1]:
                    mod = int(parts[1])
            except ValueError:
                return images

        count = images.shape[0]
        if count < mod:
            return images[:0]

        k = (count - mod) // step
        target_count = step * k + mod
        target_count = max(0, target_count)

        return images[:target_count]

    @staticmethod
    def _compute_output_fps(
        source_fps: float,
        fps: float,
        format: str,
    ) -> float:
        base_fps = float(fps or 0.0) if float(fps or 0.0) > 0.0 else 0.0
        if base_fps <= 0.0:
            base_fps = float(source_fps or 0.0)

        return float(base_fps or 0.0)



    @classmethod
    def IS_CHANGED(cls, file, include_subdir, **kwargs):
        file = resolve_video_path(file)
        start_skip = int(kwargs.get("start_skip") or 0)
        end_skip = int(kwargs.get("end_skip") or 0)
        fps = float(kwargs.get("fps") or 0.0)
        frame_limit = int(kwargs.get("frame_limit") or 0)
        format = (kwargs.get("format") or "4n+1").strip()
        video_index = int(kwargs.get("video_index") or 0)
        settings_key = (
            f"video_index:{video_index}|start_skip:{start_skip}|end_skip:{end_skip}"
            f"|fps:{fps}|frame_limit:{frame_limit}|format:{format}"
        )
        if os.path.isfile(file):
            try:
                mtime = os.path.getmtime(file)
                return hashlib.sha256(
                    f"{file}:{mtime}:{settings_key}".encode()
                ).hexdigest()
            except OSError:
                return float("nan")

        if not os.path.isdir(file):
            return float("nan")

        video_paths = get_video_paths(file, include_subdir)
        m = hashlib.sha256()
        m.update(f"{settings_key}|include_subdir:{include_subdir}".encode())
        for video_path in video_paths:
            try:
                mtime = os.path.getmtime(video_path)
                m.update(f"{video_path}:{mtime}".encode())
            except OSError:
                continue

        return m.hexdigest()


def _safe_filename(name: str) -> str:
    if not name:
        return "video"
    base = os.path.basename(name)
    base = base.replace("\\", "_").replace("/", "_").strip()
    return base or "video"


def _get_upload_base_dir() -> str:
    get_input_dir = getattr(folder_paths, "get_input_directory", None)
    if callable(get_input_dir):
        return get_input_dir()
    return folder_paths.get_temp_directory()


def _sanitize_relative_path(name: str) -> str:
    if not name:
        return "video"

    rel = name.replace("\\", "/").lstrip("/").strip()
    rel = os.path.normpath(rel)
    rel = rel.lstrip("\\/").strip()

    if rel in {".", ""}:
        return "video"

    parts = [p for p in rel.split(os.sep) if p not in {"", ".", ".."}]
    if not parts:
        return "video"

    return os.path.join(*parts)


def _preview_cache_dir() -> str:
    return os.path.join(folder_paths.get_temp_directory(), "1hew_video_previews")


def _preview_content_type_for_path(path: str) -> str:
    ext = os.path.splitext(str(path or "").strip())[1].lower()
    if ext == ".webm":
        return "video/webm"
    return "video/mp4"


def _is_direct_preview_video(path: str) -> bool:
    ext = os.path.splitext(str(path or "").strip())[1].lower()
    return ext in {".mp4", ".webm", ".m4v"}


def _has_alpha_pix_fmt(pix_fmt: str) -> bool:
    pix_fmt = (pix_fmt or "").lower()
    return any(
        token in pix_fmt
        for token in (
            "yuva", "rgba", "bgra", "argb", "abgr",
            "gbrap", "ya8", "ya16", "ayuv",
        )
    )


def _ensure_preview_proxy_webm(source_path: str, include_audio: bool = True) -> str:
    source_path = os.path.abspath(source_path)
    try:
        mtime = os.path.getmtime(source_path)
    except OSError:
        return ""

    key = hashlib.sha256(f"{source_path}:{mtime}:pyav-vp9-v1".encode()).hexdigest()
    cache_dir = _preview_cache_dir()
    os.makedirs(cache_dir, exist_ok=True)
    out_path = os.path.join(cache_dir, f"{key}.webm")

    if os.path.isfile(out_path):
        try:
            if os.path.getmtime(out_path) >= mtime:
                return out_path
        except OSError:
            pass

    tmp_path = os.path.join(cache_dir, f"{key}.{time.time_ns()}.tmp.webm")

    try:
        with av.open(source_path) as inp:
            in_video = inp.streams.video[0]
            has_alpha = VideoFromFile._prefer_alpha_for_stream(in_video)
            output_pix_fmt = "yuva420p" if has_alpha else "yuv420p"
            width = int(in_video.width or 0) // 2 * 2
            height = int(in_video.height or 0) // 2 * 2

            with av.open(tmp_path, "w") as out:
                out_video = out.add_stream("libvpx-vp9")
                out_video.width = width
                out_video.height = height
                out_video.pix_fmt = output_pix_fmt
                out_video.rate = in_video.average_rate or 30
                out_video.options = {
                    "auto-alt-ref": "0",
                    "b": "0",
                    "crf": "33",
                    "deadline": "realtime",
                    "cpu-used": "5",
                }

                out_audio = None
                if include_audio and inp.streams.audio:
                    in_audio = inp.streams.audio[0]
                    out_audio = out.add_stream("libopus", rate=in_audio.rate or 48000)

                for packet in inp.demux():
                    if packet.stream.type == "video":
                        for frame in packet.decode():
                            frame = frame.reformat(
                                width=width,
                                height=height,
                                format=output_pix_fmt,
                            )
                            for out_packet in out_video.encode(frame):
                                out.mux(out_packet)
                    elif packet.stream.type == "audio" and out_audio is not None:
                        for frame in packet.decode():
                            frame.pts = None
                            for out_packet in out_audio.encode(frame):
                                out.mux(out_packet)

                for out_packet in out_video.encode(None):
                    out.mux(out_packet)
                if out_audio is not None:
                    for out_packet in out_audio.encode(None):
                        out.mux(out_packet)

        os.replace(tmp_path, out_path)
        return out_path
    except Exception as exc:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except OSError:
            pass
        print(f"Info: PyAV WebM preview conversion failed: {exc}")
        return ""


def _ensure_preview_proxy(
    source_path: str, include_audio: bool = True
) -> tuple[str, str]:
    source_path = os.path.abspath(source_path)
    try:
        mtime = os.path.getmtime(source_path)
    except OSError:
        return "", ""

    info = VideoFromFile._probe_video_info_with_ffprobe(source_path)
    pix_fmt = info.get("pix_fmt") or ""
    has_alpha = _has_alpha_pix_fmt(pix_fmt)

    if has_alpha:
        out_path = _ensure_preview_proxy_webm(source_path, include_audio=include_audio)
        if out_path:
            return out_path, "video/webm"
        print(
            "Info: Falling back to an opaque MP4 preview for Alpha video: "
            f"{source_path}"
        )

    key = hashlib.sha256(f"{source_path}:{mtime}:mp4".encode()).hexdigest()
    cache_dir = _preview_cache_dir()
    os.makedirs(cache_dir, exist_ok=True)
    out_path = os.path.join(cache_dir, f"{key}.mp4")

    if os.path.isfile(out_path):
        try:
            if os.path.getmtime(out_path) >= mtime:
                return out_path, "video/mp4"
        except OSError:
            pass

    ffmpeg_exe = VideoFromFile._resolve_ffmpeg_exe()
    tmp_path = os.path.join(cache_dir, f"{key}.{time.time_ns()}.tmp.mp4")

    vf = (
        "scale='min(1280,iw)':-2,"
        "scale=trunc(iw/2)*2:trunc(ih/2)*2"
    )
    command = [
        ffmpeg_exe,
        "-y",
        "-v",
        "error",
        "-i",
        source_path,
        "-vf",
        vf,
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-preset",
        "veryfast",
        "-crf",
        "28",
        "-movflags",
        "+faststart",
    ]
    if include_audio:
        command.extend(
            [
                "-map",
                "0:v:0",
                "-map",
                "0:a:0?",
                "-c:a",
                "aac",
                "-b:a",
                "128k",
                "-shortest",
            ]
        )
    else:
        command.append("-an")
    command.append(tmp_path)

    try:
        res = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
        )
    except Exception:
        return "", ""

    if res.returncode != 0:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except OSError:
            pass
        return "", ""

    try:
        os.replace(tmp_path, out_path)
    except OSError:
        return "", ""

    return out_path, "video/mp4"


@PromptServer.instance.routes.post("/1hew/upload_video")
async def upload_video(request):
    reader = await request.multipart()
    field = await reader.next()
    if field is None:
        return web.json_response({"error": "missing file"}, status=400)

    filename = _safe_filename(field.filename or "video")
    ext = os.path.splitext(filename)[1].lower()
    if ext not in VALID_VIDEO_EXTENSIONS:
        return web.json_response({"error": "invalid extension"}, status=400)

    base_dir = _get_upload_base_dir()
    target_dir = os.path.join(base_dir, "1hew_uploads")
    os.makedirs(target_dir, exist_ok=True)

    stem = os.path.splitext(filename)[0] or "video"
    unique = str(time.time_ns())
    out_name = f"{stem}_{unique}{ext}"
    out_path = os.path.abspath(os.path.join(target_dir, out_name))

    with open(out_path, "wb") as f:
        while True:
            chunk = await field.read_chunk()
            if not chunk:
                break
            f.write(chunk)

    return web.json_response({"path": out_path})


@PromptServer.instance.routes.post("/1hew/upload_videos")
async def upload_videos(request):
    reader = await request.multipart()

    base_dir = _get_upload_base_dir()
    target_root = os.path.join(base_dir, "1hew_uploads")
    folder_id = str(time.time_ns())
    target_dir = os.path.abspath(os.path.join(target_root, folder_id))
    os.makedirs(target_dir, exist_ok=True)

    saved: list[str] = []

    while True:
        field = await reader.next()
        if field is None:
            break

        filename = field.filename or "video"
        rel = _sanitize_relative_path(filename)
        ext = os.path.splitext(rel)[1].lower()
        if ext not in VALID_VIDEO_EXTENSIONS:
            continue

        out_path = os.path.abspath(os.path.join(target_dir, rel))
        try:
            common = os.path.commonpath([target_dir, out_path])
        except ValueError:
            continue
        if common != target_dir:
            continue

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "wb") as f:
            while True:
                chunk = await field.read_chunk()
                if not chunk:
                    break
                f.write(chunk)
        saved.append(out_path)

    if not saved:
        return web.json_response({"error": "no valid files"}, status=400)

    return web.json_response({"folder": target_dir, "count": len(saved)})


@PromptServer.instance.routes.get("/1hew/video_info_from_folder")
async def video_info_from_folder(request):
    path = request.query.get("file") or request.query.get("path") or request.query.get(
        "folder"
    )
    index_str = (
        request.query.get("index") or request.query.get("video_index") or "0"
    )
    include_subdir = (
        (request.query.get("include_subdir") or "true").lower()
        == "true"
    )

    if not path:
        return web.json_response({}, status=404)

    path = path.strip().strip('"').strip("'")

    selected = ""
    if os.path.isfile(path):
        selected = path
    elif os.path.isdir(path):
        try:
            index = int(index_str)
        except ValueError:
            index = 0

        video_paths = get_video_paths(path, include_subdir)
        if not video_paths:
            return web.json_response({}, status=404)

        idx = index % len(video_paths)
        selected = video_paths[idx]
    else:
        return web.json_response({}, status=404)

    fps = 0.0
    frame_count = 0
    width = 0
    height = 0
    duration = 0.0

    try:
        with av.open(selected) as container:
            if len(container.streams.video) > 0:
                stream = container.streams.video[0]
                width = int(stream.width or 0)
                height = int(stream.height or 0)

                try:
                    fps = float(stream.average_rate)
                except Exception:
                    fps = 0.0

                frame_count = int(stream.frames or 0)

                try:
                    if stream.duration and stream.time_base:
                        duration = float(stream.duration * stream.time_base)
                except Exception:
                    duration = 0.0

                if frame_count <= 0 and duration > 0.0 and fps > 0.0:
                    frame_count = int(round(duration * fps))
    except Exception:
        pass

    if fps <= 0.0 or width <= 0 or height <= 0:
        info = VideoFromFile._probe_video_info_with_ffprobe(selected)
        width = width if width > 0 else int(info.get("width") or 0)
        height = height if height > 0 else int(info.get("height") or 0)
        fps = fps if fps > 0.0 else float(info.get("fps") or 0.0)

    return web.json_response(
        {
            "path": selected,
            "fps": float(fps or 0.0),
            "frame_count": int(frame_count or 0),
            "width": int(width or 0),
            "height": int(height or 0),
            "duration": float(duration or 0.0),
        }
    )


@PromptServer.instance.routes.get("/1hew/view_video_from_folder")
async def view_video_from_folder(request):
    path = request.query.get("file") or request.query.get("path") or request.query.get(
        "folder"
    )
    index_str = (
        request.query.get("index") or request.query.get("video_index") or "0"
    )
    include_subdir = (
        (request.query.get("include_subdir") or "true").lower()
        == "true"
    )

    if not path:
        return web.Response(status=404)

    path = path.strip().strip('"').strip("'")
    want_raw = (request.query.get("raw") or "").lower() in {
        "1",
        "true",
        "yes",
    }
    want_preview = (request.query.get("preview") or "").lower() in {
        "1",
        "true",
        "yes",
    }
    want_audio = (request.query.get("audio") or "true").lower() in {
        "1",
        "true",
        "yes",
    }

    if os.path.isfile(path):
        if want_raw:
            return web.FileResponse(path)
        if want_preview:
            if _is_direct_preview_video(path):
                return web.FileResponse(
                    path,
                    headers={"Content-Type": _preview_content_type_for_path(path)},
                )
            proxy_path, content_type = _ensure_preview_proxy(
                path, include_audio=want_audio
            )
            if proxy_path and content_type:
                return web.FileResponse(
                    proxy_path,
                    headers={"Content-Type": content_type},
                )
        if path.lower().endswith(".mov"):
            preview = _ensure_preview_proxy_webm(path, include_audio=want_audio)
            if preview:
                return web.FileResponse(preview, headers={"Content-Type": "video/webm"})
            return web.Response(text="MOV preview conversion failed", status=415)
        return web.FileResponse(path)

    if not os.path.isdir(path):
        return web.Response(status=404)

    try:
        index = int(index_str)
    except ValueError:
        index = 0

    video_paths = get_video_paths(path, include_subdir)
    if not video_paths:
        return web.Response(status=404)

    idx = index % len(video_paths)
    selected = video_paths[idx]

    if want_raw:
        return web.FileResponse(selected)
    if want_preview:
        if _is_direct_preview_video(selected):
            return web.FileResponse(
                selected,
                headers={"Content-Type": _preview_content_type_for_path(selected)},
            )
        proxy_path, content_type = _ensure_preview_proxy(
            selected, include_audio=want_audio
        )
        if proxy_path and content_type:
            return web.FileResponse(proxy_path, headers={"Content-Type": content_type})

    if selected.lower().endswith(".mov"):
        preview = _ensure_preview_proxy_webm(selected, include_audio=want_audio)
        if preview:
            return web.FileResponse(preview, headers={"Content-Type": "video/webm"})
        return web.Response(text="MOV preview conversion failed", status=415)
    return web.FileResponse(selected)


def _get_video_path_from_request(request) -> str:
    path = request.query.get("file") or request.query.get("path") or request.query.get(
        "folder"
    )
    index_str = (
        request.query.get("index") or request.query.get("video_index") or "0"
    )
    include_subdir = (
        (request.query.get("include_subdir") or "true").lower()
        == "true"
    )

    if not path:
        return ""

    path = path.strip().strip('"').strip("'")
    if os.path.isfile(path):
        return path

    if not os.path.isdir(path):
        return ""

    try:
        index = int(index_str)
    except ValueError:
        index = 0

    video_paths = get_video_paths(path, include_subdir)
    if not video_paths:
        return ""

    idx = index % len(video_paths)
    return video_paths[idx]


def _is_supported_video_path(path: str) -> bool:
    if not isinstance(path, str) or not path:
        return False
    if not os.path.isfile(path):
        return False
    ext = os.path.splitext(path)[1].lower()
    return ext in VALID_VIDEO_EXTENSIONS


async def _extract_video_frame_png(
    path: str,
    timestamp_s: float,
) -> bytes:
    try:
        def _extract():
            with av.open(path) as container:
                stream = container.streams.video[0]
                ts = max(0.0, float(timestamp_s or 0.0))
                if ts > 0:
                    container.seek(int(ts * 1_000_000))
                best = None
                for frame in container.decode(stream):
                    best = frame
                    ft = frame.time
                    if ft is not None and ft >= ts:
                        break
                if best is None:
                    return b""
                img = best.to_image()
                buf = BytesIO()
                img.save(buf, format="PNG")
                return buf.getvalue()
        return await asyncio.to_thread(_extract)
    except Exception:
        return b""


@PromptServer.instance.routes.get("/1hew/video_frame_from_folder")
async def video_frame_from_folder(request):
    selected = _get_video_path_from_request(request)
    if not _is_supported_video_path(selected):
        return web.Response(status=404)

    try:
        timestamp_s = float(request.query.get("t") or 0.0)
    except ValueError:
        timestamp_s = 0.0

    png = await _extract_video_frame_png(selected, timestamp_s)
    if not png:
        return web.Response(status=404)
    return web.Response(body=png, headers={"Content-Type": "image/png"})


@PromptServer.instance.routes.get("/1hew/download_video_by_path")
async def download_video_by_path(request):
    path = (request.query.get("path") or "").strip().strip('"').strip("'")
    if not _is_supported_video_path(path):
        return web.Response(status=404)

    name = _safe_filename(path)
    headers = {
        "Content-Disposition": f'attachment; filename="{name}"',
    }
    return web.FileResponse(path, headers=headers)


@PromptServer.instance.routes.get("/1hew/video_frame_by_path")
async def video_frame_by_path(request):
    path = (request.query.get("path") or "").strip().strip('"').strip("'")
    if not _is_supported_video_path(path):
        return web.Response(status=404)

    try:
        timestamp_s = float(request.query.get("t") or 0.0)
    except ValueError:
        timestamp_s = 0.0

    png = await _extract_video_frame_png(path, timestamp_s)
    if not png:
        return web.Response(status=404)
    return web.Response(body=png, headers={"Content-Type": "image/png"})
