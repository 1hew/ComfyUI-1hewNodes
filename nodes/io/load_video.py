from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import subprocess
import time
from typing import Optional

from aiohttp import web
import av
import folder_paths
import numpy as np
import torch
from comfy_api.latest import io
from server import PromptServer


VALID_VIDEO_EXTENSIONS = {
    ".avi",
    ".mkv",
    ".mov",
    ".mp4",
    ".webm",
    ".wmv",
    ".flv",
    ".gif",
    ".m4v",
    ".qt",
}


class VideoComponents:
    def __init__(self, images, audio, frame_rate):
        self.images = images
        self.audio = audio
        self.frame_rate = frame_rate


class VideoFromFile:
    def __init__(self, path: str):
        self.path = path

    @staticmethod
    def _resolve_ffmpeg_exe() -> str:
        env_value = os.environ.get("FFMPEG_PATH") or os.environ.get("FFMPEG")
        if env_value and os.path.isfile(env_value):
            return env_value

        candidates: list[str] = []
        base_path = getattr(folder_paths, "base_path", None)
        if isinstance(base_path, str) and base_path:
            base_dir = os.path.abspath(base_path)
            candidates.extend(
                [
                    os.path.join(base_dir, "ffmpeg.exe"),
                    os.path.join(base_dir, "ffmpeg", "bin", "ffmpeg.exe"),
                    os.path.join(base_dir, "python_embeded", "ffmpeg.exe"),
                    os.path.join(base_dir, "python_embeded", "Scripts", "ffmpeg.exe"),
                ]
            )
            parent_dir = os.path.abspath(os.path.join(base_dir, os.pardir))
            candidates.extend(
                [
                    os.path.join(parent_dir, "ffmpeg.exe"),
                    os.path.join(parent_dir, "ffmpeg", "bin", "ffmpeg.exe"),
                    os.path.join(parent_dir, "python_embeded", "ffmpeg.exe"),
                    os.path.join(
                        parent_dir, "python_embeded", "Scripts", "ffmpeg.exe"
                    ),
                ]
            )

        for candidate in candidates:
            if os.path.isfile(candidate):
                return candidate

        return "ffmpeg"

    @staticmethod
    def _resolve_ffprobe_exe(ffmpeg_exe: str) -> str:
        env_value = os.environ.get("FFPROBE_PATH") or os.environ.get("FFPROBE")
        if env_value and os.path.isfile(env_value):
            return env_value

        if ffmpeg_exe and os.path.isabs(ffmpeg_exe):
            ffprobe_exe = os.path.join(os.path.dirname(ffmpeg_exe), "ffprobe.exe")
            if os.path.isfile(ffprobe_exe):
                return ffprobe_exe

        return "ffprobe"

    @staticmethod
    def _parse_rate(rate: str) -> float:
        rate = (rate or "").strip()
        if not rate or rate == "0/0":
            return 0.0
        if "/" in rate:
            num_str, den_str = rate.split("/", 1)
            try:
                num = float(num_str)
                den = float(den_str)
            except ValueError:
                return 0.0
            if den == 0:
                return 0.0
            return num / den
        try:
            return float(rate)
        except ValueError:
            return 0.0

    @classmethod
    def _probe_video_info_with_ffprobe(cls, path: str) -> dict:
        ffmpeg_exe = cls._resolve_ffmpeg_exe()
        ffprobe_exe = cls._resolve_ffprobe_exe(ffmpeg_exe)
        command = [
            ffprobe_exe,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,avg_frame_rate,pix_fmt",
            "-of",
            "json",
            path,
        ]
        try:
            res = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=False,
            )
        except FileNotFoundError:
            return {}

        if res.returncode != 0:
            return {}

        try:
            payload = json.loads(res.stdout or "{}")
        except json.JSONDecodeError:
            return {}

        streams = payload.get("streams") or []
        if not streams:
            return {}

        stream = streams[0] or {}
        width = int(stream.get("width") or 0)
        height = int(stream.get("height") or 0)
        fps = cls._parse_rate(stream.get("avg_frame_rate") or "")
        pix_fmt = (stream.get("pix_fmt") or "").lower()
        return {"width": width, "height": height, "fps": fps, "pix_fmt": pix_fmt}

    @staticmethod
    def _prefer_alpha_for_stream(stream) -> bool:
        try:
            ctx = getattr(stream, "codec_context", None)
            fmt = getattr(ctx, "format", None)
            fmt_name = getattr(fmt, "name", "") or ""
            pix_fmt = getattr(ctx, "pix_fmt", "") or ""
            probe = f"{fmt_name} {pix_fmt}".lower()
            return any(
                token in probe
                for token in ("yuva", "rgba", "bgra", "argb", "abgr")
            )
        except Exception:
            return False

    @staticmethod
    def _decode_frames_with_ffmpeg(
        path: str,
        width: int,
        height: int,
        pix_fmt: str,
    ) -> list[np.ndarray]:
        if int(width) <= 0 or int(height) <= 0:
            info = VideoFromFile._probe_video_info_with_ffprobe(path)
            width = int(info.get("width") or 0)
            height = int(info.get("height") or 0)

        channels = 4 if pix_fmt == "rgba" else 3
        frame_size = int(width) * int(height) * channels
        if frame_size <= 0:
            return []

        ffmpeg_exe = VideoFromFile._resolve_ffmpeg_exe()
        command = [
            ffmpeg_exe,
            "-v",
            "error",
            "-i",
            path,
            "-an",
            "-f",
            "rawvideo",
            "-pix_fmt",
            pix_fmt,
            "-",
        ]

        try:
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                "ffmpeg executable is required to decode this video. "
                f"Resolved ffmpeg path: {ffmpeg_exe}"
            ) from exc

        frames: list[np.ndarray] = []
        try:
            stdout = process.stdout
            if stdout is None:
                return []

            while True:
                buf = stdout.read(frame_size)
                if len(buf) != frame_size:
                    break
                frame = np.frombuffer(buf, dtype=np.uint8).reshape(
                    height, width, channels
                )
                frames.append(frame.copy())

            return frames
        finally:
            stderr_text = ""
            try:
                process.wait(timeout=30)
                if process.stderr is not None:
                    try:
                        stderr_text = process.stderr.read().decode("utf-8", "replace")
                    except Exception:
                        stderr_text = ""
                if process.stdout:
                    process.stdout.close()
                if process.stderr:
                    process.stderr.close()
            finally:
                if process.returncode not in (0, None) and not frames:
                    msg = stderr_text.strip() or "ffmpeg decoding failed"
                    raise RuntimeError(msg)

    def get_dimensions(self):
        try:
            container = av.open(self.path)
            if len(container.streams.video) > 0:
                stream = container.streams.video[0]
                width = int(stream.width or 0)
                height = int(stream.height or 0)
                if width > 0 and height > 0:
                    return width, height
        except Exception:
            pass
        info = self._probe_video_info_with_ffprobe(self.path)
        width = int(info.get("width") or 0)
        height = int(info.get("height") or 0)
        return width, height

    def save_to(self, path, format="auto", codec="auto", metadata=None):
        shutil.copy2(self.path, path)

    def get_components(self) -> Optional[VideoComponents]:
        path = self.path
        if not os.path.exists(path):
            raise FileNotFoundError(f"Video file not found: {path}")

        try:
            container = av.open(path)
        except Exception as exc:
            info = self._probe_video_info_with_ffprobe(path)
            width = int(info.get("width") or 0)
            height = int(info.get("height") or 0)
            fps = float(info.get("fps") or 0.0)

            frames = []
            for pix_fmt in ("rgba", "rgb24"):
                try:
                    frames = self._decode_frames_with_ffmpeg(
                        path=path,
                        width=width,
                        height=height,
                        pix_fmt=pix_fmt,
                    )
                except Exception:
                    frames = []
                if frames:
                    break

            if not frames:
                raise RuntimeError(f"Error opening video {path}: {exc}") from exc

            video_tensor = torch.from_numpy(np.stack(frames)).float() / 255.0
            return VideoComponents(images=video_tensor, audio=None, frame_rate=fps)

        frames: list[np.ndarray] = []
        video_info: dict = {}

        if len(container.streams.video) > 0:
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"
            prefer_alpha = self._prefer_alpha_for_stream(stream)

            try:
                fps = float(stream.average_rate)
            except Exception:
                fps = 0.0

            try:
                source_fps = float(stream.average_rate)
            except Exception:
                source_fps = 0.0

            video_info = {
                "source_fps": source_fps,
                "source_frame_count": int(stream.frames),
                "source_duration": (
                    float(stream.duration * stream.time_base)
                    if stream.duration
                    else 0.0
                ),
                "fps": fps,
                "width": stream.width,
                "height": stream.height,
            }
            if (
                float(video_info.get("fps") or 0.0) <= 0.0
                or int(video_info.get("width") or 0) <= 0
                or int(video_info.get("height") or 0) <= 0
            ):
                probed = self._probe_video_info_with_ffprobe(path)
                if float(video_info.get("fps") or 0.0) <= 0.0:
                    probed_fps = float(probed.get("fps") or 0.0)
                    if probed_fps > 0.0:
                        video_info["fps"] = probed_fps
                if int(video_info.get("width") or 0) <= 0:
                    video_info["width"] = int(probed.get("width") or 0)
                if int(video_info.get("height") or 0) <= 0:
                    video_info["height"] = int(probed.get("height") or 0)

            try:
                for frame in container.decode(stream):
                    if not frames:
                        try:
                            if prefer_alpha:
                                img_np = frame.to_ndarray(format="rgba")
                                desired_channels = 4
                            else:
                                img_np = frame.to_ndarray(format="rgb24")
                                desired_channels = 3
                        except Exception:
                            if prefer_alpha:
                                img_np = frame.to_ndarray(format="rgb24")
                                desired_channels = 3
                            else:
                                img_np = frame.to_ndarray(format="rgba")
                                desired_channels = 4
                    else:
                        desired_channels = frames[0].shape[-1]
                        preferred = "rgba" if desired_channels == 4 else "rgb24"
                        fallback = "rgb24" if preferred == "rgba" else "rgba"
                        try:
                            img_np = frame.to_ndarray(format=preferred)
                        except Exception:
                            img_np = frame.to_ndarray(format=fallback)

                    if getattr(img_np, "ndim", 0) == 2:
                        img_np = img_np[..., None]

                    if getattr(img_np, "ndim", 0) != 3:
                        continue

                    if desired_channels == 4 and img_np.shape[-1] == 3:
                        alpha = np.full(
                            (img_np.shape[0], img_np.shape[1], 1),
                            255,
                            dtype=img_np.dtype,
                        )
                        img_np = np.concatenate([img_np, alpha], axis=-1)
                    elif desired_channels == 3 and img_np.shape[-1] >= 4:
                        img_np = img_np[..., :3]
                    elif desired_channels == 4 and img_np.shape[-1] > 4:
                        img_np = img_np[..., :4]
                    elif desired_channels == 3 and img_np.shape[-1] == 1:
                        img_np = np.repeat(img_np, 3, axis=-1)
                    elif desired_channels == 4 and img_np.shape[-1] == 1:
                        rgb = np.repeat(img_np, 3, axis=-1)
                        alpha = np.full(
                            (img_np.shape[0], img_np.shape[1], 1),
                            255,
                            dtype=img_np.dtype,
                        )
                        img_np = np.concatenate([rgb, alpha], axis=-1)

                    frames.append(img_np)
            except Exception as exc:
                pass

            if prefer_alpha and frames and frames[0].shape[-1] != 4:
                ff_frames = self._decode_frames_with_ffmpeg(
                    path=path,
                    width=int(stream.width or 0),
                    height=int(stream.height or 0),
                    pix_fmt="rgba",
                )
                if ff_frames:
                    frames = ff_frames

            if not frames:
                ff_pix_fmt = "rgba" if prefer_alpha else "rgb24"
                try:
                    frames = self._decode_frames_with_ffmpeg(
                        path=path,
                        width=int(stream.width or 0),
                        height=int(stream.height or 0),
                        pix_fmt=ff_pix_fmt,
                    )
                except Exception:
                    frames = []

        if not frames:
            info = self._probe_video_info_with_ffprobe(path)
            width = int(info.get("width") or 0)
            height = int(info.get("height") or 0)
            for pix_fmt in ("rgba", "rgb24"):
                try:
                    frames = self._decode_frames_with_ffmpeg(
                        path=path,
                        width=width,
                        height=height,
                        pix_fmt=pix_fmt,
                    )
                except Exception:
                    frames = []
                if frames:
                    break

        if not frames:
            raise RuntimeError("Video decoding yielded an empty frame set")

        video_tensor = torch.from_numpy(np.stack(frames)).float() / 255.0

        audio = None
        if len(container.streams.audio) > 0:
            stream = container.streams.audio[0]
            try:
                container.seek(0)
                audio_frames = []
                for frame in container.decode(stream):
                    data = frame.to_ndarray()

                    if data.dtype != np.float32:
                        data = data.astype(np.float32)

                    if getattr(stream.format, "is_planar", False):
                        pass
                    else:
                        if len(data.shape) > 1:
                            data = data.T
                        else:
                            data = data[None, :]

                    audio_frames.append(data)

                if audio_frames:
                    audio_data = np.concatenate(audio_frames, axis=1)
                    waveform = torch.from_numpy(audio_data).float().unsqueeze(0)
                    audio = {"waveform": waveform, "sample_rate": stream.rate}
            except Exception as exc:
                print(f"Error decoding audio {path}: {exc}")

        return VideoComponents(
            images=video_tensor,
            audio=audio,
            frame_rate=video_info.get("fps", 0.0),
        )


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
        return self._video.save_to(
            path, format=format, codec=codec, metadata=metadata
        )

    def get_components(self) -> Optional[VideoComponents]:
        components = self._video.get_components()
        if components is None:
            return None

        images = components.images
        source_fps = float(components.frame_rate or 0.0)
        if images is not None:
            images, fps = LoadVideo.apply_video_settings(
                images=images,
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

        return VideoComponents(images=images, audio=components.audio, frame_rate=fps)


class LoadVideo(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        format_options = ["n", "2n+1", "4n+1", "6n+1", "8n+1"]
        return io.Schema(
            node_id="1hew_LoadVideo",
            display_name="Load Video",
            category="1hewNodes/io",
            inputs=[
                io.String.Input("path", default=""),
                io.Int.Input("frame_limit", default=0, min=0, max=100000, step=1),
                io.Float.Input("fps", default=0.0, min=0.0, max=120.0, step=1.0),
                io.Int.Input("start_skip", default=0, min=0, max=100000, step=1),
                io.Int.Input("end_skip", default=0, min=0, max=100000, step=1),
                io.Combo.Input("format", options=format_options, default="4n+1"),
                io.Int.Input("video_index", default=0, min=-8192, max=8192, step=1),
                io.Boolean.Input("include_subdir", default=False),
            ],
            outputs=[io.Video.Output(display_name="video")],
        )

    @classmethod
    async def execute(
        cls,
        path: str,
        start_skip: int,
        end_skip: int,
        fps: float,
        frame_limit: int,
        format: str,
        video_index: int,
        include_subdir: bool,
    ) -> io.NodeOutput:
        path = (path or "").strip().strip('"').strip("'")
        video_paths = cls.get_video_paths(path, include_subdir)
        count = len(video_paths)

        if count == 0:
            return io.NodeOutput(None)

        idx = video_index % count
        selected = video_paths[idx]

        return io.NodeOutput(
            VideoFromFileWithSettings(
                path=selected,
                start_skip=start_skip,
                end_skip=end_skip,
                fps=fps,
                frame_limit=frame_limit,
                format=format,
            )
        )

    @staticmethod
    def apply_video_settings(
        images: torch.Tensor,
        source_fps: float,
        fps: float,
        frame_limit: int,
        start_skip: int,
        end_skip: int,
        format: str,
    ) -> tuple[torch.Tensor, float]:
        images = LoadVideo._apply_frame_subset(
            images=images,
            start_skip=int(start_skip or 0),
            end_skip=int(end_skip or 0),
            frame_limit=0,
        )

        images = LoadVideo._force_frame_rate(
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
        return images, fps

    @staticmethod
    def _force_frame_rate(
        images: torch.Tensor, source_fps: float, target_fps: float
    ) -> torch.Tensor:
        if (
            images is None
            or float(source_fps or 0.0) <= 0.0
            or float(target_fps or 0.0) <= 0.0
        ):
            return images

        frame_count = int(images.shape[0])
        if frame_count <= 0:
            return images

        duration = float(frame_count - 1) / float(source_fps)
        out_count = int(math.floor(duration * float(target_fps) + 1e-9)) + 1
        out_count = max(out_count, 1)

        idx = torch.arange(out_count, device=images.device, dtype=torch.float32)
        idx = idx * float(source_fps) / float(target_fps)
        idx = idx.round().to(dtype=torch.long)
        idx = torch.clamp(idx, 0, frame_count - 1)

        return images.index_select(0, idx)

    @staticmethod
    def _apply_frame_subset(
        images: torch.Tensor,
        start_skip: int,
        end_skip: int,
        frame_limit: int,
    ) -> torch.Tensor:
        if images is None:
            return images

        if start_skip > 0:
            images = images[start_skip:]

        if end_skip > 0:
            keep_count = int(images.shape[0]) - int(end_skip)
            keep_count = max(0, keep_count)
            images = images[:keep_count]

        if frame_limit > 0:
            images = images[:frame_limit]

        return images

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


    @staticmethod
    def get_video_paths(path: str, include_subdir: bool) -> list[str]:
        path = (path or "").strip().strip('"').strip("'")
        if os.path.isfile(path):
            ext = os.path.splitext(path)[1].lower()
            return [path] if ext in VALID_VIDEO_EXTENSIONS else []

        if not os.path.isdir(path):
            return []

        video_paths: list[str] = []

        if include_subdir:
            for root, _, files in os.walk(path):
                for file in files:
                    if os.path.splitext(file)[1].lower() in VALID_VIDEO_EXTENSIONS:
                        video_paths.append(os.path.join(root, file))
        else:
            for file in os.listdir(path):
                file_path = os.path.join(path, file)
                if (
                    os.path.isfile(file_path)
                    and os.path.splitext(file)[1].lower() in VALID_VIDEO_EXTENSIONS
                ):
                    video_paths.append(file_path)

        video_paths.sort(key=lambda x: x.lower())
        return video_paths

    @classmethod
    def IS_CHANGED(cls, path, include_subdir, **kwargs):
        path = (path or "").strip().strip('"').strip("'")
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
        if os.path.isfile(path):
            try:
                mtime = os.path.getmtime(path)
                return hashlib.sha256(
                    f"{path}:{mtime}:{settings_key}".encode()
                ).hexdigest()
            except OSError:
                return float("nan")

        if not os.path.isdir(path):
            return float("nan")

        video_paths = cls.get_video_paths(path, include_subdir)
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


def _has_alpha_pix_fmt(pix_fmt: str) -> bool:
    pix_fmt = (pix_fmt or "").lower()
    return any(token in pix_fmt for token in ("yuva", "rgba", "bgra", "argb", "abgr"))


def _ensure_preview_proxy_webm(source_path: str) -> str:
    source_path = os.path.abspath(source_path)
    try:
        mtime = os.path.getmtime(source_path)
    except OSError:
        return ""

    key = hashlib.sha256(f"{source_path}:{mtime}".encode()).hexdigest()
    cache_dir = _preview_cache_dir()
    os.makedirs(cache_dir, exist_ok=True)
    out_path = os.path.join(cache_dir, f"{key}.webm")

    if os.path.isfile(out_path):
        try:
            if os.path.getmtime(out_path) >= mtime:
                return out_path
        except OSError:
            pass

    ffmpeg_exe = VideoFromFile._resolve_ffmpeg_exe()
    info = VideoFromFile._probe_video_info_with_ffprobe(source_path)
    pix_fmt = info.get("pix_fmt") or ""
    output_pix_fmt = "yuva420p" if _has_alpha_pix_fmt(pix_fmt) else "yuv420p"

    tmp_path = os.path.join(
        cache_dir, f"{key}.{time.time_ns()}.tmp.webm"
    )

    command = [
        ffmpeg_exe,
        "-y",
        "-v",
        "error",
        "-i",
        source_path,
        "-an",
        "-vf",
        "scale=trunc(iw/2)*2:trunc(ih/2)*2",
        "-c:v",
        "libvpx-vp9",
        "-pix_fmt",
        output_pix_fmt,
        "-auto-alt-ref",
        "0",
        "-b:v",
        "0",
        "-crf",
        "33",
        "-deadline",
        "realtime",
        "-cpu-used",
        "5",
        tmp_path,
    ]

    try:
        res = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
        )
    except Exception:
        return ""

    if res.returncode != 0:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except OSError:
            pass
        return ""

    try:
        os.replace(tmp_path, out_path)
    except OSError:
        return ""

    return out_path


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
    path = request.query.get("path") or request.query.get("folder")
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

        video_paths = LoadVideo.get_video_paths(path, include_subdir)
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
        container = av.open(selected)
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
    path = request.query.get("path") or request.query.get("folder")
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

    if os.path.isfile(path):
        if path.lower().endswith(".mov"):
            preview = _ensure_preview_proxy_webm(path)
            if preview:
                return web.FileResponse(
                    preview, headers={"Content-Type": "video/webm"}
                )
            return web.FileResponse(path, headers={"Content-Type": "video/mp4"})
        return web.FileResponse(path)

    if not os.path.isdir(path):
        return web.Response(status=404)

    try:
        index = int(index_str)
    except ValueError:
        index = 0

    video_paths = LoadVideo.get_video_paths(path, include_subdir)
    if not video_paths:
        return web.Response(status=404)

    idx = index % len(video_paths)
    selected = video_paths[idx]

    if selected.lower().endswith(".mov"):
        preview = _ensure_preview_proxy_webm(selected)
        if preview:
            return web.FileResponse(preview, headers={"Content-Type": "video/webm"})
        return web.FileResponse(selected, headers={"Content-Type": "video/mp4"})
    return web.FileResponse(selected)
