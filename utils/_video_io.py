"""Shared video IO helpers for 1hewNodes.

VideoFromFile wraps a video file with ffmpeg/av probing and frame
extraction, shared by Load Video, Load Video To Image and Multi Video
Stitch. This is a library module, not a node: files under
nodes/<category>/ must define exactly one ComfyNode, so the wrapper lives
here instead of being copied into every node file.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
from typing import Optional

import av
import folder_paths
import numpy as np
import torch

try:
    from comfy.utils import ProgressBar
except Exception:  # pragma: no cover - older ComfyUI builds
    ProgressBar = None


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


def _new_progress_bar(total: int):
    if ProgressBar is None:
        return None
    if int(total or 0) <= 0:
        return None
    try:
        return ProgressBar(int(total))
    except Exception:
        return None


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
                    os.path.join(parent_dir, "python_embeded", "Scripts", "ffmpeg.exe"),
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
            "stream=width,height,avg_frame_rate,pix_fmt,nb_frames,duration:stream_tags=rotate:stream_side_data=rotation",
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
        try:
            nb_frames = int(stream.get("nb_frames") or 0)
        except Exception:
            nb_frames = 0
        try:
            duration = float(stream.get("duration") or 0.0)
        except Exception:
            duration = 0.0
        rotation = 0.0
        try:
            rotation = float((stream.get("tags") or {}).get("rotate") or 0.0)
        except Exception:
            rotation = 0.0
        for side_data in stream.get("side_data_list") or []:
            try:
                if "rotation" in side_data:
                    rotation = float(side_data["rotation"])
                    break
            except Exception:
                pass
        return {
            "width": width,
            "height": height,
            "fps": fps,
            "pix_fmt": pix_fmt,
            "nb_frames": nb_frames,
            "duration": duration,
            "rotation": rotation,
        }

    @staticmethod
    def _apply_video_rotation(image: np.ndarray, rotation: float) -> np.ndarray:
        quadrant = int(round(float(rotation or 0.0) / 90.0)) % 4
        if quadrant == 0:
            return image
        return np.rot90(image, k=(-quadrant) % 4).copy()

    @classmethod
    def _probe_audio_info_with_ffprobe(cls, path: str) -> dict:
        ffmpeg_exe = cls._resolve_ffmpeg_exe()
        ffprobe_exe = cls._resolve_ffprobe_exe(ffmpeg_exe)
        command = [
            ffprobe_exe,
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=channels,sample_rate",
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
        try:
            channels = int(stream.get("channels") or 0)
        except Exception:
            channels = 0
        try:
            sample_rate = int(stream.get("sample_rate") or 0)
        except Exception:
            sample_rate = 0
        return {"channels": channels, "sample_rate": sample_rate}

    @classmethod
    def _decode_audio_with_ffmpeg(cls, path: str) -> Optional[dict]:
        info = cls._probe_audio_info_with_ffprobe(path)
        channels = int(info.get("channels") or 0)
        sample_rate = int(info.get("sample_rate") or 0)
        if channels <= 0:
            channels = 2
        if sample_rate <= 0:
            sample_rate = 44100

        ffmpeg_exe = cls._resolve_ffmpeg_exe()
        command = [
            ffmpeg_exe,
            "-v",
            "error",
            "-i",
            path,
            "-vn",
            "-acodec",
            "pcm_f32le",
            "-f",
            "f32le",
            "-ac",
            str(channels),
            "-ar",
            str(sample_rate),
            "-",
        ]
        try:
            res = subprocess.run(
                command,
                capture_output=True,
                check=False,
            )
        except FileNotFoundError:
            return None

        if res.returncode != 0:
            return None

        raw = res.stdout or b""
        if not raw:
            return None

        audio_1d = np.frombuffer(raw, dtype=np.float32)
        if audio_1d.size <= 0:
            return None

        frame = int(channels)
        usable = (audio_1d.size // frame) * frame
        if usable <= 0:
            return None

        audio_1d = audio_1d[:usable]
        audio_2d = audio_1d.reshape(-1, frame).T
        waveform = torch.from_numpy(audio_2d.copy()).float().unsqueeze(0)
        return {"waveform": waveform, "sample_rate": sample_rate}

    @staticmethod
    def _pix_fmt_has_alpha(pix_fmt: str) -> bool:
        probe = (pix_fmt or "").lower()
        return any(
            token in probe
            for token in (
                "yuva",
                "rgba",
                "bgra",
                "argb",
                "abgr",
                "gbrap",
                "ya8",
                "ya16",
                "ayuv",
            )
        )

    @classmethod
    def _prefer_alpha_for_stream(cls, stream, probed_pix_fmt: str = "") -> bool:
        try:
            ctx = getattr(stream, "codec_context", None)
            fmt = getattr(ctx, "format", None)
            fmt_name = getattr(fmt, "name", "") or ""
            pix_fmt = getattr(ctx, "pix_fmt", "") or ""
            return cls._pix_fmt_has_alpha(f"{fmt_name} {pix_fmt} {probed_pix_fmt}")
        except Exception:
            return False

    @staticmethod
    def _decode_frames_with_ffmpeg(
        path: str,
        width: int,
        height: int,
        pix_fmt: str,
        expected_frames: int = 0,
        progress_bar: Optional[object] = None,
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

        if progress_bar is None and int(expected_frames or 0) > 0:
            progress_bar = _new_progress_bar(int(expected_frames))
        frames: list[np.ndarray] = []
        update_every = 0
        last_reported = 0
        frames_decoded = 0
        if progress_bar is not None and int(expected_frames or 0) > 0:
            update_every = max(1, int(int(expected_frames) // 200) or 1)
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
                if progress_bar is not None and update_every:
                    frames_decoded += 1
                    if frames_decoded - last_reported >= update_every:
                        delta = frames_decoded - last_reported
                        try:
                            progress_bar.update(int(delta))
                        except Exception:
                            pass
                        last_reported = frames_decoded

            return frames
        finally:
            if (
                progress_bar is not None
                and update_every
                and frames_decoded > last_reported
            ):
                delta = frames_decoded - last_reported
                try:
                    progress_bar.update(int(delta))
                except Exception:
                    pass
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
            expected_frames = int(info.get("nb_frames") or 0)
            if expected_frames <= 0:
                duration = float(info.get("duration") or 0.0)
                if duration > 0.0 and fps > 0.0:
                    expected_frames = int(round(duration * fps))

            frames = []
            for pix_fmt in ("rgba", "rgb24"):
                try:
                    frames = self._decode_frames_with_ffmpeg(
                        path=path,
                        width=width,
                        height=height,
                        pix_fmt=pix_fmt,
                        expected_frames=expected_frames,
                    )
                except Exception:
                    frames = []
                if frames:
                    break

            if not frames:
                raise RuntimeError(f"Error opening video {path}: {exc}") from exc

            video_tensor = torch.from_numpy(np.stack(frames)).float() / 255.0
            rotation = float(info.get("rotation") or 0.0)
            if rotation:
                frames = [
                    self._apply_video_rotation(frame, rotation) for frame in frames
                ]
                video_tensor = torch.from_numpy(np.stack(frames)).float() / 255.0
            return VideoComponents(images=video_tensor, audio=None, frame_rate=fps)

        frames: list[np.ndarray] = []
        video_info: dict = {}
        expected_video_frames = 0
        progress_bar = None

        if len(container.streams.video) > 0:
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"
            probed = self._probe_video_info_with_ffprobe(path)
            prefer_alpha = self._prefer_alpha_for_stream(
                stream,
                str(probed.get("pix_fmt") or ""),
            )
            rotation = float(probed.get("rotation") or 0.0)

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
            expected_video_frames = int(video_info.get("source_frame_count") or 0)
            if expected_video_frames <= 0:
                try:
                    duration = float(stream.duration * stream.time_base)
                except Exception:
                    duration = 0.0
                if duration > 0.0 and float(video_info.get("fps") or 0.0) > 0.0:
                    expected_video_frames = int(
                        round(duration * float(video_info["fps"]))
                    )
            progress_bar = _new_progress_bar(expected_video_frames)
            update_every = max(1, int(expected_video_frames // 200) or 1)
            last_reported = 0
            frames_decoded = 0
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

                    img_np = self._apply_video_rotation(img_np, rotation)

                    frames.append(img_np)
                    if progress_bar is not None:
                        frames_decoded += 1
                        if frames_decoded - last_reported >= update_every:
                            delta = frames_decoded - last_reported
                            try:
                                progress_bar.update(int(delta))
                            except Exception:
                                pass
                            last_reported = frames_decoded
            except Exception:
                pass
            if progress_bar is not None and frames_decoded > last_reported:
                delta = frames_decoded - last_reported
                try:
                    progress_bar.update(int(delta))
                except Exception:
                    pass

            if prefer_alpha and frames and frames[0].shape[-1] != 4:
                try:
                    ff_frames = self._decode_frames_with_ffmpeg(
                        path=path,
                        width=int(stream.width or 0),
                        height=int(stream.height or 0),
                        pix_fmt="rgba",
                        expected_frames=expected_video_frames,
                    )
                    if ff_frames:
                        frames = ff_frames
                except Exception:
                    pass

            if not frames:
                ff_pix_fmt = "rgba" if prefer_alpha else "rgb24"
                try:
                    frames = self._decode_frames_with_ffmpeg(
                        path=path,
                        width=int(stream.width or 0),
                        height=int(stream.height or 0),
                        pix_fmt=ff_pix_fmt,
                        expected_frames=expected_video_frames,
                    )
                except Exception:
                    frames = []

        if not frames:
            info = self._probe_video_info_with_ffprobe(path)
            width = int(info.get("width") or 0)
            height = int(info.get("height") or 0)
            fps = float(info.get("fps") or 0.0)
            expected_frames = int(info.get("nb_frames") or 0)
            if expected_frames <= 0:
                duration = float(info.get("duration") or 0.0)
                if duration > 0.0 and fps > 0.0:
                    expected_frames = int(round(duration * fps))
            for pix_fmt in ("rgba", "rgb24"):
                try:
                    frames = self._decode_frames_with_ffmpeg(
                        path=path,
                        width=width,
                        height=height,
                        pix_fmt=pix_fmt,
                        expected_frames=expected_frames,
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
                    sample_rate = int(getattr(stream, "rate", 0) or 0)
                    if sample_rate <= 0:
                        sample_rate = int(
                            self._probe_audio_info_with_ffprobe(path).get(
                                "sample_rate", 0
                            )
                            or 0
                        )
                    if sample_rate <= 0:
                        sample_rate = 44100
                    audio = {"waveform": waveform, "sample_rate": sample_rate}
            except Exception:
                audio = None

        if audio is None:
            audio = self._decode_audio_with_ffmpeg(path)

        return VideoComponents(
            images=video_tensor,
            audio=audio,
            frame_rate=video_info.get("fps", 0.0),
        )


def resolve_video_path(path: str) -> str:
    """Resolve absolute and ComfyUI input-directory-relative video paths."""
    raw = str(path or "").strip().strip('"').strip("'")
    if not raw:
        return ""

    normalized = raw.replace("\\", os.sep).replace("/", os.sep)
    if os.path.isabs(normalized) or os.path.splitdrive(normalized)[0]:
        return os.path.abspath(normalized)

    get_annotated = getattr(folder_paths, "get_annotated_filepath", None)
    if callable(get_annotated):
        try:
            resolved = get_annotated(raw, accept_missing=True)
        except TypeError:
            try:
                resolved = get_annotated(raw)
            except Exception:
                resolved = ""
        except Exception:
            resolved = ""
        if resolved:
            return os.path.abspath(str(resolved))

    get_input_dir = getattr(folder_paths, "get_input_directory", None)
    base_dir = (
        get_input_dir()
        if callable(get_input_dir)
        else folder_paths.get_temp_directory()
    )
    return os.path.abspath(os.path.join(base_dir, normalized))


def force_frame_rate(
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


def apply_frame_subset(
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


def get_video_paths(path: str, include_subdir: bool) -> list[str]:
    path = resolve_video_path(path)
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
