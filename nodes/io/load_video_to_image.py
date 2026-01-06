from __future__ import annotations

import hashlib
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
from comfy_api.latest import io

try:
    from comfy.utils import ProgressBar
except Exception:
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


def _new_progress_bar(total: int):
    if ProgressBar is None:
        return None
    if int(total or 0) <= 0:
        return None
    try:
        return ProgressBar(int(total))
    except Exception:
        return None


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
            "stream=width,height,avg_frame_rate,pix_fmt,nb_frames,duration",
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
        return {
            "width": width,
            "height": height,
            "fps": fps,
            "pix_fmt": pix_fmt,
            "nb_frames": nb_frames,
            "duration": duration,
        }

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
            return VideoComponents(images=video_tensor, audio=None, frame_rate=fps)

        frames: list[np.ndarray] = []
        video_info: dict = {}
        expected_video_frames = 0
        progress_bar = None

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
            except Exception as exc:
                pass
            if progress_bar is not None and frames_decoded > last_reported:
                delta = frames_decoded - last_reported
                try:
                    progress_bar.update(int(delta))
                except Exception:
                    pass

            if prefer_alpha and frames and frames[0].shape[-1] != 4:
                ff_frames = self._decode_frames_with_ffmpeg(
                    path=path,
                    width=int(stream.width or 0),
                    height=int(stream.height or 0),
                    pix_fmt="rgba",
                    expected_frames=expected_video_frames,
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
                    audio = {"waveform": waveform, "sample_rate": stream.rate}
            except Exception as exc:
                print(f"Error decoding audio {path}: {exc}")

        return VideoComponents(
            images=video_tensor,
            audio=audio,
            frame_rate=video_info.get("fps", 0.0),
        )


class LoadVideoToImage(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        format_options = ["n", "2n+1", "4n+1", "6n+1", "8n+1"]
        return io.Schema(
            node_id="1hew_LoadVideoToImage",
            display_name="Load Video to Image",
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
            outputs=[
                io.Image.Output(display_name="image"),
                io.Audio.Output(display_name="audio"),
                io.Float.Output(display_name="fps"),
                io.Int.Output(display_name="frame_count"),
            ],
        )

    @classmethod
    async def execute(
        cls,
        path: str,
        fps: float,
        frame_limit: int,
        start_skip: int,
        end_skip: int,
        format: str,
        video_index: int,
        include_subdir: bool,
    ) -> io.NodeOutput:
        path = (path or "").strip().strip('"').strip("'")
        video_paths = cls.get_video_paths(path, include_subdir)
        count = len(video_paths)

        if count == 0:
            return io.NodeOutput(None, None, 0.0, 0)

        idx = video_index % count
        selected = video_paths[idx]

        video_helper = VideoFromFile(selected)
        components = video_helper.get_components()

        images = components.images
        source_fps = float(components.frame_rate or 0.0)
        if images is not None:
            images, fps = cls.apply_video_settings(
                images=images,
                fps=fps,
                source_fps=source_fps,
                frame_limit=frame_limit,
                start_skip=start_skip,
                end_skip=end_skip,
                format=format,
            )
        else:
            fps = source_fps

        frame_count = images.shape[0] if images is not None else 0
        if frame_count == 0:
            images = None
            fps = 0.0

        return io.NodeOutput(
            images,
            components.audio,
            fps,
            frame_count,
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
        # 1. Apply Skip (Subset of original frames)
        # Note: We do NOT apply frame_limit here, as it should apply to the final output
        images = LoadVideoToImage._apply_frame_subset(
            images=images,
            start_skip=int(start_skip or 0),
            end_skip=int(end_skip or 0),
            frame_limit=0, # Limit applied last
        )

        # 2. Force Frame Rate (Resampling)
        # We want to maintain the duration of this subset while resampling to target fps.
        images = LoadVideoToImage._force_frame_rate(
            images=images,
            source_fps=source_fps,
            target_fps=float(fps or 0),
        )

        # 3. Apply Format Constraint
        images = LoadVideoToImage._apply_format_constraint(
            images=images,
            format=format or "4n+1",
        )

        # 4. Apply Frame Limit (Final count limit)
        limit = int(frame_limit or 0)
        if limit > 0 and images is not None:
            images = images[:limit]

        fps = LoadVideoToImage._compute_output_fps(
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

        idx = torch.arange(
            out_count, device=images.device, dtype=torch.float32
        )
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
        if f == "default":  # Legacy fallback
            return images

        step = 1
        mod = 0

        # Parse "Xn+Y"
        if "n" in f:
            parts = f.split("n")
            try:
                if parts[0]:
                    step = int(parts[0])
                if len(parts) > 1 and parts[1]:
                    # Handle "+1", "-1" etc.
                    mod = int(parts[1])
            except ValueError:
                return images

        count = images.shape[0]
        if count < mod:
            # If we strictly require mod, but have fewer frames, we can't satisfy it.
            # Return empty or all? Returning empty is safer for strict correctness,
            # but returning all might be better for UX?
            # User requirement: "output video frame count must be these".
            # If we can't satisfy, maybe empty is correct.
            # FIX: Return empty tensor instead of slicing [:0] which might preserve dimensions but be empty
            return images[:0]

        # Calculate max k such that step*k + mod <= count
        # step*k <= count - mod
        # k <= (count - mod) // step
        k = (count - mod) // step
        target_count = step * k + mod

        # Ensure target_count is at least 0 (should be given logic above)
        target_count = max(0, target_count)
        
        # If target_count is 0 but we have frames, and mod is 0 (e.g. 4n), we return 0 frames.
        # But if mod > 0 (e.g. 4n+1) and we have frames, target_count should be at least mod.
        
        if target_count == 0 and count > 0:
             # Case: count=2, format=4n+1 -> mod=1. k=(2-1)//4=0. target=0*4+1=1. Correct.
             # Case: count=3, format=4n -> mod=0. k=3//4=0. target=0. Correct.
             pass

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
        if os.path.isfile(path):
            try:
                mtime = os.path.getmtime(path)
                return hashlib.sha256(f"{path}:{mtime}".encode()).hexdigest()
            except OSError:
                return float("nan")

        if not os.path.isdir(path):
            return float("nan")

        video_paths = cls.get_video_paths(path, include_subdir)
        m = hashlib.sha256()
        for video_path in video_paths:
            try:
                mtime = os.path.getmtime(video_path)
                m.update(f"{video_path}:{mtime}".encode())
            except OSError:
                continue

        return m.hexdigest()
