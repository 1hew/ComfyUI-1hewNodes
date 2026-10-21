from __future__ import annotations

import hashlib
import os
from typing import Optional

import torch
from comfy_api.latest import io

from ...utils._video_io import (
    VideoFromFile,
    apply_frame_subset,
    force_frame_rate,
    get_video_paths,
    resolve_video_path,
)

try:
    from comfy.utils import ProgressBar
except Exception:
    ProgressBar = None




def _get_filename_stem(path: str) -> str:
    base_name = os.path.basename(str(path or "").strip())
    stem, _ext = os.path.splitext(base_name)
    return stem or "video"


class LoadVideoToImage(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        format_options = ["n", "2n+1", "4n+1", "6n+1", "8n+1"]
        return io.Schema(
            node_id="1hew_LoadVideoToImage",
            display_name="Load Video to Image",
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
                io.Image.Output(display_name="image"),
                io.Audio.Output(display_name="audio"),
                io.Float.Output(display_name="fps"),
                io.Int.Output(display_name="frame_count"),
                io.String.Output(display_name="filename"),
            ],
        )

    @classmethod
    async def execute(
        cls,
        file: str,
        fps: float,
        frame_limit: int,
        start_skip: int,
        end_skip: int,
        format: str,
        video_index: int,
        include_subdir: bool,
    ) -> io.NodeOutput:
        file = (file or "").strip().strip('"').strip("'")
        video_paths = get_video_paths(file, include_subdir)
        count = len(video_paths)

        if count == 0:
            return io.NodeOutput(None, None, 0.0, 0, "")

        idx = video_index % count
        selected = video_paths[idx]
        filename = _get_filename_stem(selected)

        video_helper = VideoFromFile(selected)
        components = video_helper.get_components()

        images = components.images
        audio = components.audio
        source_fps = float(components.frame_rate or 0.0)
        if images is not None:
            images, audio, fps = cls.apply_video_settings(
                images=images,
                audio=audio,
                source_fps=source_fps,
                fps=fps,
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
            audio,
            fps,
            frame_count,
            filename,
        )

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
        # 1. Apply Skip (Subset of original frames)
        # Note: We do NOT apply frame_limit here, as it should apply to the final output
        images = apply_frame_subset(
            images=images,
            start_skip=int(start_skip or 0),
            end_skip=int(end_skip or 0),
            frame_limit=0, # Limit applied last
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

        # 2. Force Frame Rate (Resampling)
        # We want to maintain the duration of this subset while resampling to target fps.
        images = force_frame_rate(
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


    @classmethod
    def IS_CHANGED(cls, file, include_subdir, **kwargs):
        file = resolve_video_path(file)
        if os.path.isfile(file):
            try:
                mtime = os.path.getmtime(file)
                return hashlib.sha256(f"{file}:{mtime}".encode()).hexdigest()
            except OSError:
                return float("nan")

        if not os.path.isdir(file):
            return float("nan")

        video_paths = get_video_paths(file, include_subdir)
        m = hashlib.sha256()
        for video_path in video_paths:
            try:
                mtime = os.path.getmtime(video_path)
                m.update(f"{video_path}:{mtime}".encode())
            except OSError:
                continue

        return m.hexdigest()
