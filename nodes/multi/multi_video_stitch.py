"""Multi Video Stitch：把多段视频按统一尺寸首尾拼接为一段视频。

行为对齐 "Multi Image Batch" 的 fit / pad_color 语义，并新增
"get_video_size" 参考视频尺寸输入（类似 "Image Resize Universal" 的
"get_image_size"），因此不同分辨率、不同帧率的视频可以直接拼接。
"""

from __future__ import annotations

import os
import re
import shutil
import time
import uuid
from fractions import Fraction
from typing import Any, Optional

import torch
import torch.nn.functional as F
from comfy_api.latest import io

from ...utils._image_ops import (
    ensure_channels,
    pad_to_rgb,
    parse_pad_color,
)


class MultiVideoStitch(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="1hew_MultiVideoStitch",
            display_name="Multi Video Stitch",
            category="1hewNodes/multi",
            inputs=[
                io.String.Input(
                    "fps",
                    default="video_1",
                    tooltip='输出帧率。默认 "video_1" 沿用第一段视频的帧率；可填 video_2 / v2 / v_2 / video 2… 取对应视频的帧率，也可直接填数字（如 8、29.5）。',
                ),
                io.Combo.Input(
                    "fit",
                    options=["crop", "pad", "stretch"],
                    default="pad",
                    tooltip="尺寸适配模式：crop 缩放覆盖后居中裁切；pad 缩放适配后居中补边；stretch 直接拉伸。",
                ),
                io.String.Input(
                    "pad_color",
                    default="1.0",
                    tooltip="pad 模式的补边颜色，支持颜色名/HEX/RGB 以及 edge/average/extend/mirror 策略。",
                ),
                io.Video.Input(
                    "video_1",
                    optional=True,
                    tooltip="第一段视频；后续片段通过动态端口 video_2、video_3… 接入。",
                ),
                io.Video.Input(
                    "get_video_size",
                    optional=True,
                    tooltip="可选的尺寸参考视频；连接后以它的宽高作为拼接目标尺寸，未连接时以第一段视频为准。",
                ),
            ],
            outputs=[io.Video.Output(display_name="video")],
        )

    @classmethod
    async def execute(cls, fps, fit, pad_color, **kwargs) -> io.NodeOutput:
        videos = cls._collect_ordered_videos(kwargs)
        if not videos:
            return io.NodeOutput(None)

        decoded: list[dict[str, Any]] = []
        for index, video in videos:
            item = cls._decode_video(video)
            if item is not None:
                item["index"] = index
                decoded.append(item)
        if not decoded:
            return io.NodeOutput(None)

        target_h, target_w = cls._resolve_target_size(
            kwargs.get("get_video_size"), decoded
        )
        target_fps = cls._resolve_target_fps(fps, decoded)
        pad_rgb = parse_pad_color(pad_color)
        target_channels = (
            4
            if any(int(item["images"].shape[3]) == 4 for item in decoded)
            else 3
        )

        aligned_frames: list[torch.Tensor] = []
        segment_audios: list[Any] = []
        segment_durations: list[float] = []
        for item in decoded:
            frames = cls._resample_frames(
                item["images"], float(item["frame_rate"] or 0.0), target_fps
            )
            frames = cls._fit_frames(
                frames, target_h, target_w, fit, pad_rgb, target_channels
            )
            aligned_frames.append(frames)
            segment_audios.append(item["audio"])
            segment_durations.append(int(frames.shape[0]) / target_fps)

        combined = torch.cat(aligned_frames, dim=0)
        combined = torch.clamp(combined, 0.0, 1.0).to(torch.float32)
        audio = cls._concat_audio(segment_audios, segment_durations)

        video = cls._encode_video(combined, audio, target_fps)
        return io.NodeOutput(video)

    # ------------------------------------------------------------------
    # 输入采集与解码
    # ------------------------------------------------------------------

    @classmethod
    def _collect_ordered_videos(cls, kwargs: dict[str, Any]) -> list[tuple[int, Any]]:
        ordered: list[tuple[int, Any]] = []
        for key, value in kwargs.items():
            if not isinstance(key, str) or not key.startswith("video_"):
                continue
            suffix = key[len("video_") :]
            if not suffix.isdigit():
                continue
            if value is None:
                continue
            ordered.append((int(suffix), value))
        ordered.sort(key=lambda item: item[0])
        return ordered

    @classmethod
    def _decode_video(cls, video: Any) -> Optional[dict[str, Any]]:
        get_components = getattr(video, "get_components", None)
        if not callable(get_components):
            return None
        try:
            components = get_components()
        except Exception:
            return None
        if components is None:
            return None

        images = getattr(components, "images", None)
        if images is None or int(images.shape[0]) == 0:
            return None

        images = images.detach().to(torch.float32).clamp(0.0, 1.0).cpu()
        audio = getattr(components, "audio", None)
        frame_rate = getattr(components, "frame_rate", 0.0)
        return {"images": images, "audio": audio, "frame_rate": frame_rate}

    @classmethod
    def _probe_dimensions(cls, video: Any) -> Optional[tuple[int, int]]:
        get_dimensions = getattr(video, "get_dimensions", None)
        if callable(get_dimensions):
            try:
                dims = get_dimensions()
            except Exception:
                dims = None
            if dims and len(dims) >= 2:
                width, height = int(dims[0]), int(dims[1])
                if width > 0 and height > 0:
                    return height, width
        item = cls._decode_video(video)
        if item is None:
            return None
        images = item["images"]
        return int(images.shape[1]), int(images.shape[2])

    @classmethod
    def _resolve_target_size(
        cls, size_ref: Any, decoded: list[dict[str, Any]]
    ) -> tuple[int, int]:
        if size_ref is not None:
            dims = cls._probe_dimensions(size_ref)
            if dims is not None:
                return dims
        images = decoded[0]["images"]
        return int(images.shape[1]), int(images.shape[2])

    # ------------------------------------------------------------------
    # 帧率与尺寸统一
    # ------------------------------------------------------------------

    @classmethod
    def _resolve_target_fps(cls, fps: Any, decoded: list[dict[str, Any]]) -> float:
        text = "" if fps is None else str(fps).strip().lower()

        reference = cls._parse_video_reference(text)
        if reference is not None:
            for item in decoded:
                if int(item.get("index", 0)) == reference:
                    value = float(item.get("frame_rate") or 0.0)
                    if 0.0 < value < 1000.0:
                        return value
                    break

        manual = cls._parse_manual_fps(text)
        if manual is not None:
            return manual

        return cls._first_valid_fps(decoded)

    @staticmethod
    def _parse_video_reference(text: str) -> Optional[int]:
        match = re.fullmatch(r"(?:video|v)[\s_-]*([0-9]+)", text)
        if not match:
            return None
        index = int(match.group(1))
        return index if index >= 1 else None

    @staticmethod
    def _first_valid_fps(decoded: list[dict[str, Any]]) -> float:
        for item in decoded:
            value = float(item.get("frame_rate") or 0.0)
            if 0.0 < value < 1000.0:
                return value
        return 8.0

    @staticmethod
    def _parse_manual_fps(fps: Any) -> Optional[float]:
        if fps is None:
            return None
        text = str(fps).strip().lower()
        if text in ("", "auto", "a", "first"):
            return None
        try:
            value = float(text)
        except (TypeError, ValueError):
            return None
        if value <= 0.0 or value >= 1000.0:
            return None
        return value

    @classmethod
    def _resample_frames(
        cls, frames: torch.Tensor, source_fps: float, target_fps: float
    ) -> torch.Tensor:
        total = int(frames.shape[0])
        if total <= 1 or source_fps <= 0.0 or target_fps <= 0.0:
            return frames
        if abs(source_fps - target_fps) < 1e-6:
            return frames
        duration = total / source_fps
        target_total = max(int(round(duration * target_fps)), 1)
        if target_total == total:
            return frames
        indices = torch.linspace(0, total - 1, target_total).round().long()
        indices = indices.clamp(0, total - 1)
        return frames.index_select(0, indices)

    @classmethod
    def _fit_frames(
        cls,
        frames: torch.Tensor,
        target_h: int,
        target_w: int,
        fit: str,
        pad_rgb: Any,
        target_channels: int,
    ) -> torch.Tensor:
        frames = ensure_channels(frames, target_channels)
        h = int(frames.shape[1])
        w = int(frames.shape[2])

        if fit == "stretch":
            return cls._resize_to(frames, target_h, target_w)
        if fit == "crop":
            scale = max(target_w / max(w, 1), target_h / max(h, 1))
            new_w = max(int(round(w * scale)), 1)
            new_h = max(int(round(h * scale)), 1)
            resized = cls._resize_to(frames, new_h, new_w)
            return cls._center_crop(resized, target_h, target_w)
        scale = min(target_w / max(w, 1), target_h / max(h, 1))
        new_w = max(int(round(w * scale)), 1)
        new_h = max(int(round(h * scale)), 1)
        resized = cls._resize_to(frames, new_h, new_w)
        return pad_to_rgb(resized, target_h, target_w, pad_rgb)

    @staticmethod
    def _resize_to(img: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
        if int(img.shape[1]) == target_h and int(img.shape[2]) == target_w:
            return img
        img_nchw = img.permute(0, 3, 1, 2)
        out = F.interpolate(
            img_nchw,
            size=(target_h, target_w),
            mode="bicubic",
            align_corners=False,
        ).permute(0, 2, 3, 1)
        return out

    @staticmethod
    def _center_crop(img: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
        h = int(img.shape[1])
        w = int(img.shape[2])
        top = max((h - target_h) // 2, 0)
        left = max((w - target_w) // 2, 0)
        top_end = min(top + target_h, h)
        left_end = min(left + target_w, w)
        cropped = img[:, top:top_end, left:left_end, :]
        if int(cropped.shape[1]) != target_h or int(cropped.shape[2]) != target_w:
            cropped = pad_to_rgb(
                cropped, target_h, target_w, (0.0, 0.0, 0.0)
            )
        return cropped




    # ------------------------------------------------------------------
    # 音轨拼接
    # ------------------------------------------------------------------

    @classmethod
    def _concat_audio(cls, audios: list[Any], durations: list[float]) -> Optional[dict]:
        waveforms: list[Optional[torch.Tensor]] = []
        common_sr = 0
        common_channels = 1
        for audio in audios:
            waveform = cls._extract_waveform(audio)
            sample_rate = cls._extract_sample_rate(audio)
            if waveform is not None:
                common_channels = max(common_channels, int(waveform.shape[0]))
                if common_sr <= 0 and sample_rate > 0:
                    common_sr = int(sample_rate)
            waveforms.append(waveform)

        if all(waveform is None for waveform in waveforms):
            return None
        if common_sr <= 0:
            common_sr = 44100
        common_channels = max(1, min(common_channels, 2))

        parts: list[torch.Tensor] = []
        for waveform, duration in zip(waveforms, durations):
            target = max(int(round(max(float(duration), 0.0) * common_sr)), 1)
            if waveform is None:
                parts.append(torch.zeros(common_channels, target, dtype=torch.float32))
                continue
            parts.append(cls._resample_audio(waveform, target, common_channels))

        stacked = torch.cat(parts, dim=1)
        return {"waveform": stacked.unsqueeze(0), "sample_rate": common_sr}

    @staticmethod
    def _extract_waveform(audio: Any) -> Optional[torch.Tensor]:
        if audio is None:
            return None
        if isinstance(audio, dict):
            waveform = audio.get("waveform")
        else:
            waveform = getattr(audio, "waveform", None)
        if waveform is None or not torch.is_tensor(waveform):
            return None
        waveform = waveform.detach().to(torch.float32).cpu()
        if waveform.dim() == 3:
            waveform = waveform[0] if int(waveform.shape[0]) > 0 else waveform.squeeze(0)
        elif waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        if waveform.dim() != 2 or int(waveform.shape[1]) == 0:
            return None
        return waveform

    @staticmethod
    def _extract_sample_rate(audio: Any) -> int:
        if audio is None:
            return 0
        if isinstance(audio, dict):
            value = audio.get("sample_rate")
        else:
            value = getattr(audio, "sample_rate", None)
        try:
            return int(value)
        except (TypeError, ValueError):
            return 0

    @staticmethod
    def _resample_audio(
        waveform: torch.Tensor, target: int, channels: int
    ) -> torch.Tensor:
        current_channels = int(waveform.shape[0])
        if current_channels < channels:
            if current_channels == 1:
                waveform = waveform.repeat(channels, 1)
            else:
                pad = torch.zeros(channels - current_channels, waveform.shape[1])
                waveform = torch.cat([waveform, pad], dim=0)
        elif current_channels > channels:
            waveform = waveform[:channels]
        if int(waveform.shape[1]) != target:
            if int(waveform.shape[1]) <= 0:
                return torch.zeros(channels, target, dtype=torch.float32)
            waveform = F.interpolate(
                waveform.unsqueeze(0),
                size=target,
                mode="linear",
                align_corners=False,
            )[0]
        return waveform

    # ------------------------------------------------------------------
    # 编码输出
    # ------------------------------------------------------------------

    @classmethod
    def _encode_video(
        cls, images: torch.Tensor, audio: Optional[dict], fps: float
    ) -> Any:
        import av
        import folder_paths

        if not (0.0 < fps < 1000.0):
            fps = 1.0

        images = images.detach().cpu()
        total, height, width, channels = images.shape
        has_alpha = int(channels) == 4
        extension = ".webm" if has_alpha else ".mp4"
        video_codec = "libvpx-vp9" if has_alpha else "libx264"
        audio_codec = "libopus" if has_alpha else "aac"

        images_np = (images * 255.0).clamp(0, 255).byte().numpy()
        input_pix_fmt = "rgba" if has_alpha else "rgb24"

        audio_np = None
        audio_rate = 44100
        audio_channels = 2
        audio_layout = "stereo"
        waveform = cls._extract_waveform(audio)
        if waveform is not None:
            audio_rate = int(cls._extract_sample_rate(audio) or 44100)
            audio_channels = max(1, min(int(waveform.shape[0]), 2))
            audio_layout = "stereo" if audio_channels >= 2 else "mono"
            audio_np = waveform[:audio_channels].numpy()

        temp_dir = os.path.join(folder_paths.get_temp_directory(), "1hew_video_stitch")
        os.makedirs(temp_dir, exist_ok=True)
        stamp = uuid.uuid4().hex[:8]
        out_path = os.path.join(
            temp_dir, "MultiVideoStitch_" + str(time.time_ns()) + "_" + stamp + extension
        )
        tmp_path = out_path + ".tmp" + extension

        with av.open(tmp_path, "w") as output:
            out_video = output.add_stream(video_codec)
            if has_alpha:
                out_video.pix_fmt = "yuva420p"
                out_video.options = {"auto-alt-ref": "0", "b": "0", "crf": "30"}
            else:
                out_video.pix_fmt = "yuv420p"
                out_video.options = {"preset": "fast", "crf": "23"}
            out_video.width = int(width)
            out_video.height = int(height)
            out_video.rate = Fraction(fps).limit_denominator(10001)

            out_audio = None
            if audio_np is not None:
                out_audio = output.add_stream(
                    audio_codec, rate=audio_rate, layout=audio_layout
                )
                if audio_codec == "aac":
                    out_audio.bit_rate = 128000

            for index in range(int(total)):
                frame = av.VideoFrame.from_ndarray(images_np[index], format=input_pix_fmt)
                frame = frame.reformat(format=out_video.pix_fmt)
                for packet in out_video.encode(frame):
                    output.mux(packet)
            for packet in out_video.encode(None):
                output.mux(packet)

            if out_audio is not None and audio_np is not None:
                video_duration = total / fps if fps > 0 else 0.0
                max_samples = (
                    int(video_duration * audio_rate)
                    if video_duration > 0
                    else int(audio_np.shape[1])
                )
                total_samples = min(int(audio_np.shape[1]), max_samples)
                pts = 0
                while pts < total_samples:
                    count = min(1024, total_samples - pts)
                    chunk = audio_np[:audio_channels, pts : pts + count].copy()
                    chunk = chunk.astype("float32")
                    audio_frame = av.AudioFrame.from_ndarray(
                        chunk, format="fltp", layout=audio_layout
                    )
                    audio_frame.sample_rate = audio_rate
                    audio_frame.pts = pts
                    for packet in out_audio.encode(audio_frame):
                        output.mux(packet)
                    pts += count
                for packet in out_audio.encode(None):
                    output.mux(packet)

        try:
            os.replace(tmp_path, out_path)
        except OSError:
            shutil.move(tmp_path, out_path)

        return cls._wrap_video(out_path)

    @staticmethod
    def _wrap_video(path: str) -> Any:
        try:
            from comfy_api.input_impl.video_types import VideoFromFile

            return VideoFromFile(path)
        except Exception:
            pass
        from ...utils._video_io import VideoFromFile as LocalVideoFromFile

        return LocalVideoFromFile(path)

    # ------------------------------------------------------------------
    # 颜色解析（与 Multi Image Batch 保持一致）
    # ------------------------------------------------------------------
