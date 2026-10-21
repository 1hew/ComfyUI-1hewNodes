# 保存路径说明：
# - 默认文件名 "audio/ComfyUI"（相对路径）会将音频保存到 ComfyUI 配置的 output 目录下。
# - output 目录由 folder_paths.get_output_directory() 决定：
#   * 无 --output-directory 参数 → 默认 <comfyui_root>/output
#   * 带 --output-directory <路径> 参数 → 自定义路径
# - 输入绝对路径时会跳过 output 目录，直接保存到指定位置。
from __future__ import annotations

import asyncio
import json
import os
from typing import Optional

import av
import folder_paths
import torch
from comfy.cli_args import args
from comfy_api.latest import io, ui

from ...utils._file_ops import (
    is_absolute_like,
    sanitize_filename_prefix,
    split_prefix_to_dir_and_stem,
)

_PATH_LOCK: Optional[asyncio.Lock] = None

# format -> (container format, codec, file extension)
_FORMAT_SPECS: dict[str, tuple[str, str, str]] = {
    "flac": ("flac", "flac", "flac"),
    "wav": ("wav", "pcm_s16le", "wav"),
    "mp3": ("mp3", "libmp3lame", "mp3"),
    "opus": ("opus", "libopus", "opus"),
}

# Lossy encoders only accept a fixed set of sample rates; anything else is
# resampled to the next supported rate before encoding.
_CODEC_RATES: dict[str, tuple[int, ...]] = {
    "mp3": (8000, 11025, 12000, 16000, 22050, 24000, 32000, 44100, 48000),
    "opus": (8000, 12000, 16000, 24000, 48000),
}


def _relative_subfolder(path_dir: str, base_dir: str) -> Optional[str]:
    """Subfolder relative to the served directory, or None when outside it."""
    try:
        rel = os.path.relpath(path_dir, base_dir)
    except ValueError:
        return None
    if rel in {".", ""}:
        return ""
    if rel == ".." or rel.startswith(".." + os.sep):
        return None
    return rel.replace(os.sep, "/")


def _allocate_target(
    base_dir: str,
    raw_prefix: str,
    safe_prefix: str,
    auto_increment: bool,
    extension: str,
) -> tuple[str, str, int, Optional[str]]:
    """Return (directory, stem, counter, subfolder) for the next save."""
    if auto_increment:
        if is_absolute_like(raw_prefix):
            dir_raw, stem = split_prefix_to_dir_and_stem(raw_prefix)
            directory = os.path.abspath(dir_raw) if dir_raw else os.path.abspath(base_dir)
            os.makedirs(directory, exist_ok=True)
            counter = 1
            while os.path.exists(
                os.path.join(directory, f"{stem}_{counter:05}_.{extension}")
            ):
                counter += 1
        else:
            (
                directory,
                stem,
                counter,
                subfolder,
                _resolved_prefix,
            ) = folder_paths.get_save_image_path(safe_prefix, base_dir)
            return directory, stem, counter, subfolder
        return directory, stem, counter, _relative_subfolder(directory, base_dir)

    if is_absolute_like(raw_prefix):
        dir_raw, stem = split_prefix_to_dir_and_stem(raw_prefix)
        directory = os.path.abspath(dir_raw) if dir_raw else os.path.abspath(base_dir)
    else:
        rel_dir_raw, stem = split_prefix_to_dir_and_stem(safe_prefix)
        rel_dir = rel_dir_raw.replace("/", os.sep) if rel_dir_raw else ""
        directory = os.path.abspath(os.path.join(base_dir, rel_dir))

    os.makedirs(directory, exist_ok=True)
    return directory, stem, 1, _relative_subfolder(directory, base_dir)


def _collect_metadata(cls: type) -> dict[str, str]:
    hidden = getattr(cls, "hidden", None)
    if hidden is None:
        return {}
    out: dict[str, str] = {}
    extra = getattr(hidden, "extra_pnginfo", None)
    prompt = getattr(hidden, "prompt", None)
    if isinstance(extra, dict):
        for key, value in extra.items():
            out[str(key)] = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    if prompt is not None:
        out["prompt"] = json.dumps(prompt, ensure_ascii=False, separators=(",", ":"))
    return out


def _prepare_waveform(waveform) -> Optional[torch.Tensor]:
    """Normalize AUDIO waveform to a contiguous [B, C, S] float32 tensor."""
    if waveform is None:
        return None
    if not torch.is_tensor(waveform):
        try:
            waveform = torch.as_tensor(waveform)
        except Exception:
            return None
    try:
        wf = waveform.detach().to(torch.float32).cpu()
    except Exception:
        return None
    if wf.ndim == 2:
        wf = wf.unsqueeze(0)
    if wf.ndim != 3:
        return None
    if int(wf.shape[2]) <= 0:
        return None
    return wf.contiguous()


def _layout_for(waveform: torch.Tensor) -> tuple[str, int]:
    channels = int(waveform.shape[0])
    if channels == 1:
        return "mono", 1
    if channels == 2:
        return "stereo", 2
    raise ValueError(
        f"Save Audio only supports mono/stereo audio, got {channels} channel(s)."
    )


def _target_sample_rate(fmt: str, sample_rate: int) -> int:
    supported = _CODEC_RATES.get(fmt)
    if not supported:
        return sample_rate
    if sample_rate in supported:
        return sample_rate
    for rate in supported:
        if rate >= sample_rate:
            return rate
    return supported[-1]


def _build_audio_frame(
    waveform: torch.Tensor, layout: str, sample_rate: int
) -> av.AudioFrame:
    data = waveform.movedim(0, 1).reshape(1, -1).contiguous().numpy()
    frame = av.AudioFrame.from_ndarray(data, format="flt", layout=layout)
    frame.sample_rate = int(sample_rate)
    frame.pts = 0
    return frame


def _encode_audio_file(
    path: str,
    waveform: torch.Tensor,
    sample_rate: int,
    fmt: str,
    metadata: dict[str, str],
) -> None:
    muxer, codec, _extension = _FORMAT_SPECS[fmt]
    layout, _channels = _layout_for(waveform)
    target_rate = _target_sample_rate(fmt, int(sample_rate))

    with av.open(path, "w", format=muxer) as container:
        stream = container.add_stream(codec, rate=target_rate, layout=layout)
        if codec == "libmp3lame":
            stream.bit_rate = 320_000
        elif codec == "libopus":
            stream.bit_rate = 192_000

        for key, value in metadata.items():
            try:
                container.metadata[str(key)] = str(value)
            except Exception:
                break

        frames = [_build_audio_frame(waveform, layout, sample_rate)]
        if target_rate != int(sample_rate):
            resampler = av.AudioResampler(
                format="flt", layout=layout, rate=target_rate
            )
            resampled = list(resampler.resample(frames[0]) or [])
            resampled.extend(resampler.resample(None) or [])
            frames = resampled
            for frame in frames:
                frame.sample_rate = target_rate

        for frame in frames:
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode(None):
            container.mux(packet)


def _remove_quietly(path: str) -> None:
    try:
        os.remove(path)
    except OSError:
        pass


class SaveAudio(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="1hew_SaveAudio",
            display_name="Save Audio",
            category="1hewNodes/io",
            inputs=[
                io.Audio.Input(
                    "audio",
                    optional=True,
                    tooltip="要保存的音频；为空时节点直接通过，不写任何文件。",
                ),
                io.String.Input(
                    "filename",
                    default="audio/ComfyUI",
                    tooltip="保存文件名或路径前缀；支持格式占位符（如 %date:yyyy-MM-dd%）。",
                ),
                io.Combo.Input(
                    "format",
                    options=["flac", "wav", "mp3", "opus"],
                    default="flac",
                    tooltip="保存格式；flac / wav 无损，mp3 / opus 在采样率不受支持时会重采样。",
                ),
                io.Boolean.Input(
                    "auto_increment",
                    default=True,
                    tooltip="为 True 时自动编号；为 False 时固定文件名并覆盖。",
                ),
                io.Boolean.Input(
                    "save_output",
                    default=True,
                    tooltip="是否保存到输出目录；若为 False 则保存到临时目录。",
                ),
                io.Boolean.Input(
                    "save_metadata",
                    default=True,
                    tooltip="是否写入 prompt/workflow 元数据。",
                ),
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
        audio: Optional[dict] = None,
        filename: str = "audio/ComfyUI",
        format: str = "flac",
        auto_increment: bool = True,
        save_output: bool = True,
        save_metadata: bool = True,
    ) -> io.NodeOutput:
        if audio is None:
            return cls._empty_output()

        waveform, sample_rate = cls._read_audio(audio)
        if waveform is None or int(waveform.shape[0]) <= 0:
            return cls._empty_output()

        fmt = str(format or "flac").lower()
        if fmt not in _FORMAT_SPECS:
            fmt = "flac"
        _muxer, _codec, extension = _FORMAT_SPECS[fmt]

        base_dir = (
            folder_paths.get_output_directory()
            if save_output
            else folder_paths.get_temp_directory()
        )
        folder_type = io.FolderType.output if save_output else io.FolderType.temp

        raw_prefix = str(filename or "").strip()
        safe_prefix = sanitize_filename_prefix(filename, "audio")
        metadata = (
            _collect_metadata(cls)
            if save_metadata and not args.disable_metadata
            else {}
        )

        global _PATH_LOCK
        if _PATH_LOCK is None:
            _PATH_LOCK = asyncio.Lock()

        file_paths: list[str] = []
        previews: list = []

        async with _PATH_LOCK:
            directory, stem, counter, subfolder = _allocate_target(
                base_dir, raw_prefix, safe_prefix, bool(auto_increment), extension
            )

            for batch_index in range(int(waveform.shape[0])):
                if auto_increment:
                    output_name = f"{stem}_{counter:05}_.{extension}"
                    counter += 1
                elif batch_index == 0:
                    output_name = f"{stem}.{extension}"
                else:
                    output_name = f"{stem}_{batch_index:03d}.{extension}"

                output_path = os.path.join(directory, output_name)
                try:
                    _encode_audio_file(
                        output_path, waveform[batch_index], sample_rate, fmt, metadata
                    )
                except Exception as exc:
                    _remove_quietly(output_path)
                    raise RuntimeError(
                        f"Save Audio failed to encode {output_name}: {exc}"
                    ) from exc

                file_paths.append(os.path.abspath(output_path))
                if subfolder is not None:
                    previews.append(
                        ui.SavedResult(output_name, subfolder, folder_type)
                    )

        preview_ui = None
        if previews and hasattr(ui, "SavedAudios"):
            preview_ui = ui.SavedAudios(previews)
        elif hasattr(ui, "PreviewAudio"):
            preview_ui = ui.PreviewAudio(audio)

        return io.NodeOutput("\n".join(file_paths), ui=preview_ui)

    @staticmethod
    def _empty_output() -> io.NodeOutput:
        """Successful no-op: write nothing, and clear any preview from a prior run.

        ComfyUI only sends the "executed" message when the node returns a
        non-empty UI payload, so an empty audio list is what tells the frontend
        that this run produced nothing.
        """
        saved_audios = getattr(ui, "SavedAudios", None)
        if saved_audios is None:
            return io.NodeOutput()
        return io.NodeOutput(ui=saved_audios([]))

    @staticmethod
    def _read_audio(audio) -> tuple[Optional[torch.Tensor], int]:
        if not isinstance(audio, dict):
            return None, 0
        waveform = _prepare_waveform(audio.get("waveform"))
        try:
            sample_rate = int(audio.get("sample_rate") or 0)
        except (TypeError, ValueError):
            sample_rate = 0
        if waveform is None or sample_rate <= 0:
            return None, 0
        return waveform, sample_rate
