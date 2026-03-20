import asyncio
from fractions import Fraction
import json
import os
import tempfile
from typing import Optional

import av
import folder_paths
from comfy.cli_args import args
import torch
import torch.nn.functional as functional
import torchaudio
from comfy_api.latest import io, ui

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


class SaveVideoByImage(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="1hew_SaveVideoByImage",
            display_name="Save Video by Image",
            category="1hewNodes/io",
            inputs=[
                io.Image.Input("image"),
                io.Audio.Input("audio", optional=True),
                io.Float.Input("fps", default=8.0, min=0.01, max=120.0, step=0.01),
                io.String.Input("filename", default="video/ComfyUI"),
                io.Boolean.Input(
                    "auto_increment",
                    default=True,
                    tooltip="为 True 时自动编号；为 False 时固定文件名并覆盖。",
                ),
                io.Boolean.Input("save_output", default=True),
                io.Boolean.Input("save_metadata", default=True),
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
        image: torch.Tensor,
        fps: float,
        filename: str,
        auto_increment: bool,
        save_output: bool,
        save_metadata: bool,
        audio: Optional[dict] = None,
    ) -> io.NodeOutput:
        images = image
        height, width, channels = images.shape[1:]
        has_alpha = channels == 4

        if height % 2 != 0 or width % 2 != 0:
            new_height = height - (height % 2)
            new_width = width - (width % 2)
            images = cls._resize_images(images, new_height, new_width)
            height, width = new_height, new_width

        output_dir = folder_paths.get_output_directory()
        output_type = io.FolderType.output
        if not save_output:
            output_dir = folder_paths.get_temp_directory()
            output_type = io.FolderType.temp

        global _PATH_LOCK
        if _PATH_LOCK is None:
            _PATH_LOCK = asyncio.Lock()

        raw_prefix = str(filename or "").strip()
        safe_filename_prefix = _sanitize_filename_prefix(filename)
        preview_file = ""
        preview_path = ""
        preview_subfolder = ""
        counter = 0

        async with _PATH_LOCK:
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

                if has_alpha and save_output:
                    output_file = f"{filename}_{counter:05}_.mov"
                    path = os.path.join(full_output_folder, output_file)

                    preview_dir = os.path.join(
                        folder_paths.get_temp_directory(),
                        subfolder,
                    )
                    os.makedirs(preview_dir, exist_ok=True)
                    preview_file = f"{filename}_{counter:05}_.webm"
                    preview_path = os.path.join(preview_dir, preview_file)
                    preview_subfolder = subfolder
                else:
                    extension = "webm" if has_alpha else "mp4"
                    output_file = f"{filename}_{counter:05}_.{extension}"
                    path = os.path.join(full_output_folder, output_file)
                    preview_file = output_file
                    preview_path = path
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
                if has_alpha and save_output:
                    output_file = f"{filename}.mov"
                    path = os.path.join(full_output_folder, output_file)
                    preview_subfolder = _relative_subfolder(path, output_dir)
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
                else:
                    extension = "webm" if has_alpha else "mp4"
                    output_file = f"{filename}.{extension}"
                    path = os.path.join(full_output_folder, output_file)
                    preview_file = output_file
                    preview_path = path
                    preview_subfolder = _relative_subfolder(path, output_dir)

            if not os.path.exists(path):
                with open(path, "wb"):
                    pass

            if preview_path and preview_path != path:
                if not os.path.exists(preview_path):
                    with open(preview_path, "wb"):
                        pass

        audio_path = None
        if audio is not None:
            try:
                waveform = audio.get("waveform")
                sample_rate = audio.get("sample_rate")
                if waveform is not None and sample_rate:
                    if waveform.dim() == 3:
                        waveform = waveform.squeeze(0)

                    audio_dir = folder_paths.get_temp_directory()
                    fd, audio_path = tempfile.mkstemp(
                        prefix=f"{filename}_{counter:05}_",
                        suffix="_audio.wav",
                        dir=audio_dir,
                    )
                    os.close(fd)
                    torchaudio.save(audio_path, waveform, sample_rate)
            except Exception as exc:
                print(
                    "Info: Audio processing skipped; continuing without audio. "
                    f"Details: {exc}"
                )
                audio_path = None

        try:
            frame_count = int(images.shape[0])
            encode_passes = 2 if (has_alpha and save_output) else 1
            progress_bar = _new_progress_bar(frame_count * encode_passes)
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
            if has_alpha and save_output:
                await cls.ffmpeg_encode(
                    images=images,
                    path=preview_path,
                    fps=fps,
                    audio_path=audio_path,
                    width=width,
                    height=height,
                    input_pix_fmt="rgba",
                    video_codec="libvpx-vp9",
                    output_pix_fmt="yuva420p",
                    video_args=["-auto-alt-ref", "0", "-b:v", "0", "-crf", "30"],
                    audio_codec="libopus",
                    progress_bar=progress_bar,
                    metadata_comment=metadata_comment,
                )
                await cls.ffmpeg_encode(
                    images=images,
                    path=path,
                    fps=fps,
                    audio_path=audio_path,
                    width=width,
                    height=height,
                    input_pix_fmt="rgba",
                    video_codec="prores_ks",
                    output_pix_fmt="yuva444p10le",
                    video_args=["-profile:v", "4"],
                    audio_codec="aac",
                    progress_bar=progress_bar,
                    metadata_comment=metadata_comment,
                )
            else:
                await cls.ffmpeg_encode(
                    images=images,
                    path=path,
                    fps=fps,
                    audio_path=audio_path,
                    width=width,
                    height=height,
                    input_pix_fmt="rgba" if has_alpha else "rgb24",
                    video_codec="libvpx-vp9" if has_alpha else "libx264",
                    output_pix_fmt="yuva420p" if has_alpha else "yuv420p",
                    video_args=(
                        ["-auto-alt-ref", "0", "-b:v", "0", "-crf", "30"]
                        if has_alpha
                        else [
                            "-preset",
                            "medium",
                            "-crf",
                            "23",
                            "-movflags",
                            "+faststart",
                        ]
                    ),
                    audio_codec="libopus" if has_alpha else "aac",
                    progress_bar=progress_bar,
                    metadata_comment=metadata_comment,
                )
        finally:
            if audio_path and os.path.exists(audio_path):
                await _remove_file_with_retries(audio_path)

        file_path = os.path.abspath(path)

        return io.NodeOutput(
            file_path,
            ui=ui.PreviewVideo(
                [
                    ui.SavedResult(
                        preview_file,
                        preview_subfolder,
                        (
                            io.FolderType.temp
                            if (has_alpha and save_output)
                            else output_type
                        ),
                    )
                ]
            )
        )

    @classmethod
    async def ffmpeg_encode(
        cls,
        images: torch.Tensor,
        path: str,
        fps: float,
        audio_path: Optional[str],
        width: int,
        height: int,
        input_pix_fmt: str,
        video_codec: str,
        output_pix_fmt: str,
        video_args: list[str],
        audio_codec: str,
        progress_bar: Optional[object] = None,
        metadata_comment: Optional[str] = None,
    ):
        def _parse_video_args(args_list: list[str]) -> tuple[dict, dict]:
            codec_opts: dict[str, str] = {}
            container_opts: dict[str, str] = {}
            i = 0
            while i < len(args_list):
                arg = args_list[i]
                if not arg.startswith("-"):
                    i += 1
                    continue
                key = arg.lstrip("-")
                value = ""
                if i + 1 < len(args_list) and not args_list[i + 1].startswith("-"):
                    value = args_list[i + 1]
                    i += 2
                else:
                    i += 1
                if key == "movflags":
                    container_opts["movflags"] = value
                elif key == "b:v":
                    codec_opts["b"] = value
                elif key == "profile:v":
                    codec_opts["profile"] = value
                else:
                    codec_opts[key] = value
            return codec_opts, container_opts

        def _encode():
            codec_opts, container_opts = _parse_video_args(video_args)

            with av.open(path, "w", options=container_opts) as output:
                out_video = output.add_stream(video_codec)
                out_video.width = width
                out_video.height = height
                out_video.pix_fmt = output_pix_fmt
                out_video.rate = Fraction(fps).limit_denominator(10001)
                out_video.options = codec_opts

                out_audio = None
                audio_container = None
                try:
                    if audio_path and os.path.isfile(audio_path):
                        try:
                            audio_container = av.open(audio_path)
                            if audio_container.streams.audio:
                                in_audio = audio_container.streams.audio[0]
                                out_audio = output.add_stream(
                                    audio_codec, rate=in_audio.rate or 44100,
                                )
                        except Exception:
                            if audio_container is not None:
                                audio_container.close()
                            audio_container = None

                    if metadata_comment:
                        output.metadata["comment"] = metadata_comment

                    av_input_fmt = "rgba" if input_pix_fmt == "rgba" else "rgb24"
                    total_frames = int(images.shape[0])
                    update_every = max(1, int(total_frames // 200) or 1)
                    last_reported = 0
                    frames_written = 0

                    for i in range(total_frames):
                        frame_np = (images[i] * 255).byte().cpu().numpy()
                        av_frame = av.VideoFrame.from_ndarray(frame_np, format=av_input_fmt)
                        av_frame = av_frame.reformat(format=output_pix_fmt)

                        for pkt in out_video.encode(av_frame):
                            output.mux(pkt)

                        frames_written += 1
                        if (
                            progress_bar is not None
                            and frames_written - last_reported >= update_every
                        ):
                            delta = frames_written - last_reported
                            try:
                                progress_bar.update(int(delta))
                            except Exception:
                                pass
                            last_reported = frames_written

                    for pkt in out_video.encode(None):
                        output.mux(pkt)

                    if progress_bar is not None and frames_written > last_reported:
                        delta = frames_written - last_reported
                        try:
                            progress_bar.update(int(delta))
                        except Exception:
                            pass

                    if out_audio and audio_container:
                        try:
                            in_audio_stream = audio_container.streams.audio[0]
                            video_duration = total_frames / fps if fps > 0 else 0.0
                            max_samples = (
                                int(video_duration * (in_audio_stream.rate or 44100))
                                if video_duration > 0
                                else 0
                            )
                            samples_encoded = 0
                            for packet in audio_container.demux(in_audio_stream):
                                for frame in packet.decode():
                                    if max_samples > 0 and samples_encoded >= max_samples:
                                        break
                                    frame.pts = None
                                    for pkt in out_audio.encode(frame):
                                        output.mux(pkt)
                                    samples_encoded += frame.samples
                                if max_samples > 0 and samples_encoded >= max_samples:
                                    break
                            for pkt in out_audio.encode(None):
                                output.mux(pkt)
                        except Exception:
                            pass
                finally:
                    if audio_container is not None:
                        audio_container.close()

        try:
            await asyncio.to_thread(_encode)
        except Exception as e:
            if os.path.exists(path):
                os.remove(path)
            raise e

    @staticmethod
    def _resize_images(images: torch.Tensor, height: int, width: int) -> torch.Tensor:
        images_nchw = images.permute(0, 3, 1, 2).to(torch.float32)
        resized = functional.interpolate(
            images_nchw,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )
        return resized.permute(0, 2, 3, 1).clamp(0.0, 1.0)
