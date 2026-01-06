import asyncio
import os
from typing import Optional

import folder_paths
import torch
import torch.nn.functional as functional
import torchaudio
from comfy_api.latest import io, ui

try:
    from comfy.utils import ProgressBar
except Exception:
    ProgressBar = None

_PATH_LOCK: Optional[asyncio.Lock] = None


def _new_progress_bar(total: int):
    if ProgressBar is None:
        return None
    if int(total or 0) <= 0:
        return None
    try:
        return ProgressBar(int(total))
    except Exception:
        return None


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
                io.String.Input("filename_prefix", default="video/ComfyUI"),
                io.Boolean.Input("save_output", default=True),
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
        filename_prefix: str,
        save_output: bool,
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

        preview_file = ""
        preview_path = ""
        preview_subfolder = ""

        async with _PATH_LOCK:
            (
                full_output_folder,
                filename,
                counter,
                subfolder,
                filename_prefix,
            ) = folder_paths.get_save_image_path(
                filename_prefix, output_dir, width, height
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
                file = f"{filename}_{counter:05}_.{extension}"
                path = os.path.join(full_output_folder, file)
                preview_file = file
                preview_path = path
                preview_subfolder = subfolder

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

                    audio_file = f"{filename}_{counter:05}_audio.wav"
                    audio_path = os.path.join(
                        folder_paths.get_temp_directory(),
                        audio_file,
                    )
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
                )
        finally:
            if audio_path and os.path.exists(audio_path):
                os.remove(audio_path)

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
    ):
        ffmpeg_path = "ffmpeg"

        command = [
            ffmpeg_path,
            "-y",
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-s",
            f"{width}x{height}",
            "-pix_fmt",
            input_pix_fmt,
            "-r",
            str(fps),
            "-i",
            "-",
        ]

        if audio_path:
            command.extend(["-i", audio_path])

        command.extend(["-c:v", video_codec, "-pix_fmt", output_pix_fmt])
        command.extend(video_args)

        if audio_path:
            command.extend(["-c:a", audio_codec, "-shortest"])

        command.append(path)

        try:
            process = await asyncio.create_subprocess_exec(
                *command,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            async def feed_stdin():
                total_frames = int(images.shape[0])
                update_every = max(1, int(total_frames // 200) or 1)
                last_reported = 0
                frames_written = 0
                for i in range(total_frames):
                    frame = images[i]
                    frame = (frame * 255).byte().cpu().numpy()
                    try:
                        process.stdin.write(frame.tobytes())
                        await process.stdin.drain()
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
                    except (BrokenPipeError, OSError):
                        break
                if progress_bar is not None and frames_written > last_reported:
                    delta = frames_written - last_reported
                    try:
                        progress_bar.update(int(delta))
                    except Exception:
                        pass
                process.stdin.close()

            async def log_stderr():
                stderr_data = b""
                while True:
                    chunk = await process.stderr.read(4096)
                    if not chunk:
                        break
                    stderr_data += chunk
                return stderr_data

            _, stderr_output = await asyncio.gather(feed_stdin(), log_stderr())
            
            await process.wait()

            if process.returncode != 0:
                raise Exception(f"FFmpeg encoding failed: {stderr_output.decode()}")

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
