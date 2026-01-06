from __future__ import annotations

import asyncio
import json
import os
from typing import Optional

import folder_paths
from comfy.cli_args import args
from comfy_api.input import VideoInput
from comfy_api.latest import io, ui
from comfy_api.util import VideoContainer

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


class SaveVideo(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="1hew_SaveVideo",
            display_name="Save Video",
            category="1hewNodes/io",
            inputs=[
                io.Video.Input("video", optional=True, tooltip="要保存的视频；为空时节点直接通过。"),
                io.String.Input("filename_prefix", default="video/ComfyUI", tooltip="保存文件前缀；支持格式占位符（如 %date:yyyy-MM-dd%）。"),
                io.Boolean.Input("save_output", default=True, tooltip="是否保存到输出目录；若为 False 则保存到临时目录。"),
                io.Boolean.Input("save_metadata", default=False, tooltip="是否写入 prompt/workflow 元数据。"),
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
        filename_prefix: str,
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

        # Check for alpha channel
        has_alpha = False
        if hasattr(video, "path") and os.path.exists(video.path):
            has_alpha = await cls.check_has_alpha(video.path)

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

            format = "auto"
            codec = "auto"
            extension = VideoContainer.get_extension(format)

            # Try to use source extension if available
            if hasattr(video, "path"):
                _, source_ext = os.path.splitext(video.path)
                if source_ext:
                    extension = source_ext.lstrip(".")

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

        command = [
            "ffmpeg",
            "-y",
            "-i",
            input_path,
            "-map",
            "0",
            "-c",
            "copy",
            "-map_metadata",
            "-1",
        ]

        if metadata_comment:
            command.extend(["-metadata", f"comment={metadata_comment}"])

        command.append(tmp_path)

        process = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await process.communicate()

        if process.returncode != 0:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise Exception(f"FFmpeg remux failed: {stderr.decode()}")

        os.replace(tmp_path, input_path)

    @staticmethod
    async def check_has_alpha(path: str) -> bool:
        try:
            command = [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=pix_fmt",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                path,
            ]
            process = await asyncio.create_subprocess_exec(
                *command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await process.communicate()
            pix_fmt = stdout.decode().strip()
            # Common alpha pixel formats
            alpha_formats = [
                "yuva",
                "rgba",
                "argb",
                "abgr",
                "bgra",
                "gbrap",
                "ya8",
                "ya16",
                "ayuv",
            ]
            return any(fmt in pix_fmt for fmt in alpha_formats)
        except Exception:
            return False

    @staticmethod
    async def generate_preview(
        input_path: str,
        output_path: str,
        strip_metadata: bool = False,
    ):
        try:
            command = [
                "ffmpeg",
                "-y",
                "-i",
                input_path,
                "-c:v",
                "libvpx-vp9",
                "-pix_fmt",
                "yuva420p",
                "-auto-alt-ref",
                "0",
                "-b:v",
                "0",
                "-crf",
                "30",
                "-c:a",
                "libopus",
            ]

            if strip_metadata:
                command.extend(["-map_metadata", "-1"])

            command.append(output_path)

            process = await asyncio.create_subprocess_exec(
                *command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            _, stderr = await process.communicate()
            
            if process.returncode != 0:
                print(f"Info: Preview generation skipped: {stderr.decode()}")
                
        except Exception as e:
            print(f"Info: Preview generation skipped: {e}")
