from __future__ import annotations

import asyncio
import os
from typing import Optional

import folder_paths
from comfy.cli_args import args
from comfy_api.input import VideoInput
from comfy_api.latest import io, ui
from comfy_api.util import VideoContainer


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
            ],
            outputs=[],
            hidden=[io.Hidden.prompt, io.Hidden.extra_pnginfo],
            is_output_node=True,
        )

    @classmethod
    async def execute(
        cls,
        video: Optional[VideoInput],
        filename_prefix: str,
        save_output: bool,
    ) -> io.NodeOutput:
        if video is None:
            return io.NodeOutput()

        output_dir = folder_paths.get_output_directory()
        output_type = io.FolderType.output
        if not save_output:
            output_dir = folder_paths.get_temp_directory()
            output_type = io.FolderType.temp

        width, height = video.get_dimensions()
        (
            full_output_folder,
            filename,
            counter,
            subfolder,
            filename_prefix,
        ) = folder_paths.get_save_image_path(
            filename_prefix, output_dir, width, height
        )

        saved_metadata = None
        if not args.disable_metadata:
            metadata: dict[str, object] = {}
            if cls.hidden.extra_pnginfo is not None:
                metadata.update(cls.hidden.extra_pnginfo)
            if cls.hidden.prompt is not None:
                metadata["prompt"] = cls.hidden.prompt
            if len(metadata) > 0:
                saved_metadata = metadata

        format = "auto"
        codec = "auto"
        extension = VideoContainer.get_extension(format)
        file = f"{filename}_{counter:05}_.{extension}"

        path = os.path.join(full_output_folder, file)
        await asyncio.to_thread(video.save_to, path, format=format, codec=codec, metadata=saved_metadata)

        return io.NodeOutput(ui=ui.PreviewVideo([ui.SavedResult(file, subfolder, output_type)])
        )
