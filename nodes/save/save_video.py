from __future__ import annotations

import asyncio
import os
from typing import Optional

import folder_paths
from comfy.cli_args import args
from comfy_api.input import VideoInput
from comfy_api.latest import io, ui
from comfy_api.util import VideoCodec, VideoContainer


class SaveVideo(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="1hew_SaveVideo",
            display_name="Save Video",
            category="1hewNodes/save",
            description="保存视频到输出；支持空值输入。",
            inputs=[
                io.Video.Input("video", optional=True, tooltip="要保存的视频；为空时节点直接通过。"),
                io.String.Input("filename_prefix", default="video/ComfyUI", tooltip="保存文件前缀；支持格式占位符（如 %date:yyyy-MM-dd%）。"),
                io.Combo.Input("format", options=VideoContainer.as_input(), default="auto", tooltip="容器格式；auto 将自动选择合适格式。"),
                io.Combo.Input("codec", options=VideoCodec.as_input(), default="auto", tooltip="编码器；auto 将根据容器与内容自动选择。"),
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
        format: str,
        codec: str,
    ) -> io.NodeOutput:
        if video is None:
            return io.NodeOutput()

        width, height = video.get_dimensions()
        (
            full_output_folder,
            filename,
            counter,
            subfolder,
            filename_prefix,
        ) = folder_paths.get_save_image_path(
            filename_prefix, folder_paths.get_output_directory(), width, height
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

        extension = VideoContainer.get_extension(format)
        file = f"{filename}_{counter:05}_.{extension}"

        path = os.path.join(full_output_folder, file)
        await asyncio.to_thread(video.save_to, path, format=format, codec=codec, metadata=saved_metadata)

        return io.NodeOutput(ui=ui.PreviewVideo([ui.SavedResult(file, subfolder, io.FolderType.output)])
        )