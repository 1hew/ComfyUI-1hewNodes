from __future__ import annotations

import asyncio
import os
from typing import Optional

import folder_paths
import torch
from comfy_api.latest import io, ui

_PATH_LOCK: Optional[asyncio.Lock] = None


class SaveImage(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="1hew_SaveImage",
            display_name="Save Image",
            category="1hewNodes/io",
            inputs=[
                io.Image.Input("image", optional=True),
                io.String.Input("filename_prefix", default="image/ComfyUI"),
                io.Boolean.Input("save_output", default=True),
                io.Boolean.Input("save_metadata", default=False),
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
        image: Optional[torch.Tensor],
        filename_prefix: str,
        save_output: bool,
        save_metadata: bool,
    ) -> io.NodeOutput:
        if image is None:
            return io.NodeOutput()

        folder_type = io.FolderType.output if save_output else io.FolderType.temp

        global _PATH_LOCK
        if _PATH_LOCK is None:
            _PATH_LOCK = asyncio.Lock()

        class _Hidden:
            prompt = None
            extra_pnginfo = None

        class _NoMetadataNode:
            hidden = _Hidden()

        effective_cls = cls if save_metadata else _NoMetadataNode

        async with _PATH_LOCK:
            results = ui.ImageSaveHelper.save_images(
                image,
                filename_prefix=filename_prefix,
                folder_type=folder_type,
                cls=effective_cls,
                compress_level=4,
            )

        file_paths = []
        if results:
            base_dir = (
                folder_paths.get_output_directory()
                if folder_type == io.FolderType.output
                else folder_paths.get_temp_directory()
            )
            for result in results:
                file_paths.append(
                    os.path.abspath(
                        os.path.join(base_dir, result.subfolder, result.filename)
                    )
                )

        return io.NodeOutput("\n".join(file_paths), ui=ui.SavedImages(results))
