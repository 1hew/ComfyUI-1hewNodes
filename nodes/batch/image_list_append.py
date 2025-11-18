import asyncio
from comfy_api.latest import io
import torch


class ImageListAppend(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="1hew_ImageListAppend",
            display_name="Image List Append",
            category="1hewNodes/batch",
            inputs=[
                io.Image.Input("image_1", optional=True),
                io.Image.Input("image_2", optional=True),
            ],
            outputs=[
                io.Image.Output(display_name="image_list"),
            ],
        )

    @classmethod
    async def execute(cls, **kwargs) -> io.NodeOutput:
        try:
            ordered = []
            for k in kwargs.keys():
                if k.startswith("image_"):
                    suf = k[len("image_") :]
                    if suf.isdigit():
                        ordered.append((int(suf), k))
            ordered.sort(key=lambda x: x[0])

            async def _norm(x):
                def _do():
                    if isinstance(x, list):
                        return x
                    return [x]
                return await asyncio.to_thread(_do)

            parts = []
            for _, key in ordered:
                val = kwargs.get(key)
                if val is None:
                    continue
                lst = await _norm(val)
                parts.extend(lst)
            return io.NodeOutput(parts)
        except Exception:
            return io.NodeOutput([])