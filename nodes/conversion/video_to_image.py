from __future__ import annotations

from comfy_api.latest import io


class VideoToImage(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="1hew_VideoToImage",
            display_name="Video to Image",
            category="1hewNodes/conversion",
            inputs=[
                io.Video.Input("video", optional=True),
            ],
            outputs=[
                io.Image.Output(display_name="image"),
                io.Audio.Output(display_name="audio"),
                io.Float.Output(display_name="fps"),
                io.Int.Output(display_name="frame_count"),
            ],
        )

    @classmethod
    async def execute(cls, video=None) -> io.NodeOutput:
        if video is None:
            return io.NodeOutput(None, None, 0.0, 0)

        get_components = getattr(video, "get_components", None)
        if not callable(get_components):
            raise TypeError("video must provide a get_components() method")

        components = get_components()
        if components is None:
            return io.NodeOutput(None, None, 0.0, 0)

        images = components.images
        audio = components.audio
        fps = float(components.frame_rate or 0.0)
        frame_count = int(images.shape[0]) if images is not None else 0

        if frame_count == 0:
            return io.NodeOutput(None, audio, 0.0, 0)

        return io.NodeOutput(images, audio, fps, frame_count)
