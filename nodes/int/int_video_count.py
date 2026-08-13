from typing import Any

from comfy_api.latest import io

from ...utils import make_ui_text


class IntVideoCount(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="1hew_IntVideoCount",
            display_name="Int Video Count",
            category="1hewNodes/int",
            inputs=[
                io.Video.Input("video_1", optional=True),
            ],
            outputs=[io.Int.Output(display_name="int")],
        )

    @classmethod
    async def execute(
        cls,
        video_1=None,
        **kwargs,
    ) -> io.NodeOutput:
        count = 0
        videos = cls._collect_ordered_videos(video_1, kwargs)
        for _, video in videos:
            count += cls._count_connected_video(video)

        return io.NodeOutput(
            int(count),
            ui=make_ui_text(str(int(count))),
        )

    @classmethod
    def _collect_ordered_videos(
        cls,
        video_1: Any,
        kwargs: dict[str, Any],
    ) -> list[tuple[int, Any]]:
        videos: list[tuple[int, Any]] = [(1, video_1)]
        for key, value in kwargs.items():
            if not isinstance(key, str) or not key.startswith("video_"):
                continue
            suffix = key[len("video_") :]
            if not suffix.isdigit():
                continue
            index = int(suffix)
            if index == 1:
                continue
            videos.append((index, value))
        videos.sort(key=lambda item: item[0])
        return videos

    @staticmethod
    def _count_connected_video(video) -> int:
        return 1 if video is not None else 0
