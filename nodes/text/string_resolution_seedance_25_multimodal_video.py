"""Seedance 2.5 多模态视频专用分辨率标签节点。

与 1hewOffice 的 RunningHub Seedance 2.5 多模态视频节点成对使用
（nodes/api/runninghub/runninghub_seedance_25_multimodal_video.py，
node_id 1hewOffice_RunningHubSeedance25MultimodalVideo）：按输入视频的像素面积推断出
该节点 resolution 支持的档位之一（480p / 720p / 1080p / 2k / 4k），
未连接视频时直接透传手动选择的标签。

目标面积取各档 16:9 标准尺寸的面积，与 Seedance 的常见输出规格一致：
480p = 854x480，720p = 1280x720，1080p = 1920x1080，
2k = 2560x1440，4k = 3840x2160。
用面积而不是单边判断，竖屏（如 1080x1920）同样会被正确判为 1080p。
"""

import math
from typing import Any, Optional

from comfy_api.latest import io


class StringResolutionSeedance25MultimodalVideo(io.ComfyNode):
    # 各档位目标面积取该档 16:9 标准尺寸的面积，与 API 节点的分辨率档位一一对应。
    RESOLUTION_OPTIONS = [
        ("480p", 854 * 480),
        ("720p", 1280 * 720),
        ("1080p", 1920 * 1080),
        ("2k", 2560 * 1440),
        ("4k", 3840 * 2160),
    ]
    RESOLUTION_LABELS = [label for label, _ in RESOLUTION_OPTIONS]

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="1hew_StringResolutionSeedance25MultimodalVideo",
            display_name="String Resolution Seedance 2.5 Multimodal Video",
            category="1hewNodes/text",
            description=(
                "从输入视频推断最接近的 Seedance 2.5 分辨率档位（480p / 720p / 1080p / 2k / 4k）；"
                "未连接视频时透传所选标签。与 1hewOffice 的 RunningHub Seedance 2.5 多模态视频节点成对使用。"
            ),
            inputs=[
                io.Combo.Input("selection", options=cls.RESOLUTION_LABELS, default="480p"),
                io.Video.Input("video", optional=True),
            ],
            outputs=[
                io.String.Output(display_name="string"),
            ],
        )

    @classmethod
    async def execute(
        cls,
        selection: str,
        video: Any = None,
    ) -> io.NodeOutput:
        fallback = selection if selection in cls.RESOLUTION_LABELS else "480p"
        size = cls._probe_size(video)
        if size is None:
            return io.NodeOutput(fallback)

        width, height = size
        return io.NodeOutput(cls._match_resolution(width, height))

    @classmethod
    def _probe_size(cls, video: Any) -> Optional[tuple[int, int]]:
        """读取视频宽高：优先 get_dimensions()，失败时退回解码后的首帧张量。"""
        get_dimensions = getattr(video, "get_dimensions", None)
        if callable(get_dimensions):
            try:
                dimensions = get_dimensions()
            except Exception:
                dimensions = None
            try:
                if dimensions is not None and len(dimensions) >= 2:
                    width, height = int(dimensions[0]), int(dimensions[1])
                    if width > 0 and height > 0:
                        return width, height
            except (TypeError, ValueError):
                pass

        get_components = getattr(video, "get_components", None)
        if callable(get_components):
            try:
                components = get_components()
                images = getattr(components, "images", None)
                if images is not None and int(images.ndim) == 4 and int(images.shape[0]) > 0:
                    return int(images.shape[2]), int(images.shape[1])
            except Exception:
                pass
        return None

    @classmethod
    def _match_resolution(cls, width: int, height: int) -> str:
        area = max(width * height, 1)
        best_label = cls.RESOLUTION_OPTIONS[0][0]
        best_diff = float("inf")

        for label, target_area in cls.RESOLUTION_OPTIONS:
            diff = abs(math.log(area) - math.log(target_area))
            if diff < best_diff:
                best_diff = diff
                best_label = label

        return best_label
