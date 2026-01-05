from __future__ import annotations

import hashlib
import os

from comfy_api.latest import io

from ...utils import make_ui_text


class GetFileCount(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="1hew_GetFileCount",
            display_name="Get File Count",
            category="1hewNodes/io",
            inputs=[
                io.String.Input("folder", default=""),
                io.Combo.Input("type", default="image", options=["image", "video"]),
                io.Boolean.Input("include_subdir", default=True),
            ],
            outputs=[
                io.Int.Output(display_name="count"),
                io.String.Output(display_name="folder"),
                io.Boolean.Output(display_name="include_subdir"),
            ],
        )

    @staticmethod
    def get_paths(folder, include_subdir, extensions):
        if not os.path.isdir(folder):
            return []

        paths = []
        if include_subdir:
            for root, _dirs, files in os.walk(folder):
                for file in files:
                    if os.path.splitext(file)[1].lower() in extensions:
                        paths.append(os.path.join(root, file))
        else:
            for file in os.listdir(folder):
                path = os.path.join(folder, file)
                if (
                    os.path.isfile(path)
                    and os.path.splitext(file)[1].lower() in extensions
                ):
                    paths.append(path)

        # 确保排序一致，避免不同平台/文件系统顺序差异
        paths.sort(key=lambda x: x.lower())
        return paths

    @classmethod
    def IS_CHANGED(cls, folder: str, type: str, include_subdir: bool, **kwargs):
        if not os.path.isdir(folder):
            return float("nan")

        if type == "image":
            extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp'}
        elif type == "video":
            extensions = {'.webm', '.mp4', '.mkv', '.gif', '.mov', '.avi'}
        else:
            extensions = set()

        paths = cls.get_paths(folder, include_subdir, extensions)
        m = hashlib.sha256()
        # 这里对文件“路径列表”做哈希，作为 IS_CHANGED 的变化依据：
        # - 计数的变化通常来自文件新增/删除
        # - 只哈希路径比哈希 mtime 更轻量，适合本节点用途
        # - 下游如按 index 加载图片/视频，一般会自行处理内容变化

        for path in paths:
            m.update(path.encode())

        return m.hexdigest()

    @classmethod
    def execute(cls, folder: str, type: str, include_subdir: bool) -> io.NodeOutput:
        if type == "image":
            # ComfyUI 常用图片格式
            extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp'}
        elif type == "video":
            # ComfyUI 常用视频格式
            extensions = {'.webm', '.mp4', '.mkv', '.mov', '.avi'}
        else:
            extensions = set()

        paths = cls.get_paths(folder, include_subdir, extensions)
        count = len(paths)
        return io.NodeOutput(
            count,
            folder,
            include_subdir,
            ui=make_ui_text(str(count)),
        )
