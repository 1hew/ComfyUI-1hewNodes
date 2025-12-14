from __future__ import annotations
import os
import hashlib
from comfy_api.latest import io

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
                io.Boolean.Input("include_subfolder", default=True),
            ],
            outputs=[
                io.Int.Output(display_name="count"),
            ],
        )

    @staticmethod
    def get_paths(folder, include_subfolder, extensions):
        if not os.path.isdir(folder):
            return []
            
        paths = []
        if include_subfolder:
            for root, dirs, files in os.walk(folder):
                for file in files:
                    if os.path.splitext(file)[1].lower() in extensions:
                        paths.append(os.path.join(root, file))
        else:
            for file in os.listdir(folder):
                path = os.path.join(folder, file)
                if os.path.isfile(path) and os.path.splitext(file)[1].lower() in extensions:
                    paths.append(path)
        
        # Ensure consistent order
        paths.sort(key=lambda x: x.lower())
        return paths

    @classmethod
    def IS_CHANGED(cls, folder: str, type: str, include_subfolder: bool, **kwargs):
        if not os.path.isdir(folder):
            return float("nan")
            
        if type == "image":
            extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp'}
        elif type == "video":
            extensions = {'.webm', '.mp4', '.mkv', '.gif', '.mov', '.avi'}
        else:
            extensions = set()
            
        paths = cls.get_paths(folder, include_subfolder, extensions)
        m = hashlib.sha256()
        # Only hash paths (existence), not mtime, because count only changes if files are added/removed
        # But if we want to be safe, maybe hash paths is enough.
        # However, if a file is replaced by another with same name, count is same.
        # If file is renamed, count is same.
        # But usually GetFileCount is used to know how many files to iterate.
        # So maybe just returning count is enough for IS_CHANGED?
        # No, IS_CHANGED return value is compared. If it returns the same count, the node is NOT re-executed.
        # But execute is trivial (len(paths)).
        # The important thing is if downstream nodes rely on this.
        # Actually, if count is same, output is same. So re-executing doesn't change output.
        # So IS_CHANGED returning count is actually optimal?
        # Wait, if I replace "a.jpg" with "b.jpg", count is same. Downstream might be iterating 0..N.
        # If downstream uses LoadImageFromFolder with index, and LoadImageFromFolder has its own IS_CHANGED, then it handles it.
        # So GetFileCount only needs to update if count changes?
        # Yes, technically correct.
        # But let's be safe and detect content changes too in case someone uses it for triggering.
        # I'll stick to full path hashing.
        
        for path in paths:
             m.update(path.encode())
             
        return m.hexdigest()

    @classmethod
    def execute(cls, folder: str, type: str, include_subfolder: bool) -> io.NodeOutput:
        if type == "image":
            # ComfyUI typically supports these image formats
            extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp'}
        elif type == "video":
            # ComfyUI typically supports these video formats
            extensions = {'.webm', '.mp4', '.mkv', '.mov', '.avi'}
        else:
            extensions = set()
            
        paths = cls.get_paths(folder, include_subfolder, extensions)
        return io.NodeOutput(len(paths))
