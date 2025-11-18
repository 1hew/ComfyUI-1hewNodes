import asyncio
from comfy_api.latest import io
from datetime import datetime
import folder_paths
import os
import time


class WorkflowName(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="1hew_WorkflowName",
            display_name="Workflow Name",
            category="1hewNodes/util",
            inputs=[
                io.String.Input("prefix", default=""),
                io.String.Input("suffix", default=""),
                io.Combo.Input(
                    "date_format",
                    options=[
                        "none",
                        "yyyy-MM-dd",
                        "yyyy/MM/dd",
                        "yyyyMMdd",
                        "yyyy-MM-dd HH:mm",
                        "yyyy/MM/dd HH:mm",
                        "yyyy-MM-dd HH:mm:ss",
                        "MM-dd",
                        "MM/dd",
                        "MMdd",
                        "dd",
                        "HH:mm",
                        "HH:mm:ss",
                        "yyyy年MM月dd日",
                        "MM月dd日",
                        "yyyyMMdd_HHmm",
                        "yyyyMMdd_HHmmss",
                    ],
                    default="yyyy-MM-dd",
                ),
                io.Boolean.Input("full_path", default=False),
                io.Boolean.Input("strip_extension", default=True),
            ],
            outputs=[io.String.Output(display_name="string")],
        )

    @classmethod
    def fingerprint_inputs(cls, **kwargs):
        return str(time.time())

    @classmethod
    async def execute(
        cls,
        prefix: str,
        suffix: str,
        date_format: str,
        full_path: bool,
        strip_extension: bool,
    ) -> io.NodeOutput:
        try:
            comfyui_root = folder_paths.base_path
            candidates = [
                os.path.join(
                    comfyui_root,
                    "custom_nodes",
                    "ComfyUI-1hewNodesV3",
                    "utils",
                    "workflow",
                    "current_workflow.tmp",
                ),
                os.path.join(
                    comfyui_root,
                    "custom_nodes",
                    "ComfyUI-1hewNodes",
                    "utils",
                    "workflow",
                    "current_workflow.tmp",
                ),
            ]

            temp_file_path = None
            for p in candidates:
                if os.path.exists(p):
                    temp_file_path = p
                    break

            if not temp_file_path:
                return io.NodeOutput("未检测到临时文件（请启动监控脚本）")

            for attempt in range(3):
                try:
                    with open(temp_file_path, "r", encoding="utf-8") as f:
                        file_path = f.read().strip()
                    if not file_path:
                        return io.NodeOutput("临时文件为空（未保存工作流）")
                    result = cls._process_workflow_path(
                        file_path,
                        full_path,
                        date_format,
                        prefix,
                        suffix,
                        strip_extension,
                    )
                    return io.NodeOutput(result)
                except PermissionError:
                    if attempt < 2:
                        await asyncio.sleep(0.05)
                        continue
                    return io.NodeOutput("无法读取临时文件（文件被占用）")
                except Exception as read_error:
                    if attempt < 2:
                        await asyncio.sleep(0.05)
                        continue
                    return io.NodeOutput(f"读取错误：{str(read_error)}")

            return io.NodeOutput("无法读取临时文件（多次尝试失败）")
        except Exception as e:
            return io.NodeOutput(f"错误：{str(e)}")

    @classmethod
    def _process_workflow_path(
        cls,
        file_path: str,
        full_path: bool,
        date_format: str,
        prefix: str,
        suffix: str,
        strip_extension: bool,
    ) -> str:
        try:
            file_path = file_path.replace("\\", "/")
            if full_path:
                if "/" in file_path:
                    dir_part, file_name = file_path.rsplit("/", 1)
                else:
                    dir_part = ""
                    file_name = file_path
            else:
                file_name = os.path.basename(file_path)
                dir_part = ""
            if file_name.endswith(".json"):
                name_without_ext = file_name[:-5]
                extension = ".json"
            else:
                name_without_ext = file_name
                extension = ""
            processed_name = name_without_ext
            if prefix:
                processed_name = prefix + processed_name
            if suffix:
                processed_name = processed_name + suffix
            if strip_extension:
                final_name = processed_name
            else:
                final_name = processed_name + extension
            if full_path and dir_part:
                result = f"{dir_part}/{final_name}"
            else:
                result = final_name
            if date_format != "none":
                now = datetime.now()
                if date_format == "yyyy-MM-dd":
                    date_str = now.strftime("%Y-%m-%d")
                elif date_format == "yyyy/MM/dd":
                    date_str = now.strftime("%Y/%m/%d")
                elif date_format == "yyyyMMdd":
                    date_str = now.strftime("%Y%m%d")
                elif date_format == "yyyy-MM-dd HH:mm":
                    date_str = now.strftime("%Y-%m-%d %H:%M")
                elif date_format == "yyyy/MM/dd HH:mm":
                    date_str = now.strftime("%Y/%m/%d %H:%M")
                elif date_format == "yyyy-MM-dd HH:mm:ss":
                    date_str = now.strftime("%Y-%m-%d %H:%M:%S")
                elif date_format == "MM-dd":
                    date_str = now.strftime("%m-%d")
                elif date_format == "MM/dd":
                    date_str = now.strftime("%m/%d")
                elif date_format == "MMdd":
                    date_str = now.strftime("%m%d")
                elif date_format == "dd":
                    date_str = now.strftime("%d")
                elif date_format == "HH:mm":
                    date_str = now.strftime("%H:%M")
                elif date_format == "HH:mm:ss":
                    date_str = now.strftime("%H:%M:%S")
                elif date_format == "yyyy年MM月dd日":
                    date_str = now.strftime("%Y年%m月%d日")
                elif date_format == "MM月dd日":
                    date_str = now.strftime("%m月%d日")
                elif date_format == "yyyyMMdd_HHmm":
                    date_str = now.strftime("%Y%m%d_%H%M")
                elif date_format == "yyyyMMdd_HHmmss":
                    date_str = now.strftime("%Y%m%d_%H%M%S")
                else:
                    date_str = now.strftime("%Y-%m-%d")
                result = f"{date_str}/{result}"
            return result
        except Exception as e:
            return f"路径处理错误：{str(e)}"