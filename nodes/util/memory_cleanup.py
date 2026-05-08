import gc

from comfy_api.latest import io


class MemoryCleanup(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="1hew_MemoryCleanup",
            display_name="Memory Cleanup",
            category="1hewNodes/util",
            description=(
                "请求 ComfyUI 在当前任务结束后释放执行缓存与模型占用，"
                "适合多次排队执行之间清理 RAM/VRAM。"
            ),
            inputs=[
                io.Custom("*").Input("anything", optional=True),
                io.Boolean.Input(
                    "unload_model",
                    default=False,
                    tooltip="是否在当前任务结束后请求卸载已加载模型。开启更省 VRAM/RAM，但下次执行可能需要重新加载模型。",
                ),
            ],
            outputs=[
                io.Custom("*").Output(display_name="output"),
            ],
            is_output_node=True,
        )

    @classmethod
    def validate_inputs(cls, **kwargs):
        return True

    @classmethod
    def execute(
        cls,
        anything=None,
        unload_model: bool = False,
    ) -> io.NodeOutput:
        gc.collect()
        try:
            import comfy.model_management as model_management

            model_management.soft_empty_cache()
        except Exception as exc:
            print(f"[Memory Cleanup] soft_empty_cache skipped: {exc}")

        try:
            from server import PromptServer

            prompt_queue = getattr(PromptServer.instance, "prompt_queue", None)
            if prompt_queue is None:
                print("[Memory Cleanup] prompt_queue unavailable")
            else:
                if unload_model:
                    prompt_queue.set_flag("unload_models", True)
                prompt_queue.set_flag("free_memory", True)
        except Exception as exc:
            print(f"[Memory Cleanup] cleanup request failed: {exc}")

        return io.NodeOutput(anything)
