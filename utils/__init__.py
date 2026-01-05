# ComfyUI-1hewNodes 工具模块
# 此模块包含各种辅助工具和实用功能

from .workflow import start_workflow_monitor, stop_workflow_monitor, is_monitoring


def make_ui_text(text: str) -> dict:
    return {"text": [str(text)]}


def first_torch_tensor(value):
    import torch

    if isinstance(value, torch.Tensor):
        return value

    if isinstance(value, (list, tuple)):
        for item in value:
            tensor = first_torch_tensor(item)
            if tensor is not None:
                return tensor

    return None


__all__ = [
    # 工作流相关
    'start_workflow_monitor', 
    'stop_workflow_monitor', 
    'is_monitoring',
    'make_ui_text',
    'first_torch_tensor',
]
