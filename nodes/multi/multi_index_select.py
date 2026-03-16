import torch
from comfy_api.latest import io


class MultiIndexSelect(io.ComfyNode):
    MAX_DECLARED_OUTPUTS = 10

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="1hew_MultiIndexSelect",
            display_name="Multi Index Select",
            category="1hewNodes/multi",
            is_input_list=True,
            inputs=[
                io.Int.Input("index", default=0, min=-999999, max=999999, step=1),
                io.Custom("*").Input("input_1", optional=True),
            ],
            outputs=[
                io.Custom("*").Output(display_name=f"output_{i}")
                for i in range(1, cls.MAX_DECLARED_OUTPUTS + 1)
            ],
        )

    @classmethod
    async def execute(cls, index, **kwargs) -> io.NodeOutput:
        idx = cls._to_int(index, default=0)

        indexed_values = {}
        for key in kwargs.keys():
            if key.startswith("input_"):
                suf = key[len("input_") :]
                if suf.isdigit():
                    slot_num = int(suf)
                    raw_val = kwargs.get(key)
                    indexed_values[slot_num] = raw_val

        if not indexed_values:
            return io.NodeOutput(None)

        results = []
        max_port = max(indexed_values.keys())
        for port_idx in range(1, max_port + 1):
            value = indexed_values.get(port_idx)
            picked = cls._pick_by_index(value, idx)
            results.append(picked)

        if len(results) > cls.MAX_DECLARED_OUTPUTS:
            results = results[: cls.MAX_DECLARED_OUTPUTS]
        return io.NodeOutput(*results)

    @staticmethod
    def _to_int(value, default=0):
        if isinstance(value, (list, tuple)):
            value = value[0] if len(value) > 0 else default
        try:
            return int(value)
        except Exception:
            return int(default)

    @classmethod
    def _pick_by_index(cls, value, idx):
        if value is None:
            return None

        value = cls._unwrap_single_container(value)

        if isinstance(value, (list, tuple)) and len(value) == 1:
            only = cls._unwrap_single_container(value[0])
            if isinstance(only, torch.Tensor):
                return cls._pick_tensor(only, idx)
            if isinstance(only, (list, tuple)):
                value = only

        if isinstance(value, torch.Tensor):
            return cls._pick_tensor(value, idx)

        if isinstance(value, (list, tuple)):
            if len(value) == 0:
                return None
            safe_idx = cls._normalize_index(idx, len(value))
            picked = value[safe_idx]
            return cls._unwrap_single_container(picked)

        return value

    @classmethod
    def _unwrap_single_container(cls, value):
        current = value
        for _ in range(8):
            if not isinstance(current, (list, tuple)) or len(current) != 1:
                break
            current = current[0]
        return current

    @classmethod
    def _pick_tensor(cls, tensor, idx):
        if tensor.ndim <= 0:
            return tensor

        count = int(tensor.shape[0]) if tensor.shape[0] is not None else 0
        if count <= 0:
            return tensor

        safe_idx = cls._normalize_index(idx, count)
        return tensor[safe_idx : safe_idx + 1]

    @staticmethod
    def _normalize_index(idx, length):
        if length <= 0:
            return 0
        if idx < 0:
            idx = length + idx
        if idx < 0:
            return 0
        if idx >= length:
            return length - 1
        return idx
