import math
import torch
from comfy_api.latest import io


class ImagePingPong(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="1hew_ImagePingPong",
            display_name="Image PingPong",
            category="1hewNodes/image",
            inputs=[
                io.Image.Input("image"),
                io.Boolean.Input("pre_reverse", default=False),
                io.Int.Input("ops_count", default=1, min=0, max=100000, step=1),
                io.Int.Input("frame_count", default=0, min=0, max=1000000, step=1),
                io.Boolean.Input("remove_link_frame", default=True),
            ],
            outputs=[io.Image.Output(display_name="image")],
        )

    @classmethod
    async def execute(
        cls,
        image: torch.Tensor,
        ops_count: int,
        frame_count: int,
        remove_link_frame: bool,
        pre_reverse: bool,
    ) -> io.NodeOutput:
        B = int(image.shape[0])
        if B <= 0:
            if image.ndim == 4:
                _, h, w, c = image.shape
                empty = torch.zeros(
                    (0, h, w, c), dtype=image.dtype, device=image.device
                )
            else:
                empty = torch.zeros((0, 64, 64, 3), dtype=torch.float32)
            return io.NodeOutput(empty)

        if pre_reverse:
            idx_rev = torch.arange(
                B - 1, -1, -1, dtype=torch.long, device=image.device
            )
            image = image[idx_rev]

        forward = list(range(B))
        backward = list(range(B - 1, -1, -1))

        def seg(dir_idx: int):
            if dir_idx % 2 == 0:
                return forward[:]
            return backward[:]

        def seg_trimmed(dir_idx: int):
            s = seg(dir_idx)
            if not remove_link_frame:
                return s
            if dir_idx == 0:
                return s
            if dir_idx % 2 == 1:
                return s[1:] if len(s) > 0 else s
            return s[1:] if len(s) > 0 else s

        indices = []

        if ops_count == 0 and frame_count == 0:
            indices = []
        elif ops_count > 0:
            for i in range(int(ops_count)):
                indices.extend(seg_trimmed(i))
            if frame_count > 0:
                indices = indices[: int(frame_count)]
        else:
            # 无限操作，按 frame_count 截取
            base_cycle = seg_trimmed(0) + seg_trimmed(1)
            if len(base_cycle) == 0:
                indices = []
            else:
                N = int(frame_count)
                need = max(1, math.ceil(N / len(base_cycle)))
                indices = (base_cycle * need)[:N]

        index_t = torch.tensor(indices, dtype=torch.long, device=image.device)
        out = image[index_t]
        return io.NodeOutput(out)