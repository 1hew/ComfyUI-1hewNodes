
import asyncio
from comfy_api.latest import io
import torch


class MaskBatchSplit(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="1hew_MaskBatchSplit",
            display_name="Mask Batch Split",
            category="1hewNodes/batch",
            inputs=[
                io.Mask.Input("mask"),
                io.Int.Input("take_count", default=8, min=1, max=1024, step=1),
                io.Boolean.Input("from_start", default=False),
            ],
            outputs=[
                io.Mask.Output(display_name="mask_1"),
                io.Mask.Output(display_name="mask_2"),
            ],
        )

    @classmethod
    async def execute(
        cls, mask: torch.Tensor, take_count: int, from_start: bool
    ) -> io.NodeOutput:
        try:
            batch_size = int(mask.shape[0])
            print(
                f"[MaskBatchSplit] 输入: 形状={tuple(mask.shape)}, 总数={batch_size}"
            )
            print(
                f"[MaskBatchSplit] 参数: 取数={take_count}, 从开头切={from_start}"
            )

            if take_count >= batch_size:
                if from_start:
                    empty_second = torch.empty(
                        (0,) + mask.shape[1:],
                        dtype=mask.dtype,
                        device=mask.device,
                    )
                    return io.NodeOutput(mask, empty_second)
                else:
                    empty_first = torch.empty(
                        (0,) + mask.shape[1:],
                        dtype=mask.dtype,
                        device=mask.device,
                    )
                    return io.NodeOutput(empty_first, mask)

            if from_start:
                first_count = take_count
                second_count = batch_size - take_count
                async def _slice_first():
                    def _do_first():
                        return mask[:first_count]
                    return await asyncio.to_thread(_do_first)
                async def _slice_second():
                    def _do_second():
                        return mask[first_count:]
                    return await asyncio.to_thread(_do_second)
                first_batch, second_batch = await asyncio.gather(_slice_first(), _slice_second())
                print(
                    f"[MaskBatchSplit] from_start=True: 第一={first_count}, 第二={second_count}"
                )
            else:
                first_count = batch_size - take_count
                second_count = take_count
                async def _slice_first2():
                    def _do_first2():
                        return mask[:first_count]
                    return await asyncio.to_thread(_do_first2)
                async def _slice_second2():
                    def _do_second2():
                        return mask[first_count:]
                    return await asyncio.to_thread(_do_second2)
                first_batch, second_batch = await asyncio.gather(_slice_first2(), _slice_second2())
                print(
                    f"[MaskBatchSplit] from_start=False: 第一={first_count}, 第二={second_count}"
                )

            print(
                f"[MaskBatchSplit] 输出形状: 第一={tuple(first_batch.shape)}, "
                f"第二={tuple(second_batch.shape)}"
            )
            return io.NodeOutput(first_batch, second_batch)
        except Exception as e:
            print(f"[MaskBatchSplit] 错误: {str(e)}")
            empty = torch.empty(
                (0,) + mask.shape[1:], dtype=mask.dtype, device=mask.device
            )
            return io.NodeOutput(mask, empty)
