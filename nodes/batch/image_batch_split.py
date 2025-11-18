import asyncio
from comfy_api.latest import io, ui
import torch


class ImageBatchSplit(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="1hew_ImageBatchSplit",
            display_name="Image Batch Split",
            category="1hewNodes/batch",
            inputs=[
                io.Image.Input("image"),
                io.Int.Input("take_count", default=8, min=1, max=1024, step=1),
                io.Boolean.Input("from_start", default=False),
            ],
            outputs=[
                io.Image.Output(display_name="image_1"),
                io.Image.Output(display_name="image_2"),
            ],
        )

    @classmethod
    async def execute(cls, image: torch.Tensor, take_count: int, from_start: bool) -> io.NodeOutput:
        try:
            batch_size = int(image.shape[0])
            print(f"[ImageBatchSplit] 输入: 形状={tuple(image.shape)}, 总数={batch_size}")
            print(f"[ImageBatchSplit] 参数: 取数={take_count}, 从开头切={from_start}")

            if take_count >= batch_size:
                print(f"[ImageBatchSplit] 边界: 取数({take_count})≥总数({batch_size})")
                if from_start:
                    empty_second = torch.empty((0,) + image.shape[1:], dtype=image.dtype, device=image.device)
                    print(f"[ImageBatchSplit] from_start=True 输出: 第一=全部, 第二=空")
                    return io.NodeOutput(image, empty_second)
                else:
                    empty_first = torch.empty((0,) + image.shape[1:], dtype=image.dtype, device=image.device)
                    print(f"[ImageBatchSplit] from_start=False 输出: 第一=空, 第二=全部")
                    return io.NodeOutput(empty_first, image)

            if from_start:
                first_count = take_count
                second_count = batch_size - take_count
                async def _slice_first():
                    def _do_first():
                        return image[:first_count]
                    return await asyncio.to_thread(_do_first)
                async def _slice_second():
                    def _do_second():
                        return image[first_count:]
                    return await asyncio.to_thread(_do_second)
                first_batch, second_batch = await asyncio.gather(_slice_first(), _slice_second())
                print(f"[ImageBatchSplit] from_start=True: 第一={first_count}, 第二={second_count}")
            else:
                first_count = batch_size - take_count
                second_count = take_count
                async def _slice_first2():
                    def _do_first2():
                        return image[:first_count]
                    return await asyncio.to_thread(_do_first2)
                async def _slice_second2():
                    def _do_second2():
                        return image[first_count:]
                    return await asyncio.to_thread(_do_second2)
                first_batch, second_batch = await asyncio.gather(_slice_first2(), _slice_second2())
                print(f"[ImageBatchSplit] from_start=False: 第一={first_count}, 第二={second_count}")

            print(f"[ImageBatchSplit] 输出形状: 第一={tuple(first_batch.shape)},第二={tuple(second_batch.shape)}")
            return io.NodeOutput(first_batch, second_batch)
        except Exception as e:
            print(f"[ImageBatchSplit] 错误: {str(e)}")
            empty = torch.empty((0,) + image.shape[1:], dtype=image.dtype, device=image.device)
            return io.NodeOutput(image, empty)