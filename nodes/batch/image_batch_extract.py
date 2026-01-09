import asyncio
import re

from comfy_api.latest import io, ui
import torch


class ImageBatchExtract(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="1hew_ImageBatchExtract",
            display_name="Image Batch Extract",
            category="1hewNodes/batch",
            inputs=[
                io.Image.Input("image"),
                io.Combo.Input("mode", options=["index", "step", "uniform"], default="step"),
                io.String.Input("index", default="0"),
                io.Int.Input("step", default=4, min=1, max=8192, step=1),
                io.Int.Input("uniform", default=4, min=0, max=8192, step=1),
                io.Int.Input("max_keep", default=10, min=0, max=8192, step=1),
            ],
            outputs=[io.Image.Output(display_name="image")],
        )

    @classmethod
    async def execute(
        cls,
        image: torch.Tensor,
        mode: str,
        index: str,
        step: int,
        uniform: int,
        max_keep: int,
    ) -> io.NodeOutput:
        try:
            batch_size = int(image.shape[0])
            print(f"[ImageBatchExtract] 输入: 形状={tuple(image.shape)}, 总帧={batch_size}")
            print(f"[ImageBatchExtract] 参数: 模式={mode}, 索引='{index}',步长={step}, 数量={uniform}, 最大保留={max_keep}")

            indices = await asyncio.to_thread(
                cls._get_extract_indices, batch_size, mode, index, step, uniform
            )

            if not indices:
                print("[ImageBatchExtract] 索引为空，返回空结果")
                empty = torch.empty((0,) + image.shape[1:], dtype=image.dtype, device=image.device)
                return io.NodeOutput(empty)

            if max_keep > 0 and len(indices) > max_keep:
                print(f"[ImageBatchExtract] 保留限制: {len(indices)} -> {max_keep}")
                indices = indices[:max_keep]
            elif max_keep == 0:
                print(f"[ImageBatchExtract] max_keep=0，保留所有{len(indices)}张图像")

            valid = [i for i in indices if 0 <= i < batch_size]
            invalid = [i for i in indices if not (0 <= i < batch_size)]
            if invalid:
                print(f"[ImageBatchExtract] 跳过索引: {invalid}")

            if not valid:
                print("[ImageBatchExtract] 无有效索引，返回空结果")
                empty = torch.empty((0,) + image.shape[1:], dtype=image.dtype, device=image.device)
                return io.NodeOutput(empty)

            chunk_size = 512
            chunks = [valid[i : i + chunk_size] for i in range(0, len(valid), chunk_size)]
            async def _gather_chunk(chunk):
                def _select():
                    return image[chunk]
                return await asyncio.to_thread(_select)

            tasks = [_gather_chunk(ch) for ch in chunks]
            parts = await asyncio.gather(*tasks)
            extracted = torch.cat(parts, dim=0)

            print(f"[ImageBatchExtract] 完成: {len(valid)}张, 索引={[int(i) for i in valid]}")
            print(f"[ImageBatchExtract] 输出形状: {tuple(extracted.shape)}")
            return io.NodeOutput(extracted)
        except Exception as e:
            print(f"[ImageBatchExtract] 错误: {str(e)}")
            empty = torch.empty((0,) + image.shape[1:], dtype=image.dtype, device=image.device)
            return io.NodeOutput(empty)

    @classmethod
    def _get_extract_indices(
        cls,
        batch_size: int,
        mode: str,
        index: str,
        step: int,
        uniform: int,
    ):
        try:
            if mode == "index":
                if not index.strip():
                    print("[ImageBatchExtract] 自定义索引为空")
                    return []
                print(f"[ImageBatchExtract] 自定义索引: '{index}'")
                return cls._parse_custom_indices(index, batch_size)
            if mode == "step":
                if step < 1:
                    print("[ImageBatchExtract] 步长必须≥1")
                    return []
                print(f"[ImageBatchExtract] 步长模式: {step}")
                return cls._calculate_step_indices(batch_size, step)
            if mode == "uniform":
                if uniform <= 0:
                    print("[ImageBatchExtract] 数量≤0")
                    return []
                print(f"[ImageBatchExtract] 均匀数量: {uniform}")
                return cls._calculate_count_indices(batch_size, uniform)
            return []
        except Exception as e:
            print(f"[ImageBatchExtract] 索引计算错误: {str(e)}")
            return []

    @staticmethod
    def _parse_custom_indices(indices_str: str, batch_size: int | None = None):
        indices: list[int] = []
        try:
            normalized = (
                indices_str.replace("，", ",")
                .replace("；", ",")
                .replace(";", ",")
            )
            parts = [p.strip() for p in normalized.split(",") if p.strip()]
            for part in parts:
                match_range = re.match(
                    r"^\s*(-?\d+)\s*-\s*(-?\d+)\s*$",
                    part,
                )
                if match_range:
                    start = int(match_range.group(1))
                    end = int(match_range.group(2))
                    if batch_size is not None:
                        if start < 0:
                            start = batch_size + start
                        if end < 0:
                            end = batch_size + end
                    step = 1 if start <= end else -1
                    indices.extend(list(range(start, end + step, step)))
                    continue

                match_int = re.match(r"^\s*(-?\d+)\s*$", part)
                if match_int:
                    idx = int(match_int.group(1))
                    if batch_size is not None and idx < 0:
                        idx = batch_size + idx
                    indices.append(idx)
                    continue

                print(f"[ImageBatchExtract] 跳过无效索引: '{part}'")
            print(f"[ImageBatchExtract] 解析索引 -> {indices}")
        except Exception as e:
            print(f"[ImageBatchExtract] 解析错误: {str(e)}")
            indices = []
        return indices

    @staticmethod
    def _calculate_step_indices(batch_size: int, step: int):
        idxs = list(range(0, batch_size, step))
        print(
            f"[ImageBatchExtract] 步长计算: 总帧={batch_size}, 步长={step} -> {idxs}"
        )
        return idxs

    @staticmethod
    def _calculate_count_indices(batch_size: int, count: int):
        if count <= 0:
            return []
        if count == 1:
            return [0]
        if count == 2:
            return [0, batch_size - 1] if batch_size > 1 else [0]
        if count >= batch_size:
            return list(range(batch_size))
        step = (batch_size - 1) / float(count - 1)
        idxs = [int(round(i * step)) for i in range(count)]
        idxs[-1] = batch_size - 1
        idxs = sorted(list(set(idxs)))
        print(
            f"[ImageBatchExtract] 均匀计算: 总帧={batch_size}, 数量={count} -> {idxs}"
        )
        return idxs
