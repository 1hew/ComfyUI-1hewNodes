import asyncio
from comfy_api.latest import io, ui
import torch
from PIL import ImageColor


class ImageBatchGroup(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="1hew_ImageBatchGroup",
            display_name="Image Batch Group",
            category="1hewNodes/batch",
            inputs=[
                io.Image.Input("image"),
                io.Int.Input("batch_size", default=81, min=1, max=1024, step=4),
                io.Int.Input("overlap", default=0, min=0, max=1024, step=1),
                io.Combo.Input(
                    "last_batch_mode",
                    options=[
                        "drop_incomplete",
                        "keep_remaining",
                        "backtrack_last",
                        "fill_color",
                    ],
                    default="backtrack_last",
                ),
                io.String.Input("color", default="1.0", optional=True),
            ],
            outputs=[
                io.Image.Output(display_name="image"),
                io.Int.Output(display_name="group_total"),
                io.MultiType.Output(display_name="start_index", is_output_list=True),
                io.MultiType.Output(display_name="batch_count", is_output_list=True),
                io.MultiType.Output(display_name="valid_count", is_output_list=True),
            ],
        )

    @classmethod
    async def execute(
        cls,
        image: torch.Tensor,
        batch_size: int,
        overlap: int,
        last_batch_mode: str,
        color: str = "1.0",
    ) -> io.NodeOutput:
        total_images = int(image.shape[0])
        cls._validate_parameters(total_images, batch_size, overlap, last_batch_mode)

        original_total = total_images

        if total_images < batch_size:
            need = batch_size - total_images
            async def _make_one():
                def _create():
                    return cls._create_white_image(image, color)
                return await asyncio.to_thread(_create)
            tasks = [_make_one() for _ in range(need)]
            colored_parts = await asyncio.gather(*tasks)
            colored_batch = torch.cat(colored_parts, dim=0)
            image = torch.cat([image, colored_batch], dim=0)
            total_images = int(image.shape[0])

        start_indices = await asyncio.to_thread(
            cls._calculate_start_indices,
            total_images,
            batch_size,
            overlap,
            last_batch_mode,
        )
        if not start_indices:
            return io.NodeOutput(image[:original_total], 0, [], [], [])

        batch_counts = await asyncio.to_thread(
            cls._calculate_batch_counts,
            start_indices,
            original_total,
            batch_size,
            last_batch_mode,
        )

        if last_batch_mode == "fill_color":
            max_needed = max(
                start_idx + batch_count
                for start_idx, batch_count in zip(start_indices, batch_counts)
            )
            if max_needed > total_images:
                need = max_needed - total_images
                async def _make_one2():
                    def _create2():
                        return cls._create_white_image(image, color)
                    return await asyncio.to_thread(_create2)
                tasks2 = [_make_one2() for _ in range(need)]
                colored_parts2 = await asyncio.gather(*tasks2)
                colored_batch = torch.cat(colored_parts2, dim=0)
                image = torch.cat([image, colored_batch], dim=0)

        valid_counts = await asyncio.to_thread(
            cls._calculate_valid_counts,
            start_indices,
            batch_counts,
            overlap,
            last_batch_mode,
            original_total,
        )

        if last_batch_mode == "fill_color" and len(valid_counts) > 0:
            last_start = start_indices[-1]
            actual_remaining = original_total - last_start
            if actual_remaining > 0:
                valid_counts[-1] = actual_remaining

        output_image = image if last_batch_mode == "fill_color" else image[:original_total]
        return io.NodeOutput(
            output_image,
            len(start_indices),
            start_indices,
            batch_counts,
            valid_counts,
        )

    @staticmethod
    def parse_color(color_str):
        if not color_str:
            return (0, 0, 0)
        color_str = str(color_str).strip()
        if color_str.startswith("(") and color_str.endswith(")"):
            color_str = color_str[1:-1].strip()
        shortcuts = {
            "r": "red",
            "g": "green",
            "b": "blue",
            "c": "cyan",
            "m": "magenta",
            "y": "yellow",
            "k": "black",
            "w": "white",
        }
        if len(color_str) == 1 and color_str.lower() in shortcuts:
            color_str = shortcuts[color_str.lower()]
        try:
            gray = float(color_str)
            if 0.0 <= gray <= 1.0:
                v = int(gray * 255)
                return (v, v, v)
        except ValueError:
            pass
        if "," in color_str:
            try:
                parts = [p.strip() for p in color_str.split(",")]
                if len(parts) >= 3:
                    r = float(parts[0])
                    g = float(parts[1])
                    b = float(parts[2])
                    if max(r, g, b) <= 1.0:
                        return (int(r * 255), int(g * 255), int(b * 255))
                    return (int(r), int(g), int(b))
            except Exception:
                pass
        try:
            return ImageColor.getrgb(color_str)
        except Exception:
            return (255, 255, 255)

    @classmethod
    def _create_white_image(cls, reference_image, color_str="1.0"):
        rgb = cls.parse_color(color_str)
        r = rgb[0] / 255.0
        g = rgb[1] / 255.0
        b = rgb[2] / 255.0
        if len(reference_image.shape) == 4:
            h, w, c = reference_image[0].shape
            colored = torch.ones(
                (1, h, w, c), dtype=reference_image.dtype, device=reference_image.device
            )
        else:
            h, w, c = reference_image.shape
            colored = torch.ones(
                (1, h, w, c), dtype=reference_image.dtype, device=reference_image.device
            )
        if c == 1:
            val = (r + g + b) / 3.0
            colored[0, :, :, 0] = val
        elif c >= 3:
            colored[0, :, :, 0] = r
            colored[0, :, :, 1] = g
            colored[0, :, :, 2] = b
            if c == 4:
                colored[0, :, :, 3] = 1.0
        return colored

    @staticmethod
    def _validate_parameters(total_images, batch_size, overlap, last_batch_mode=None):
        if total_images < 1:
            raise ValueError("输入图片数量必须大于0")
        if batch_size < 1:
            raise ValueError("批次大小必须大于0")
        if overlap < 0:
            raise ValueError("重叠帧数不能为负数")
        if last_batch_mode == "backtrack_last":
            if overlap > batch_size:
                raise ValueError(
                    f"重叠帧数 ({overlap}) 不能大于批次大小 ({batch_size})"
                )
        else:
            if overlap >= batch_size:
                raise ValueError(
                    f"重叠帧数 ({overlap}) 必须小于批次大小 ({batch_size})"
                )

    @staticmethod
    def _calculate_start_indices(total_images, batch_size, overlap, last_batch_mode):
        if total_images <= batch_size:
            if last_batch_mode == "drop_incomplete":
                return []
            return [0]
        step_size = batch_size - overlap
        if step_size <= 0:
            if overlap == batch_size:
                step_size = max(1, (batch_size + 1) // 2)
            else:
                step_size = 1
        start_indices = []
        current_start = 0
        while current_start < total_images:
            if last_batch_mode == "drop_incomplete":
                if current_start + batch_size > total_images:
                    break
            start_indices.append(current_start)
            current_start += step_size
            if (
                last_batch_mode not in ["backtrack_last", "drop_incomplete"]
                and len(start_indices) > 0
                and start_indices[-1] + batch_size >= total_images
            ):
                break
        if last_batch_mode == "backtrack_last" and len(start_indices) > 1:
            last_start = total_images - batch_size
            if last_start <= 0:
                start_indices = [0]
            else:
                if last_start < start_indices[-1]:
                    valid_indices = [0]
                    for i in range(1, len(start_indices)):
                        if start_indices[i] + overlap <= last_start:
                            valid_indices.append(start_indices[i])
                    if valid_indices[-1] != last_start:
                        valid_indices.append(last_start)
                    start_indices = valid_indices
                else:
                    start_indices[-1] = last_start
        return start_indices

    @staticmethod
    def _calculate_batch_counts(start_indices, total_images, batch_size, last_batch_mode):
        batch_counts = []
        if total_images <= batch_size:
            if len(start_indices) == 0:
                return []
            if last_batch_mode == "fill_color":
                return [batch_size]
            return [total_images]
        for i, start_idx in enumerate(start_indices):
            remaining = total_images - start_idx
            if i == len(start_indices) - 1:
                if last_batch_mode == "fill_color":
                    batch_counts.append(batch_size)
                elif last_batch_mode == "drop_incomplete":
                    batch_counts.append(batch_size)
                elif last_batch_mode == "backtrack_last":
                    if len(start_indices) == 1:
                        batch_counts.append(min(remaining, total_images))
                    else:
                        batch_counts.append(batch_size)
                else:
                    batch_counts.append(min(remaining, total_images))
            else:
                batch_counts.append(batch_size)
        return batch_counts

    @staticmethod
    def _calculate_valid_counts(
        start_indices, batch_counts, overlap, last_batch_mode, total_images=None
    ):
        valid_counts = []
        if total_images is not None and len(start_indices) <= 1:
            if len(start_indices) == 0:
                return []
            return [total_images]
        for i, (start_idx, batch_count) in enumerate(zip(start_indices, batch_counts)):
            if i == len(start_indices) - 1:
                if len(start_indices) == 1 and total_images is not None and last_batch_mode != "drop_incomplete":
                    actual_images_in_batch = total_images - start_idx
                    valid_counts.append(actual_images_in_batch)
                elif last_batch_mode == "fill_color" and total_images is not None:
                    remaining_images = total_images - start_idx
                    actual_images_in_batch = min(remaining_images, batch_count)
                    valid_counts.append(actual_images_in_batch)
                else:
                    valid_counts.append(batch_count)
            else:
                next_start = start_indices[i + 1]
                valid_count = next_start - start_idx
                valid_counts.append(valid_count)
        return valid_counts