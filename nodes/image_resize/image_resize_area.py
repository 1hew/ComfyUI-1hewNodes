import math

from comfy_api.latest import io
import numpy as np
from PIL import Image, ImageColor
import torch
import torch.nn.functional as F

from ...utils import first_torch_tensor, make_ui_text


class ImageResizeArea(io.ComfyNode):
    """按面积自动缩放图像，保持原图比例，宽高对齐 divisible_by 的倍数。

    - N = sqrt(width x height)，夹到 [min_sq_area, max_sq_area] 区间，并对齐 divisible_by 的倍数。
    - 目标尺寸 = 原图按比例缩放到面积约 N^2，宽高再向上取整到 divisible_by 的倍数，
      与 Image Resize Universal 的 length_to_sq_area 计算一致。
    - 面积在 [min_sq_area^2, max_sq_area^2] 内：原图 / 原遮罩透传。
    - 超过上限：用 method 缩小。
    - 低于下限：接了 upscale_model 就先模型放大，再用 method 归一到目标尺寸；否则 method 直接放大。

    本文件自包含：不 import 其它节点模块。fit / pad_color / mask 的缩放内核
    （fit_resize_image / fit_resize_mask / generate_default_mask / parse_color 等）
    已就地内联，行为与 Image Resize Universal 一致。
    """

    METHODS = ["nearest", "bilinear", "lanczos", "bicubic", "hamming", "box"]
    FIT_MODES = ["crop", "pad", "stretch"]

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="1hew_ImageResizeArea",
            display_name="Image Resize Area",
            category="1hewNodes/image/resize",
            inputs=[
                io.Image.Input("image"),
                io.Mask.Input("mask", optional=True),
                io.Custom("UPSCALE_MODEL").Input("upscale_model", optional=True, lazy=True),
                io.Int.Input("min_sq_area", default=1024, min=1, max=8192, step=1),
                io.Int.Input("max_sq_area", default=2048, min=1, max=8192, step=1),
                io.Combo.Input("method", options=cls.METHODS, default="lanczos"),
                io.Combo.Input("fit", options=cls.FIT_MODES, default="crop"),
                io.String.Input("pad_color", default="1.0"),
                io.Int.Input("divisible_by", default=32, min=1, max=1024, step=1),
            ],
            outputs=[
                io.Image.Output(display_name="image"),
                io.Mask.Output(display_name="mask"),
                io.Int.Output(display_name="int"),
            ],
        )

    @classmethod
    async def execute(
        cls,
        image: torch.Tensor,
        min_sq_area: int,
        max_sq_area: int,
        method: str,
        fit: str,
        pad_color: str,
        divisible_by: int,
        mask: torch.Tensor | None = None,
        upscale_model=None,
    ) -> io.NodeOutput:
        tensor = first_torch_tensor(image)
        if tensor is None or getattr(tensor, "ndim", 0) < 3:
            raise ValueError("image must be a BHWC IMAGE tensor")
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)

        height = int(tensor.shape[1])
        width = int(tensor.shape[2])
        if height < 1 or width < 1:
            raise ValueError("image dimensions must be positive")

        mask_frames = cls.split_mask(mask, width, height)

        low, high = int(min_sq_area), int(max_sq_area)
        if low > high:
            low, high = high, low

        raw = math.sqrt(width * height)
        n = cls.sq_area_length(width, height, low, high, divisible_by)

        if low <= raw <= high:
            # 区间内：图像 / 遮罩原样透传，只报告对齐后的 N。
            out_image = tensor
            if mask_frames is None:
                out_mask = cls.default_mask(len(tensor), width, height, None, width, height)
            else:
                out_mask = torch.stack(mask_frames, dim=0)
        else:
            work = tensor
            # 低于下限且接了超分模型：先模型放大，再用 method 归一到目标尺寸。
            if raw < low and upscale_model is not None:
                work = cls.upscale_with_model(work, upscale_model)

            source_width = int(work.shape[2])
            source_height = int(work.shape[1])
            target_width, target_height = cls.target_size(
                source_width, source_height, n, divisible_by
            )

            out_image = cls.resize_image_batch(
                work, target_width, target_height, fit, pad_color, method
            )
            if mask_frames is None:
                out_mask = cls.default_mask(
                    len(work), target_width, target_height, fit, source_width, source_height
                )
            else:
                out_mask = cls.resize_mask_frames(
                    mask_frames, target_width, target_height, fit, method
                )

        return io.NodeOutput(out_image, out_mask, n, ui=make_ui_text(str(n)))

    @classmethod
    def validate_inputs(cls, **kwargs):
        return True

    @classmethod
    def check_lazy_status(cls, image=None, min_sq_area=1024, max_sq_area=2048, **kwargs):
        """只有图像低于 min_sq_area 时才请求 upscale_model，避免无谓地加载/运行模型。"""
        if "upscale_model" not in kwargs or kwargs.get("upscale_model") is not None:
            return []
        tensor = first_torch_tensor(image)
        if tensor is None or getattr(tensor, "ndim", 0) < 3:
            return []
        height = int(tensor.shape[-3])
        width = int(tensor.shape[-2])
        if height < 1 or width < 1:
            return []
        low, high = int(min_sq_area), int(max_sq_area)
        if low > high:
            low, high = high, low
        if math.sqrt(width * height) < low:
            return ["upscale_model"]
        return []

    # ------------------------------------------------------------------
    # 尺寸计算
    # ------------------------------------------------------------------

    @staticmethod
    def sq_area_length(width: int, height: int, min_sq_area: int, max_sq_area: int, divisible_by: int) -> int:
        """返回 sqrt(width x height) 的区间夹取 + 倍数对齐结果 N。"""
        step = max(int(divisible_by), 1)
        low_bound, high_bound = int(min_sq_area), int(max_sq_area)
        if low_bound > high_bound:
            low_bound, high_bound = high_bound, low_bound

        # 区间内可用的 divisible_by 倍数（向内收缩，保证不越界）。
        low = math.ceil(low_bound / step) * step
        high = math.floor(high_bound / step) * step
        if low > high:
            raise ValueError(
                f"no multiple of {step} exists within [{low_bound}, {high_bound}]"
            )

        raw = math.sqrt(width * height)
        # 就近取整到 step 的倍数（half-up），再夹回可用区间。
        value = int(math.floor(raw / step + 0.5)) * step
        value = max(low, min(value, high))
        return max(value, 1)

    @staticmethod
    def target_size(width: int, height: int, n: int, divisible_by: int):
        """保持原图比例、面积约等于 n^2，并把宽高向上取整到 divisible_by 的倍数。

        与 Image Resize Universal 的 length_to_sq_area 计算保持一致。
        """
        ratio = width / height
        target_width = math.sqrt(ratio) * n
        target_height = target_width / ratio
        target_width = max(1, int(target_width))
        target_height = max(1, int(target_height))

        step = max(int(divisible_by), 1)
        if step > 1:
            target_width = math.ceil(target_width / step) * step
            target_height = math.ceil(target_height / step) * step
        return target_width, target_height

    @staticmethod
    def resolve_sampler(method: str):
        return {
            "nearest": Image.NEAREST,
            "bilinear": Image.BILINEAR,
            "bicubic": Image.BICUBIC,
            "hamming": Image.HAMMING,
            "box": Image.BOX,
        }.get(method, Image.LANCZOS)

    # ------------------------------------------------------------------
    # 遮罩 / 批次处理
    # ------------------------------------------------------------------

    @staticmethod
    def split_mask(mask, width: int, height: int):
        """校验遮罩尺寸并按帧拆开；未连接时返回 None。"""
        if mask is None:
            return None
        frames_tensor = mask
        if getattr(frames_tensor, "dim", lambda: 0)() == 2:
            frames_tensor = frames_tensor.unsqueeze(0)
        mask_height = int(frames_tensor.shape[1])
        mask_width = int(frames_tensor.shape[2])
        if mask_width != width or mask_height != height:
            raise ValueError(
                f"mask size {mask_width}x{mask_height} must match image size {width}x{height}"
            )
        return [frames_tensor[i] for i in range(int(frames_tensor.shape[0]))]

    @classmethod
    def default_mask(cls, count: int, target_width: int, target_height: int, fit, orig_width: int, orig_height: int):
        """未连接遮罩时生成默认遮罩：pad 为原图区域白 + 填充区黑，其余全白。"""
        base = cls.generate_default_mask(target_width, target_height, fit, orig_width, orig_height)
        single = cls.pil_to_mask_tensor(base)
        return single.repeat(max(int(count), 1), 1, 1)

    @classmethod
    def resize_image_batch(cls, image: torch.Tensor, target_width: int, target_height: int, fit: str, pad_color: str, method: str):
        """按 fit / pad_color 逐帧缩放，保留 alpha。"""
        sampler = cls.resolve_sampler(method)
        frames = []
        for frame in image:
            pil = cls.tensor_to_pil(frame)
            if "A" in pil.getbands():
                alpha = pil.getchannel("A")
                rgb = pil.convert("RGB")
                resized_rgb = cls.fit_resize_image(
                    rgb, target_width, target_height, fit, sampler, pad_color
                ).convert("RGB")
                resized_alpha = cls.fit_resize_mask(
                    alpha, target_width, target_height, fit, sampler, pil.width, pil.height
                ).convert("L")
                resized_rgb.putalpha(resized_alpha)
                frames.append(cls.pil_to_tensor(resized_rgb))
            else:
                resized = cls.fit_resize_image(
                    pil.convert("RGB"), target_width, target_height, fit, sampler, pad_color
                )
                frames.append(cls.pil_to_tensor(resized))
        return torch.cat(frames, dim=0)

    @classmethod
    def resize_mask_frames(cls, frames, target_width: int, target_height: int, fit: str, method: str):
        sampler = cls.resolve_sampler(method)
        out = []
        for frame in frames:
            pil = cls.mask_tensor_to_pil(frame)
            resized = cls.fit_resize_mask(
                pil, target_width, target_height, fit, sampler, pil.width, pil.height
            )
            out.append(cls.pil_to_mask_tensor(resized))
        return torch.cat(out, dim=0)

    # ------------------------------------------------------------------
    # 缩放内核（内联自 Image Resize Universal，保持行为一致）
    # ------------------------------------------------------------------

    @staticmethod
    def generate_default_mask(target_width, target_height, fit_mode, orig_width, orig_height):
        """生成默认遮罩。

        - pad：目标尺寸，原图区域白色（255）、填充区域黑色（0）。
        - 其他模式：目标尺寸全白（255）。
        """
        if fit_mode == 'pad' and orig_width > 0 and orig_height > 0:
            orig_ratio = orig_width / orig_height
            target_ratio = target_width / target_height
            mask = Image.new('L', (target_width, target_height), 0)
            if orig_ratio > target_ratio:
                new_width = target_width
                new_height = int(new_width / orig_ratio)
                mask.paste(Image.new('L', (new_width, new_height), 255), (0, (target_height - new_height) // 2))
            else:
                new_height = target_height
                new_width = int(new_height * orig_ratio)
                mask.paste(Image.new('L', (new_width, new_height), 255), ((target_width - new_width) // 2, 0))
            return mask
        return Image.new('L', (target_width, target_height), 255)

    @classmethod
    def fit_resize_mask(cls, mask, target_width, target_height, fit_mode, resize_sampler, orig_width, orig_height):
        """按 fit_mode 缩放单张遮罩（L 模式）。"""
        if fit_mode == 'pad':
            orig_ratio = orig_width / orig_height
            target_ratio = target_width / target_height
            result_mask = Image.new('L', (target_width, target_height), 0)
            if orig_ratio > target_ratio:
                new_width = target_width
                new_height = int(new_width / orig_ratio)
                resized = mask.resize((new_width, new_height), resize_sampler)
                result_mask.paste(resized, (0, (target_height - new_height) // 2))
            else:
                new_height = target_height
                new_width = int(new_height * orig_ratio)
                resized = mask.resize((new_width, new_height), resize_sampler)
                result_mask.paste(resized, ((target_width - new_width) // 2, 0))
            return result_mask
        return cls.fit_resize_image(mask, target_width, target_height, fit_mode, resize_sampler).convert('L')

    @classmethod
    def fit_resize_image(cls, img, target_width, target_height, fit_mode, resize_sampler, background_color=None):
        """按 fit_mode 缩放单张 PIL 图像；pad 模式下支持颜色 / edge / average / extend / mirror。"""
        bg_color = cls.parse_color(background_color) if background_color else (0, 0, 0)
        orig_width, orig_height = img.size
        orig_ratio = orig_width / orig_height
        target_ratio = target_width / target_height

        if fit_mode == 'stretch':
            return img.resize((target_width, target_height), resize_sampler)

        if fit_mode == 'crop':
            if orig_ratio > target_ratio:
                new_height = target_height
                new_width = int(new_height * orig_ratio)
                img = img.resize((new_width, new_height), resize_sampler)
                left = (new_width - target_width) // 2
                return img.crop((left, 0, left + target_width, new_height))
            new_width = target_width
            new_height = int(new_width / orig_ratio)
            img = img.resize((new_width, new_height), resize_sampler)
            top = (new_height - target_height) // 2
            return img.crop((0, top, new_width, top + target_height))

        if fit_mode == 'pad':
            if orig_ratio > target_ratio:
                new_width = target_width
                new_height = int(new_width / orig_ratio)
                img = img.resize((new_width, new_height), resize_sampler)
                pad_top = (target_height - new_height) // 2
                pad_bottom = target_height - new_height - pad_top
                return cls.pad_image(img, target_width, target_height, bg_color, pad_top, pad_bottom, 0, 0)
            new_height = target_height
            new_width = int(new_height * orig_ratio)
            img = img.resize((new_width, new_height), resize_sampler)
            pad_left = (target_width - new_width) // 2
            pad_right = target_width - new_width - pad_left
            return cls.pad_image(img, target_width, target_height, bg_color, 0, 0, pad_left, pad_right)

        return img.resize((target_width, target_height), resize_sampler)

    @classmethod
    def pad_image(cls, img, target_width, target_height, bg_color, pad_top, pad_bottom, pad_left, pad_right):
        """把已按比例缩放的图像放进 target 画布，按 bg_color 决定填充方式。"""
        if bg_color == 'extend':
            arr = np.array(img).astype(np.float32) / 255.0
            t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
            t = F.pad(t, (pad_left, pad_right, pad_top, pad_bottom), mode='replicate')
            return Image.fromarray((t.squeeze(0).permute(1, 2, 0).numpy() * 255.0).astype(np.uint8))

        if bg_color == 'mirror':
            arr = np.array(img).astype(np.float32) / 255.0
            t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
            t = cls.mirror_pad(t, pad_top, pad_bottom, pad_left, pad_right)
            return Image.fromarray((t.squeeze(0).permute(1, 2, 0).numpy() * 255.0).astype(np.uint8))

        if bg_color == 'edge':
            canvas = Image.new('RGB', (target_width, target_height), (0, 0, 0))
            if pad_top > 0:
                canvas.paste(Image.new('RGB', (target_width, pad_top), cls.get_edge_color(img, 'top')), (0, 0))
            if pad_bottom > 0:
                canvas.paste(
                    Image.new('RGB', (target_width, pad_bottom), cls.get_edge_color(img, 'bottom')),
                    (0, target_height - pad_bottom),
                )
            if pad_left > 0:
                canvas.paste(Image.new('RGB', (pad_left, target_height), cls.get_edge_color(img, 'left')), (0, 0))
            if pad_right > 0:
                canvas.paste(
                    Image.new('RGB', (pad_right, target_height), cls.get_edge_color(img, 'right')),
                    (target_width - pad_right, 0),
                )
            canvas.paste(img, (pad_left, pad_top))
            return canvas

        if bg_color == 'average':
            canvas = Image.new('RGB', (target_width, target_height), cls.get_average_color(img))
            canvas.paste(img, (pad_left, pad_top))
            return canvas

        canvas = Image.new('RGB', (target_width, target_height), bg_color)
        canvas.paste(img, (pad_left, pad_top))
        return canvas

    @staticmethod
    def mirror_pad(tensor, pad_top, pad_bottom, pad_left, pad_right):
        """镜像填充 BCHW 张量。

        torch 的 reflect 要求填充量小于该维尺寸，这里分多次填充；当该维只有 1 像素时
        退化为 replicate，避免 Universal 里出现的 0 步长死循环。
        """
        for dim, before, after in ((2, pad_top, pad_bottom), (3, pad_left, pad_right)):
            while before > 0 or after > 0:
                size = int(tensor.shape[dim])
                if size > 1:
                    take_before = min(before, size - 1)
                    take_after = min(after, size - 1)
                else:
                    take_before = take_after = 0
                if take_before == 0 and take_after == 0:
                    spec = (before, after, 0, 0) if dim == 3 else (0, 0, before, after)
                    tensor = F.pad(tensor, spec, mode='replicate')
                    break
                spec = (take_before, take_after, 0, 0) if dim == 3 else (0, 0, take_before, take_after)
                tensor = F.pad(tensor, spec, mode='reflect')
                before -= take_before
                after -= take_after
        return tensor

    @staticmethod
    def parse_color(color_str):
        """解析不同格式的颜色输入（灰度 / RGB / HEX / 颜色名 / edge / average / extend / mirror）。"""
        if not color_str:
            return (255, 255, 255)

        if color_str.lower() in ['edge', 'e']:
            return 'edge'
        if color_str.lower() in ['average', 'a']:
            return 'average'
        if color_str.lower() in ['extend', 'ex']:
            return 'extend'
        if color_str.lower() in ['mirror', 'mr']:
            return 'mirror'

        color_str = color_str.strip()
        if color_str.startswith('(') and color_str.endswith(')'):
            color_str = color_str[1:-1].strip()

        single_letter_colors = {
            'r': 'red', 'g': 'green', 'b': 'blue', 'c': 'cyan', 'm': 'magenta', 'y': 'yellow',
            'k': 'black', 'w': 'white', 'o': 'orange', 'p': 'purple', 'n': 'brown', 's': 'silver',
            'l': 'lime', 'i': 'indigo', 'v': 'violet', 't': 'turquoise', 'q': 'aqua',
            'f': 'fuchsia', 'h': 'hotpink', 'd': 'darkblue',
        }
        if len(color_str) == 1 and color_str.lower() in single_letter_colors:
            color_str = single_letter_colors[color_str.lower()]

        try:
            gray = float(color_str)
            return (int(gray * 255), int(gray * 255), int(gray * 255))
        except ValueError:
            pass

        if ',' in color_str:
            try:
                parts = [part.strip() for part in color_str.split(',')]
                if len(parts) >= 3:
                    r, g, b = [float(parts[i]) for i in range(3)]
                    if max(r, g, b) <= 1.0:
                        return (int(r * 255), int(g * 255), int(b * 255))
                    return (int(r), int(g), int(b))
            except (ValueError, IndexError):
                pass

        try:
            return ImageColor.getrgb(color_str)
        except ValueError:
            return (255, 255, 255)

    @staticmethod
    def get_average_color(img):
        arr = np.array(img.convert('RGB'))
        v = arr.reshape(-1, 3).mean(axis=0)
        return tuple(int(x) for x in v)

    @staticmethod
    def get_edge_color(img, side):
        """获取图像边缘的平均颜色。"""
        width, height = img.size
        img = img.convert('RGB')

        if side == 'left':
            edge = img.crop((0, 0, 1, height))
        elif side == 'right':
            edge = img.crop((width - 1, 0, width, height))
        elif side == 'top':
            edge = img.crop((0, 0, width, 1))
        elif side == 'bottom':
            edge = img.crop((0, height - 1, width, height))
        else:
            top = np.array(img.crop((0, 0, width, 1)))
            bottom = np.array(img.crop((0, height - 1, width, height)))
            left = np.array(img.crop((0, 0, 1, height)))
            right = np.array(img.crop((width - 1, 0, width, height)))
            all_edges = np.vstack([
                top.reshape(-1, 3), bottom.reshape(-1, 3),
                left.reshape(-1, 3), right.reshape(-1, 3),
            ])
            return tuple(np.mean(all_edges, axis=0).astype(int))

        edge_array = np.array(edge)
        return tuple(np.mean(edge_array.reshape(-1, 3), axis=0).astype(int))

    # ------------------------------------------------------------------
    # 张量 / PIL 转换
    # ------------------------------------------------------------------

    @staticmethod
    def tensor_to_pil(image: torch.Tensor) -> Image.Image:
        array = np.clip(255.0 * image.detach().cpu().numpy(), 0, 255).astype(np.uint8)
        if array.ndim == 3 and array.shape[-1] == 1:
            return Image.fromarray(array[..., 0], mode="L")
        return Image.fromarray(array)

    @staticmethod
    def pil_to_tensor(image: Image.Image) -> torch.Tensor:
        array = np.array(image)
        if array.ndim == 2:
            array = array[..., None]
        return torch.from_numpy(array.astype(np.float32) / 255.0).unsqueeze(0)

    @staticmethod
    def mask_tensor_to_pil(mask: torch.Tensor) -> Image.Image:
        array = np.clip(255.0 * mask.detach().cpu().numpy(), 0, 255).astype(np.uint8)
        return Image.fromarray(array, mode="L")

    @staticmethod
    def pil_to_mask_tensor(image: Image.Image) -> torch.Tensor:
        array = np.array(image.convert("L")).astype(np.float32) / 255.0
        return torch.from_numpy(array).unsqueeze(0)

    # ------------------------------------------------------------------
    # 可选超分模型
    # ------------------------------------------------------------------

    @classmethod
    def upscale_with_model(cls, image: torch.Tensor, upscale_model):
        """等价于官方 Upscale Image (using Model)：分块放大并做 OOM 降级。"""
        import comfy.model_management as model_management
        import comfy.utils as comfy_utils

        channels = int(image.shape[-1])
        alpha = None
        if channels == 4:
            alpha = image[..., 3:4]
            rgb = image[..., :3]
        elif channels == 3:
            rgb = image
        elif channels == 1:
            rgb = image.repeat(1, 1, 1, 3)
        else:
            rgb = image[..., :3]

        device = model_management.get_torch_device()
        upscale_model.to(device)
        in_img = rgb.movedim(-1, -3).to(device)

        tile = 512
        overlap = 32
        oom = True
        upscaled = None
        while oom:
            try:
                steps = in_img.shape[0] * comfy_utils.get_tiled_scale_steps(
                    in_img.shape[3], in_img.shape[2], tile_x=tile, tile_y=tile, overlap=overlap
                )
                pbar = comfy_utils.ProgressBar(steps)
                upscaled = comfy_utils.tiled_scale(
                    in_img,
                    lambda a: upscale_model(a),
                    tile_x=tile,
                    tile_y=tile,
                    overlap=overlap,
                    upscale_amount=upscale_model.scale,
                    pbar=pbar,
                )
                oom = False
            except model_management.OOM_EXCEPTION as exc:
                tile //= 2
                if tile < 128:
                    raise exc

        upscale_model.cpu()
        upscaled = torch.clamp(upscaled.movedim(-3, -1), min=0.0, max=1.0)

        if alpha is not None:
            alpha_up = F.interpolate(
                alpha.movedim(-1, 1), size=upscaled.shape[1:3], mode="nearest"
            ).movedim(1, -1)
            alpha_up = alpha_up.to(device=upscaled.device, dtype=upscaled.dtype)
            return torch.cat([upscaled, alpha_up], dim=-1)
        if channels == 1:
            return upscaled[..., :1]
        return upscaled
