"""Shared image helpers for 1hewNodes.

Tensor/PIL/mask conversion, colour parsing, fit-resize and alpha helpers
used by several nodes. This is a shared module, not a node: files under
nodes/<category>/ must define exactly one ComfyNode, so reusable code
lives here (utils/_*.py) instead of inside a node file.
"""

from __future__ import annotations

import math
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageColor


def generate_default_mask(
    target_width, target_height, fit_mode, orig_width, orig_height
):
    """
    生成默认 mask
    - pad 模式：原图区域为白色，填充区域为黑色
    - 其他模式：按模式生成
    """
    if fit_mode == "pad":
        if orig_width <= 0 or orig_height <= 0:
            return Image.new("L", (target_width, target_height), 255)
        # 计算原图在目标尺寸中的位置和大小
        orig_ratio = orig_width / orig_height
        target_ratio = target_width / target_height

        # 创建黑色背景
        mask = Image.new("L", (target_width, target_height), 0)

        if orig_ratio > target_ratio:
            # 原图更宽，按宽度缩放
            new_width = target_width
            new_height = int(new_width / orig_ratio)
            pad_top = (target_height - new_height) // 2

            # 在原图区域填充白色
            white_region = Image.new("L", (new_width, new_height), 255)
            mask.paste(white_region, (0, pad_top))
        else:
            # 原图更高，按高度缩放
            new_height = target_height
            new_width = int(new_height * orig_ratio)
            pad_left = (target_width - new_width) // 2

            # 在原图区域填充白色
            white_region = Image.new("L", (new_width, new_height), 255)
            mask.paste(white_region, (pad_left, 0))

        return mask
    elif fit_mode == "crop":
        if orig_width <= 0 or orig_height <= 0:
            return Image.new("L", (target_width, target_height), 255)
        orig_ratio = orig_width / orig_height
        target_ratio = target_width / target_height
        if orig_ratio > target_ratio:
            scale = target_height / orig_height
            new_width = int(orig_width * scale)
            left_resized = max(0, (new_width - target_width) // 2)
            left_original = int(left_resized / scale)
            crop_w_original = int(target_width / scale)
            crop_w_original = max(0, min(crop_w_original, orig_width))
            if left_original + crop_w_original > orig_width:
                crop_w_original = orig_width - left_original
            mask = Image.new("L", (orig_width, orig_height), 0)
            white_region = Image.new("L", (crop_w_original, orig_height), 255)
            mask.paste(white_region, (left_original, 0))
            return mask
        elif orig_ratio < target_ratio:
            scale = target_width / orig_width
            new_height = int(orig_height * scale)
            top_resized = max(0, (new_height - target_height) // 2)
            top_original = int(top_resized / scale)
            crop_h_original = int(target_height / scale)
            crop_h_original = max(0, min(crop_h_original, orig_height))
            if top_original + crop_h_original > orig_height:
                crop_h_original = orig_height - top_original
            mask = Image.new("L", (orig_width, orig_height), 0)
            white_region = Image.new("L", (orig_width, crop_h_original), 255)
            mask.paste(white_region, (0, top_original))
            return mask
        else:
            return Image.new("L", (orig_width, orig_height), 255)
    else:
        return Image.new("L", (target_width, target_height), 255)


def fit_resize_mask(
    mask, target_width, target_height, fit_mode, resize_sampler, orig_width, orig_height
):
    """
    根据不同适应模式调整 mask 大小
    """
    if fit_mode == "pad":
        # pad 模式下的特殊处理
        orig_ratio = orig_width / orig_height
        target_ratio = target_width / target_height

        # 创建黑色背景
        result_mask = Image.new("L", (target_width, target_height), 0)

        if orig_ratio > target_ratio:
            # 原图更宽，按宽度缩放
            new_width = target_width
            new_height = int(new_width / orig_ratio)
            resized_mask = mask.resize((new_width, new_height), resize_sampler)
            pad_top = (target_height - new_height) // 2
            result_mask.paste(resized_mask, (0, pad_top))
        else:
            # 原图更高，按高度缩放
            new_height = target_height
            new_width = int(new_height * orig_ratio)
            resized_mask = mask.resize((new_width, new_height), resize_sampler)
            pad_left = (target_width - new_width) // 2
            result_mask.paste(resized_mask, (pad_left, 0))

        return result_mask
    else:
        # stretch 和 crop 模式：直接使用原有逻辑
        return fit_resize_image(
            mask, target_width, target_height, fit_mode, resize_sampler
        ).convert("L")


def parse_color(color_str):
    """解析不同格式的颜色输入"""
    if not color_str:
        return (255, 255, 255)

    if color_str.lower() in ["edge", "e"]:
        return "edge"
    if color_str.lower() in ["average", "a"]:
        return "average"
    if color_str.lower() in ["extend", "ex"]:
        return "extend"
    if color_str.lower() in ["mirror", "mr"]:
        return "mirror"

    # 移除括号（如果存在）
    color_str = color_str.strip()
    if color_str.startswith("(") and color_str.endswith(")"):
        color_str = color_str[1:-1].strip()

    # 单字母颜色缩写映射
    single_letter_colors = {
        "r": "red",
        "g": "green",
        "b": "blue",
        "c": "cyan",
        "m": "magenta",
        "y": "yellow",
        "k": "black",
        "w": "white",
        "o": "orange",
        "p": "purple",
        "n": "brown",
        "s": "silver",
        "l": "lime",
        "i": "indigo",
        "v": "violet",
        "t": "turquoise",
        "q": "aqua",
        "f": "fuchsia",
        "h": "hotpink",
        "d": "darkblue",
    }

    # 检查是否为单字母颜色缩写
    if len(color_str) == 1 and color_str.lower() in single_letter_colors:
        color_str = single_letter_colors[color_str.lower()]

    # 尝试解析为灰度值
    try:
        gray = float(color_str)
        return (int(gray * 255), int(gray * 255), int(gray * 255))
    except ValueError:
        pass

    # 尝试解析为 RGB 格式 (如 "0.5,0.7,0.9" 或 "128,192,255")
    if "," in color_str:
        try:
            # 分割并清理每个部分
            parts = [part.strip() for part in color_str.split(",")]
            if len(parts) >= 3:
                r, g, b = [float(parts[i]) for i in range(3)]
                # 判断是否为 0-1 范围
                if max(r, g, b) <= 1.0:
                    return (int(r * 255), int(g * 255), int(b * 255))
                else:
                    return (int(r), int(g), int(b))
        except (ValueError, IndexError):
            pass

    # 尝试解析为十六进制或颜色名称
    try:
        return ImageColor.getrgb(color_str)
    except ValueError:
        return (255, 255, 255)


def get_average_color(img):
    arr = np.array(img.convert("RGB"))
    v = arr.reshape(-1, 3).mean(axis=0)
    return tuple(int(x) for x in v)


def get_edge_color(img, side):
    """获取图像边缘的平均颜色"""
    width, height = img.size
    img = img.convert("RGB")

    if side == "left":
        edge = img.crop((0, 0, 1, height))
    elif side == "right":
        edge = img.crop((width - 1, 0, width, height))
    elif side == "top":
        edge = img.crop((0, 0, width, 1))
    elif side == "bottom":
        edge = img.crop((0, height - 1, width, height))
    else:
        # 所有边缘
        top = np.array(img.crop((0, 0, width, 1)))
        bottom = np.array(img.crop((0, height - 1, width, height)))
        left = np.array(img.crop((0, 0, 1, height)))
        right = np.array(img.crop((width - 1, 0, width, height)))

        # 合并所有边缘像素并计算平均值
        all_edges = np.vstack(
            [
                top.reshape(-1, 3),
                bottom.reshape(-1, 3),
                left.reshape(-1, 3),
                right.reshape(-1, 3),
            ]
        )
        return tuple(np.mean(all_edges, axis=0).astype(int))

    # 计算平均颜色
    edge_array = np.array(edge)
    return tuple(np.mean(edge_array.reshape(-1, 3), axis=0).astype(int))


def fit_resize_image(
    img, target_width, target_height, fit_mode, resize_sampler, background_color=None
):
    """根据不同适应模式调整图像大小"""
    # 解析背景颜色
    bg_color = parse_color(background_color) if background_color else (0, 0, 0)

    # 获取原始尺寸
    orig_width, orig_height = img.size

    # 计算宽高比
    orig_ratio = orig_width / orig_height
    target_ratio = target_width / target_height

    # 根据适应模式处理图像
    if fit_mode == "stretch":  # 直接拉伸
        # 直接拉伸到目标尺寸
        return img.resize((target_width, target_height), resize_sampler)

    elif fit_mode == "crop":  # 裁剪
        # 调整大小保持比例，然后裁剪
        if orig_ratio > target_ratio:
            # 原图更宽，按高度缩放后裁剪宽度
            new_height = target_height
            new_width = int(new_height * orig_ratio)
            img = img.resize((new_width, new_height), resize_sampler)
            # 居中裁剪
            left = (new_width - target_width) // 2
            img = img.crop((left, 0, left + target_width, new_height))
        else:
            # 原图更高，按宽度缩放后裁剪高度
            new_width = target_width
            new_height = int(new_width / orig_ratio)
            img = img.resize((new_width, new_height), resize_sampler)
            # 居中裁剪
            top = (new_height - target_height) // 2
            img = img.crop((0, top, new_width, top + target_height))
        return img

    elif fit_mode == "pad":  # 填充
        # 调整大小保持比例，然后填充
        if orig_ratio > target_ratio:
            # 原图更宽，按宽度缩放后填充高度
            new_width = target_width
            new_height = int(new_width / orig_ratio)
            img = img.resize((new_width, new_height), resize_sampler)

            # 计算需要填充的像素数
            pad_top = (target_height - new_height) // 2
            pad_bottom = target_height - new_height - pad_top

            if bg_color == "extend":
                arr = np.array(img).astype(np.float32) / 255.0
                t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
                padded = F.pad(t, (0, 0, pad_top, pad_bottom), mode="replicate")
                out = (padded.squeeze(0).permute(1, 2, 0).numpy() * 255.0).astype(
                    np.uint8
                )
                return Image.fromarray(out)
            elif bg_color == "mirror":
                arr = np.array(img).astype(np.float32) / 255.0
                t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
                pt = pad_top
                pb = pad_bottom
                while pt > 0 or pb > 0:
                    hh = int(t.shape[2])
                    st = min(pt, max(hh - 1, 0))
                    sb = min(pb, max(hh - 1, 0))
                    t = F.pad(t, (0, 0, st, sb), mode="reflect")
                    pt -= st
                    pb -= sb
                out = (t.squeeze(0).permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
                return Image.fromarray(out)
            if bg_color == "edge":
                top_color = get_edge_color(img, "top")
                bottom_color = get_edge_color(img, "bottom")

                # 创建填充图像
                padded = Image.new("RGB", (target_width, target_height), (0, 0, 0))
                # 填充顶部
                if pad_top > 0:
                    top_pad = Image.new("RGB", (target_width, pad_top), top_color)
                    padded.paste(top_pad, (0, 0))
                # 粘贴原图
                padded.paste(img, (0, pad_top))
                # 填充底部
                if pad_bottom > 0:
                    bottom_pad = Image.new(
                        "RGB", (target_width, pad_bottom), bottom_color
                    )
                    padded.paste(bottom_pad, (0, pad_top + new_height))
            elif bg_color == "average":
                avg_color = get_average_color(img)
                padded = Image.new("RGB", (target_width, target_height), avg_color)
                padded.paste(img, (0, pad_top))
            else:
                padded = Image.new("RGB", (target_width, target_height), bg_color)
                padded.paste(img, (0, pad_top))
        else:
            # 原图更高，按高度缩放后填充宽度
            new_height = target_height
            new_width = int(new_height * orig_ratio)
            img = img.resize((new_width, new_height), resize_sampler)

            # 计算需要填充的像素数
            pad_left = (target_width - new_width) // 2
            pad_right = target_width - new_width - pad_left

            if bg_color == "extend":
                arr = np.array(img).astype(np.float32) / 255.0
                t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
                padded = F.pad(t, (pad_left, pad_right, 0, 0), mode="replicate")
                out = (padded.squeeze(0).permute(1, 2, 0).numpy() * 255.0).astype(
                    np.uint8
                )
                return Image.fromarray(out)
            elif bg_color == "mirror":
                arr = np.array(img).astype(np.float32) / 255.0
                t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
                pl = pad_left
                pr = pad_right
                while pl > 0 or pr > 0:
                    ww = int(t.shape[3])
                    sl = min(pl, max(ww - 1, 0))
                    sr = min(pr, max(ww - 1, 0))
                    t = F.pad(t, (sl, sr, 0, 0), mode="reflect")
                    pl -= sl
                    pr -= sr
                out = (t.squeeze(0).permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
                return Image.fromarray(out)
            if bg_color == "edge":
                left_color = get_edge_color(img, "left")
                right_color = get_edge_color(img, "right")

                # 创建填充图像
                padded = Image.new("RGB", (target_width, target_height), (0, 0, 0))
                # 填充左侧
                if pad_left > 0:
                    left_pad = Image.new("RGB", (pad_left, target_height), left_color)
                    padded.paste(left_pad, (0, 0))
                # 粘贴原图
                padded.paste(img, (pad_left, 0))
                # 填充右侧
                if pad_right > 0:
                    right_pad = Image.new(
                        "RGB", (pad_right, target_height), right_color
                    )
                    padded.paste(right_pad, (pad_left + new_width, 0))
            elif bg_color == "average":
                avg_color = get_average_color(img)
                padded = Image.new("RGB", (target_width, target_height), avg_color)
                padded.paste(img, (pad_left, 0))
            else:
                padded = Image.new("RGB", (target_width, target_height), bg_color)
                padded.paste(img, (pad_left, 0))

        return padded

    # 默认情况下直接调整大小
    return img.resize((target_width, target_height), resize_sampler)


def tensor2pil(image):
    """将张量转换为PIL图像"""
    return Image.fromarray(
        np.clip(255.0 * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)
    )


def pil2tensor(image):
    """将PIL图像转换为张量"""
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)


def image2mask(image):
    """将图像转换为遮罩"""
    return torch.from_numpy(
        np.array(image.convert("L")).astype(np.float32) / 255.0
    ).unsqueeze(0)


def is_valid_mask(mask):
    """检查遮罩是否有效"""
    if mask is None:
        return False
    return torch.min(mask) < 0.9999


def num_round_up_to_multiple(num, multiple):
    """将数字向上取整到指定倍数"""
    return math.ceil(num / multiple) * multiple


def log(message, message_type="info"):
    """输出日志信息"""
    print(f"[{message_type.upper()}] {message}")


def adapt_fill_to_channels(fill_rgb, channels):
    if isinstance(fill_rgb, str):
        return fill_rgb
    if not isinstance(fill_rgb, (tuple, list)):
        return fill_rgb
    values = list(fill_rgb)
    if channels == 4:
        if len(values) >= 4:
            return tuple(values[:4])
        if len(values) == 3:
            return (values[0], values[1], values[2], 1.0)
    if channels == 3 and len(values) >= 3:
        return tuple(values[:3])
    return tuple(values)


def ensure_channels(img, target_channels):
    current_channels = int(img.shape[3])
    if current_channels == target_channels:
        return img
    if current_channels == 3 and target_channels == 4:
        alpha = torch.ones(
            (img.shape[0], img.shape[1], img.shape[2], 1),
            dtype=img.dtype,
            device=img.device,
        )
        return torch.cat([img, alpha], dim=3)
    if current_channels == 4 and target_channels == 3:
        return img[:, :, :, :3]
    return img


def parse_spacing_color(color_str):
    if color_str is None:
        return (1.0, 1.0, 1.0)
    text = str(color_str).strip().lower()
    if text.startswith("(") and text.endswith(")"):
        text = text[1:-1].strip()
    single = {
        "r": "red",
        "g": "lime",
        "b": "blue",
        "c": "cyan",
        "m": "magenta",
        "y": "yellow",
        "k": "black",
        "w": "white",
        "o": "orange",
        "p": "purple",
        "n": "brown",
        "s": "silver",
        "l": "lime",
        "i": "indigo",
        "v": "violet",
        "t": "turquoise",
        "f": "fuchsia",
        "h": "hotpink",
        "d": "darkblue",
    }
    if len(text) == 1 and text in single:
        text = single[text]
    try:
        v = float(text)
        if 0.0 <= v <= 1.0:
            return (v, v, v)
    except Exception:
        pass
    if "," in text:
        try:
            parts = [p.strip() for p in text.split(",")]
            if len(parts) >= 3:
                r, g, b = [float(parts[i]) for i in range(3)]
                if max(r, g, b) <= 1.0:
                    return (r, g, b)
                return (r / 255.0, g / 255.0, b / 255.0)
        except Exception:
            pass
    if text.startswith("#") and len(text) in (4, 7):
        try:
            hex_str = text[1:]
            if len(hex_str) == 3:
                hex_str = "".join(ch * 2 for ch in hex_str)
            r = int(hex_str[0:2], 16) / 255.0
            g = int(hex_str[2:4], 16) / 255.0
            b = int(hex_str[4:6], 16) / 255.0
            return (r, g, b)
        except Exception:
            pass
    try:
        rgb = ImageColor.getrgb(text)
        return (rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0)
    except Exception:
        return (1.0, 1.0, 1.0)


def parse_pad_color(color_str):
    if color_str is None:
        return (1.0, 1.0, 1.0)
    text = str(color_str).strip().lower()
    if text in ("edge", "e"):
        return "edge"
    if text in ("extend", "ex"):
        return "extend"
    if text in ("mirror", "mr"):
        return "mirror"
    if text in ("average", "a"):
        return "average"
    if text.startswith("(") and text.endswith(")"):
        text = text[1:-1].strip()
    single = {
        "r": "red",
        "g": "lime",
        "b": "blue",
        "c": "cyan",
        "m": "magenta",
        "y": "yellow",
        "k": "black",
        "w": "white",
        "o": "orange",
        "p": "purple",
        "n": "brown",
        "s": "silver",
        "l": "lime",
        "i": "indigo",
        "v": "violet",
        "t": "turquoise",
        "q": "aqua",
        "f": "fuchsia",
        "h": "hotpink",
        "d": "darkblue",
    }
    if len(text) == 1 and text in single:
        text = single[text]
    try:
        v = float(text)
        if 0.0 <= v <= 1.0:
            return (v, v, v)
    except Exception:
        pass
    if "," in text:
        try:
            parts = [p.strip() for p in text.split(",")]
            if len(parts) >= 3:
                r, g, b = [float(parts[i]) for i in range(3)]
                if max(r, g, b) <= 1.0:
                    return (r, g, b)
                return (r / 255.0, g / 255.0, b / 255.0)
        except Exception:
            pass
    if text.startswith("#") and len(text) in (4, 7):
        try:
            hex_str = text[1:]
            if len(hex_str) == 3:
                hex_str = "".join(ch * 2 for ch in hex_str)
            r = int(hex_str[0:2], 16) / 255.0
            g = int(hex_str[2:4], 16) / 255.0
            b = int(hex_str[4:6], 16) / 255.0
            return (r, g, b)
        except Exception:
            pass
    try:
        from PIL import ImageColor

        rgb = ImageColor.getrgb(text)
        return (rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0)
    except Exception:
        return (1.0, 1.0, 1.0)


def pad_to_rgb(img, target_h, target_w, fill_rgb):
    """居中填充到指定尺寸，自动适配当前图像通道数。"""
    b, h, w, c = img.shape
    top = max((target_h - h) // 2, 0)
    left = max((target_w - w) // 2, 0)
    h_end = min(top + h, target_h)
    w_end = min(left + w, target_w)
    if isinstance(fill_rgb, str) and fill_rgb == "extend":
        pad_top = top
        pad_bottom = max(target_h - (top + h), 0)
        pad_left = left
        pad_right = max(target_w - (left + w), 0)
        nchw = img.permute(0, 3, 1, 2)
        padded = F.pad(
            nchw, (pad_left, pad_right, pad_top, pad_bottom), mode="replicate"
        )
        return padded.permute(0, 2, 3, 1)
    if isinstance(fill_rgb, str) and fill_rgb == "mirror":
        pad_top = top
        pad_bottom = max(target_h - (top + h), 0)
        pad_left = left
        pad_right = max(target_w - (left + w), 0)
        nchw = img.permute(0, 3, 1, 2)
        while pad_top > 0 or pad_bottom > 0 or pad_left > 0 or pad_right > 0:
            hh = int(nchw.shape[2])
            ww = int(nchw.shape[3])
            st = min(pad_top, max(hh - 1, 0))
            sb = min(pad_bottom, max(hh - 1, 0))
            sl = min(pad_left, max(ww - 1, 0))
            sr = min(pad_right, max(ww - 1, 0))
            nchw = F.pad(nchw, (sl, sr, st, sb), mode="reflect")
            pad_top -= st
            pad_bottom -= sb
            pad_left -= sl
            pad_right -= sr
        return nchw.permute(0, 2, 3, 1)
    out = torch.zeros((b, target_h, target_w, c), dtype=img.dtype, device=img.device)
    if isinstance(fill_rgb, str) and fill_rgb == "edge":
        if top > 0 or (target_h - h_end) > 0:
            top_row = img[:, 0:1, :, :].mean(dim=(1, 2))
            top_color = top_row.view(b, 1, 1, c)
            bottom_row = img[:, -1:, :, :].mean(dim=(1, 2))
            bottom_color = bottom_row.view(b, 1, 1, c)
            if top > 0:
                out[:, :top, :, :] = top_color.expand(b, top, target_w, c)
            if (target_h - h_end) > 0:
                bottom_pad = target_h - h_end
                out[:, h_end:, :, :] = bottom_color.expand(b, bottom_pad, target_w, c)
        if left > 0 or (target_w - w_end) > 0:
            left_col = img[:, :, 0:1, :].mean(dim=(1, 2))
            left_color = left_col.view(b, 1, 1, c)
            right_col = img[:, :, -1:, :].mean(dim=(1, 2))
            right_color = right_col.view(b, 1, 1, c)
            if left > 0:
                out[:, :, :left, :] = left_color.expand(b, target_h, left, c)
            if (target_w - w_end) > 0:
                right_pad = target_w - w_end
                out[:, :, w_end:, :] = right_color.expand(b, target_h, right_pad, c)
    elif isinstance(fill_rgb, str) and fill_rgb == "average":
        avg = img.mean(dim=(1, 2))
        avg_color = avg.view(b, 1, 1, c)
        out[:] = avg_color.expand(b, target_h, target_w, c)
    else:
        fill_value = adapt_fill_to_channels(fill_rgb, c)
        fill_t = torch.tensor(fill_value, dtype=img.dtype, device=img.device)
        out[:] = fill_t
    out[:, top:h_end, left:w_end, :] = img[:, : h_end - top, : w_end - left, :]
    return out


def remove_small_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    if min_area <= 1:
        return mask
    h, w = int(mask.shape[0]), int(mask.shape[1])
    visited = np.zeros((h, w), dtype=bool)
    keep = np.zeros((h, w), dtype=bool)
    neighbors = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

    ys, xs = np.nonzero(mask)
    for y0, x0 in zip(ys.tolist(), xs.tolist()):
        if visited[y0, x0]:
            continue
        visited[y0, x0] = True
        q: deque[tuple[int, int]] = deque()
        q.append((y0, x0))
        comp: list[tuple[int, int]] = [(y0, x0)]

        while q:
            y, x = q.popleft()
            for dy, dx in neighbors:
                ny = y + dy
                nx = x + dx
                if ny < 0 or ny >= h or nx < 0 or nx >= w:
                    continue
                if visited[ny, nx] or (not mask[ny, nx]):
                    continue
                visited[ny, nx] = True
                q.append((ny, nx))
                comp.append((ny, nx))

        if len(comp) >= int(min_area):
            for cy, cx in comp:
                keep[cy, cx] = True
    return keep


def apply_levels_numpy(
        image_array: np.ndarray,
        black_point: int,
        white_point: int,
        gray_point: float = 1.0,
        output_black_point: int = 0,
        output_white_point: int = 255,
    ) -> np.ndarray:
        bp = int(max(0, min(255, black_point)))
        wp = int(max(0, min(255, white_point)))
        obp = int(max(0, min(255, output_black_point)))
        owp = int(max(0, min(255, output_white_point)))
        result = np.zeros_like(image_array)
        for i in range(image_array.shape[2]):
            ch = image_array[:, :, i].astype(np.float32)
            ch = np.clip(ch, bp, wp)
            denom = max(1, (wp - bp))
            ch = (ch - bp) / float(denom) * 255.0
            if gray_point != 1.0:
                ch = 255.0 * (ch / 255.0) ** (1.0 / float(gray_point))
            ch = (ch / 255.0) * (owp - obp) + obp
            result[:, :, i] = np.clip(ch, 0, 255).astype(np.uint8)
        return result
