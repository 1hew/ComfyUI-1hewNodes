import inspect
import math
import torch
import torch.nn.functional as F


class MultiStringJoin:
    """
    动态字符串连接节点：支持 string_X 可变输入，连接到最后一项自动追加。
    仅保留一个尾部空槽，断开中间连接时将空槽整理到末尾。
    - filter_empty_line：是否去除空行。
    - filter_comment：是否过滤行内注释与三引号注释块。
    """

    @classmethod
    def INPUT_TYPES(cls):
        # 默认提供一个下划线风格的首端口 string_1
        opt_inputs = {
            "string_1": ("STRING",{"forceInput": True, "multiline": True}),
            }

        try:
            stack = inspect.stack()
            if len(stack) > 2 and stack[2].function == "get_input_info":
                class AllContainer:
                    def __contains__(self, item):
                        return True

                    def __getitem__(self, key):
                        k = str(key)
                        if k.startswith("string_"):
                            return "STRING", {"forceInput": True, "multiline": True}
                        if k == "input":
                            return "STRING", {"default": ""}
                        return "STRING", {"forceInput": True}

                opt_inputs = AllContainer()
        except Exception:
            pass

        optional = opt_inputs if not isinstance(opt_inputs, dict) else {
            "input": ("STRING", {"default": ""}),
            **opt_inputs,
        }

        return {
            "required": {
                "filter_empty_line": ("BOOLEAN", {"default": False}),
                "filter_comment": ("BOOLEAN", {"default": False}),
                "separator": ("STRING", {"default": "\\n"}),
            },
            "optional": optional,
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("string",)
    FUNCTION = "multi_string_join"
    CATEGORY = "1hewNodes/multi"

    def _parse_text_with_input(
            self,
            text,
            input_text,
            filter_comment=False,
            filter_empty_line=False):
        # 处理 {input} 引用
        safe_input = "" if input_text is None else str(input_text)
        parsed = "" if text is None else str(text)
        parsed = parsed.replace("{input}", safe_input)

        lines = parsed.split("\n")
        result = []
        in_block = False
        block_quote = None
        for line in lines:
            original = line
            processed = ""

            if filter_comment:
                i = 0
                while i < len(line):
                    if in_block:
                        end_pos = line.find(block_quote, i)
                        if end_pos != -1:
                            i = end_pos + len(block_quote)
                            in_block = False
                            block_quote = None
                        else:
                            break
                    else:
                        is_triple = (
                            i + 2 < len(line) and (
                                line[i:i+3] == '"""' or
                                line[i:i+3] == "'''"
                            )
                        )
                        if is_triple:
                            block_quote = line[i:i+3]
                            end_pos = line.find(block_quote, i + 3)
                            if end_pos != -1:
                                i = end_pos + 3
                            else:
                                in_block = True
                                break
                        elif line[i] == '#':
                            break
                        else:
                            processed += line[i]
                            i += 1
            else:
                processed = original

            if not in_block:
                processed = processed.rstrip()

                # 当开启过滤注释时：
                # 1) 以注释开头的整行直接跳过；
                # 2) 因过滤注释而变成空行的当前行也跳过（不受 filter_empty_line 影响）。
                if filter_comment:
                    starts_with_comment = original.lstrip().startswith('#')
                    became_empty_by_filter = (
                        processed.strip() == "" and original.strip() != "" and (
                            '#' in original or
                            original.strip().startswith('"""') or
                            original.strip().startswith("'''")
                        )
                    )
                    if starts_with_comment or became_empty_by_filter:
                        continue

                if filter_empty_line:
                    if processed.strip():
                        result.append(processed)
                else:
                    result.append(processed)

        final_text = "\n".join(result)
        return "" if not final_text.strip() else final_text

    def multi_string_join(
            self,
            separator="\n",
            input="",
            filter_comment=True,
            filter_empty_line=True,
            **kwargs):
        try:
            # 当分隔符为空字符串时，使用无分隔直接拼接
            sep = "" if separator is None else str(separator)
            sep = sep.replace("\\n", "\n").replace("\\t", "\t").replace("\\r", "\r")

            # 按序收集 string_X（仅下划线风格）
            ordered = []
            for k in kwargs.keys():
                if k.startswith("string_"):
                    suf = k[len("string_"):]
                    if suf.isdigit():
                        ordered.append((int(suf), k))
            ordered.sort(key=lambda x: x[0])

            parts = []
            for _, key in ordered:
                val = kwargs.get(key)
                if val is None:
                    continue
                parsed = self._parse_text_with_input(
                    val,
                    input,
                    filter_comment,
                    filter_empty_line,
                )
                if parsed.strip():
                    parts.append(parsed)

            result = sep.join(parts)
            return (str(result),)
        except Exception as e:
            print(f"MultiStringJoin error: {e}")
            return ("",)


class MultiImageBatch:
    """
    通过动态 image_X 输入构建图像批次，尾部保持一个空槽。
    - 自动按序收集 image_1..image_N。
    - 支持 fit 模式：'stretch'、'crop'、'pad'。
      * stretch：等比/非等比缩放到基准尺寸（bicubic）。
      * crop：超出部分居中裁剪到统一尺寸；不足部分不缩放。
      * pad：统一到最大画布，居中填充到统一尺寸（不裁剪、不缩放）。
    """

    @classmethod
    def INPUT_TYPES(cls):
        opt_inputs = {
            "image_1": ("IMAGE", {"forceInput": True}),
        }

        try:
            stack = inspect.stack()
            if len(stack) > 2 and stack[2].function == "get_input_info":
                class AllContainer:
                    def __contains__(self, item):
                        return True

                    def __getitem__(self, key):
                        k = str(key)
                        if k.startswith("image_"):
                            return "IMAGE", {"forceInput": True}
                        return "IMAGE", {"forceInput": True}

                opt_inputs = AllContainer()
        except Exception:
            pass

        optional = opt_inputs

        return {
            "required": {
                "fit": (['crop', 'pad', 'stretch'], {"default": "pad"}),
                "pad_color": ("STRING", {"default": "1.0"}),
            },
            "optional": optional,
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "combine"
    CATEGORY = "1hewNodes/multi"
    DESCRIPTION = (
        "将多个动态 image_X 输入合并为批次，并按 fit 模式统一尺寸"
    )

    def combine(self, fit="pad", pad_color="1.0", **kwargs):
        # 收集 image_X（以 image_1 为基准）
        ordered = []
        for k in kwargs.keys():
            if k.startswith("image_"):
                suf = k[len("image_") :]
                if suf.isdigit():
                    ordered.append((int(suf), k))
        ordered.sort(key=lambda x: x[0])

        images = []
        for _, key in ordered:
            val = kwargs.get(key)
            if val is None:
                continue
            images.append(val.cpu())

        if not images:
            # 至少需要 image_1；若缺失则返回空批次以避免崩溃
            return (torch.zeros((0, 64, 64, 3), dtype=torch.float32),)

        # 基准尺寸：以 image_1 的尺寸为参照
        ref_h = int(images[0].shape[1])
        ref_w = int(images[0].shape[2])

        # 解析填充颜色
        pad_rgb = self._parse_color_string(pad_color)

        aligned = []
        for img in images:
            h = int(img.shape[1])
            w = int(img.shape[2])
            if fit == "stretch":
                # 直接拉伸到基准尺寸（可能非等比）
                aligned.append(self._resize_to(img, ref_h, ref_w))
            elif fit == "crop":
                # cover：按更大的比例缩放，确保覆盖参照画布；再居中裁剪
                sw = ref_w / max(w, 1)
                sh = ref_h / max(h, 1)
                scale = max(sw, sh)
                new_w = max(int(round(w * scale)), 1)
                new_h = max(int(round(h * scale)), 1)
                resized = self._resize_to(img, new_h, new_w)
                aligned.append(self._center_crop(resized, ref_h, ref_w))
            else:  # pad
                # contain：按更小比例缩放以完全容纳在参照画布；再居中填充
                sw = ref_w / max(w, 1)
                sh = ref_h / max(h, 1)
                scale = min(sw, sh)
                new_w = max(int(round(w * scale)), 1)
                new_h = max(int(round(h * scale)), 1)
                resized = self._resize_to(img, new_h, new_w)
                aligned.append(self._pad_to_rgb(resized, ref_h, ref_w, pad_rgb))

        result = (
            torch.cat(aligned, dim=0) if len(aligned) > 1 else aligned[0]
        )
        result = torch.clamp(result, min=0.0, max=1.0).to(torch.float32)
        return (result,)

    def _parse_color_string(self, color_str):
        """解析颜色字符串。
        支持：
        - 灰度/颜色字符串，复用 ImageSolid.parse_color -> 返回 (r, g, b) in [0,1]
        - 特殊值："edge"/"e" -> 返回字符串标记 'edge'，用于边缘取色填充
        """
        if color_str is None:
            return (1.0, 1.0, 1.0)

        text = str(color_str).strip().lower()
        if text in ("edge", "e"):
            return "edge"

        try:
            from .image import ImageSolid
            parser = ImageSolid()
            rgb = parser.parse_color(color_str)
            r = rgb[0] / 255.0
            g = rgb[1] / 255.0
            b = rgb[2] / 255.0
            return (r, g, b)
        except Exception:
            return (1.0, 1.0, 1.0)

    def _resize_to(self, img, target_h, target_w):
        b, h, w, c = img.shape
        img_nchw = img.permute(0, 3, 1, 2)
        out = torch.nn.functional.interpolate(
            img_nchw,
            size=(target_h, target_w),
            mode="bicubic",
            align_corners=False,
        ).permute(0, 2, 3, 1)
        return out

    def _pad_to_rgb(self, img, target_h, target_w, fill_rgb):
        """居中填充图像到目标尺寸。
        当 fill_rgb 为 'edge' 时，按 ImageResizeUniversal 的语义：
        - 垂直方向存在填充：顶部用顶部边缘平均色、底部用底部边缘平均色；
        - 水平方向存在填充：左侧用左边缘平均色、右侧用右边缘平均色；
        其他情况使用统一颜色填充。
        """
        b, h, w, c = img.shape
        out = torch.zeros(
            (b, target_h, target_w, c), dtype=img.dtype, device=img.device
        )

        top = max((target_h - h) // 2, 0)
        left = max((target_w - w) // 2, 0)
        h_end = min(top + h, target_h)
        w_end = min(left + w, target_w)

        if isinstance(fill_rgb, str) and fill_rgb == "edge":
            # 计算边缘平均颜色（0..1 RGB）
            # 顶/底边缘
            if top > 0 or (target_h - h_end) > 0:
                # 顶部一行平均色
                top_row = img[:, 0:1, :, :].mean(dim=(1, 2))  # (b, 3)
                top_color = top_row.view(b, 1, 1, c)
                # 底部一行平均色
                bottom_row = img[:, -1:, :, :].mean(dim=(1, 2))  # (b, 3)
                bottom_color = bottom_row.view(b, 1, 1, c)
                if top > 0:
                    out[:, :top, :, :] = top_color.expand(b, top, target_w, c)
                if (target_h - h_end) > 0:
                    bottom_pad = target_h - h_end
                    out[:, h_end:, :, :] = bottom_color.expand(
                        b, bottom_pad, target_w, c
                    )

            # 左/右边缘
            if left > 0 or (target_w - w_end) > 0:
                # 左侧一列平均色
                left_col = img[:, :, 0:1, :].mean(dim=(1, 2))  # (b, 3)
                left_color = left_col.view(b, 1, 1, c)
                # 右侧一列平均色
                right_col = img[:, :, -1:, :].mean(dim=(1, 2))  # (b, 3)
                right_color = right_col.view(b, 1, 1, c)
                if left > 0:
                    out[:, :, :left, :] = left_color.expand(b, target_h, left, c)
                if (target_w - w_end) > 0:
                    right_pad = target_w - w_end
                    out[:, :, w_end:, :] = right_color.expand(
                        b, target_h, right_pad, c
                    )
        else:
            fill_t = torch.tensor(fill_rgb, dtype=img.dtype, device=img.device)
            out[:] = fill_t

        # 粘贴原图（居中）
        out[:, top:h_end, left:w_end, :] = img[:, : h_end - top, : w_end - left, :]
        return out

    def _center_crop(self, img, target_h, target_w):
        b, h, w, c = img.shape
        top = max((h - target_h) // 2, 0)
        left = max((w - target_w) // 2, 0)
        top_end = min(top + target_h, h)
        left_end = min(left + target_w, w)
        cropped = img[:, top:top_end, left:left_end, :]
        # 若原图比目标更小（不常见），做居中填充以达成统一尺寸
        if cropped.shape[1] != target_h or cropped.shape[2] != target_w:
            pad_rgb = (0.0, 0.0, 0.0)
            cropped = self._pad_to_rgb(cropped, target_h, target_w, pad_rgb)
        return cropped


class MultiMaskBatch:
    """
    通过动态 mask_X 输入构建掩码批次，尾部保持一个空槽。
    - 自动按序收集 mask_1..mask_N。
    - 支持 fit 模式：'stretch'、'crop'、'pad'。
      * stretch：缩放到基准尺寸（bilinear）。
      * crop：居中裁剪到统一尺寸。
      * pad：统一到最大画布，居中填充到统一尺寸（不裁剪、不缩放）。
    """

    @classmethod
    def INPUT_TYPES(cls):
        opt_inputs = {
            "mask_1": ("MASK", {"forceInput": True}),
        }

        try:
            stack = inspect.stack()
            if len(stack) > 2 and stack[2].function == "get_input_info":
                class AllContainer:
                    def __contains__(self, item):
                        return True

                    def __getitem__(self, key):
                        k = str(key)
                        if k.startswith("mask_"):
                            return "MASK", {"forceInput": True}
                        return "MASK", {"forceInput": True}

                opt_inputs = AllContainer()
        except Exception:
            pass

        optional = opt_inputs

        return {
            "required": {
                "fit": (['crop', 'pad', 'stretch'], {"default": "pad"}),
                "pad_color": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01
                }),
            },
            "optional": optional,
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("masks",)
    FUNCTION = "combine"
    CATEGORY = "1hewNodes/multi"
    DESCRIPTION = (
        "将多个动态 mask_X 输入合并为批次，并按 fit 模式统一尺寸"
    )


    def combine(self, fit="pad", pad_color=0.0, **kwargs):
        # 收集 mask_X（以 mask_1 为基准）
        ordered = []
        for k in kwargs.keys():
            if k.startswith("mask_"):
                suf = k[len("mask_") :]
                if suf.isdigit():
                    ordered.append((int(suf), k))
        ordered.sort(key=lambda x: x[0])

        masks = []
        for _, key in ordered:
            val = kwargs.get(key)
            if val is None:
                continue
            masks.append(val.cpu())

        if not masks:
            return (torch.zeros((0, 64, 64), dtype=torch.float32),)

        # 基准尺寸：以 mask_1 的尺寸为参照
        ref_h = int(masks[0].shape[1])
        ref_w = int(masks[0].shape[2])

        # 解析填充颜色（浮点 0..1 灰度）
        pad_value = float(pad_color)

        aligned = []
        for m in masks:
            h = int(m.shape[1])
            w = int(m.shape[2])
            if fit == "stretch":
                # 直接拉伸到基准尺寸
                out = F.interpolate(
                    m.unsqueeze(1),
                    size=(ref_h, ref_w),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(1)
                aligned.append(out)
            elif fit == "crop":
                # cover：按更大的比例缩放，确保覆盖参照画布；再居中裁剪
                sw = ref_w / max(w, 1)
                sh = ref_h / max(h, 1)
                scale = max(sw, sh)
                new_w = max(int(round(w * scale)), 1)
                new_h = max(int(round(h * scale)), 1)
                resized = F.interpolate(
                    m.unsqueeze(1),
                    size=(new_h, new_w),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(1)
                aligned.append(self._center_crop_mask(resized, ref_h, ref_w))
            else:  # pad
                # contain：按更小比例缩放以完全容纳在参照画布；再居中填充
                sw = ref_w / max(w, 1)
                sh = ref_h / max(h, 1)
                scale = min(sw, sh)
                new_w = max(int(round(w * scale)), 1)
                new_h = max(int(round(h * scale)), 1)
                resized = F.interpolate(
                    m.unsqueeze(1),
                    size=(new_h, new_w),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(1)
                aligned.append(self._pad_mask(resized, ref_h, ref_w, pad_value))

        result = (
            torch.cat(aligned, dim=0) if len(aligned) > 1 else aligned[0]
        )
        result = torch.clamp(result, min=0.0, max=1.0).to(torch.float32)
        return (result,)

    def _pad_mask(self, mask, target_h, target_w, value):
        b, h, w = mask.shape
        out = torch.zeros(
            (b, target_h, target_w), dtype=mask.dtype, device=mask.device
        )
        out[:] = float(value)
        top = max((target_h - h) // 2, 0)
        left = max((target_w - w) // 2, 0)
        h_end = min(top + h, target_h)
        w_end = min(left + w, target_w)
        out[:, top:h_end, left:w_end] = mask[:, : h_end - top, : w_end - left]
        return out

    def _center_crop_mask(self, mask, target_h, target_w):
        b, h, w = mask.shape
        top = max((h - target_h) // 2, 0)
        left = max((w - target_w) // 2, 0)
        top_end = min(top + target_h, h)
        left_end = min(left + target_w, w)
        cropped = mask[:, top:top_end, left:left_end]
        if cropped.shape[1] != target_h or cropped.shape[2] != target_w:
            cropped = self._pad_mask(cropped, target_h, target_w, 0.0)
        return cropped


class MultiImageStitch:
    """
    动态多图像拼接：按序将 image_1..image_N 进行缝合。
    - 支持方向（top/bottom/left/right）、匹配尺寸、间距宽度与颜色。
    - 当不匹配尺寸时，按方向维度进行居中填充（不拉伸）。
    """

    @classmethod
    def INPUT_TYPES(cls):
        opt_inputs = {
            "image_1": ("IMAGE", {"forceInput": True}),
            "image_2": ("IMAGE", {"forceInput": True}),
        }

        try:
            stack = inspect.stack()
            if len(stack) > 2 and stack[2].function == "get_input_info":
                class AllContainer:
                    def __contains__(self, item):
                        return True

                    def __getitem__(self, key):
                        k = str(key)
                        if k.startswith("image_"):
                            return "IMAGE", {"forceInput": True}
                        return "IMAGE", {"forceInput": True}

                opt_inputs = AllContainer()
        except Exception:
            pass

        return {
            "required": {
                "direction": (["top", "bottom", "left", "right"], {"default": "right"}),
                "match_image_size": ("BOOLEAN", {"default": True}),
                "spacing_width": ("INT", {"default": 10, "min": 0, "max": 1000, "step": 1}),
                "spacing_color": ("STRING", {"default": "1.0"}),
                "pad_color": ("STRING", {"default": "1.0"}),
            },
            "optional": opt_inputs,
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "stitch"
    CATEGORY = "1hewNodes/multi"

    def stitch(
        self,
        direction,
        match_image_size,
        spacing_width,
        spacing_color,
        pad_color,
        **kwargs,
    ):
        ordered = []
        for key in kwargs.keys():
            if key.startswith("image_"):
                suf = key[len("image_") :]
                if suf.isdigit():
                    ordered.append((int(suf), key))
        ordered.sort(key=lambda x: x[0])

        images = []
        for _, key in ordered:
            val = kwargs.get(key)
            if val is None:
                continue
            images.append(val)

        if not images:
            # 返回空白图避免崩溃
            return (torch.zeros((0, 64, 64, 3), dtype=torch.float32),)

        if len(images) == 1:
            image = torch.clamp(images[0], min=0.0, max=1.0).to(torch.float32)
            return (image,)

        current = images[0].cpu()
        for img in images[1:]:
            next_img = (
                img.cpu() if img is not None else current.new_zeros(current.shape)
            )

            # 对齐批次尺寸（广播到最大批次）
            bs = max(current.shape[0], next_img.shape[0])
            current = self._broadcast_image(current, bs)
            next_img = self._broadcast_image(next_img, bs)

            current = (
                self._stitch_pair(
                    current,
                    next_img,
                    direction,
                    match_image_size,
                    spacing_width,
                    spacing_color,
                    pad_color,
                )
            ).cpu()

        image = torch.clamp(current, min=0.0, max=1.0).to(torch.float32)
        return (image,)

    # 辅助：将两张图按原生语义进行拼接
    def _stitch_pair(
        self,
        a,
        b,
        direction,
        match_image_size,
        spacing_width,
        spacing_color,
        pad_color,
    ):
        space_rgb = self._parse_color_string(spacing_color)
        pad_rgb = self._parse_color_string(pad_color)

        # 保障批量维度一致
        bs = max(a.shape[0], b.shape[0])
        a = self._broadcast_image(a, bs)
        b = self._broadcast_image(b, bs)

        _, ha, wa, _ = a.shape
        _, hb, wb, _ = b.shape

        if direction in ("left", "right"):
            # 水平拼接：沿高度轴处理
            if match_image_size:
                # 与原生一致：以当前图 a 的高度为基准，按比例缩放
                target_h = ha
                a = self._resize_keep_ratio(a, None, target_h)
                b = self._resize_keep_ratio(b, None, target_h)
                # 不再统一宽度，直接并排拼接
            else:
                # 不匹配尺寸：统一到更大的高度，使用 pad_color 居中填充
                unified_h = max(ha, hb)
                a = self._pad_to_rgb(a, unified_h, wa, pad_rgb)
                b = self._pad_to_rgb(b, unified_h, wb, pad_rgb)

            # 构造垂直间距条（高度与当前 a 的最终高度一致）
            _, ha2, wa2, _ = a.shape
            spacer = self._make_strip(
                ha2,
                spacing_width,
                space_rgb,
                axis="v",
                dtype=a.dtype,
                device=a.device,
                batch_size=bs,
            )

            if direction == "right":
                out = torch.cat([a, spacer, b], dim=2)
            else:  # left
                out = torch.cat([b, spacer, a], dim=2)

            return out

        # 垂直拼接：统一宽度
        if match_image_size:
            # 与原生一致：以当前图 a 的宽度为基准，按比例缩放
            target_w = wa
            a = self._resize_keep_ratio(a, target_w, None)
            b = self._resize_keep_ratio(b, target_w, None)
            # 不再统一高度，直接上下堆叠
        else:
            # 不匹配尺寸：统一到更大的宽度，使用 pad_color 居中填充
            unified_w = max(wa, wb)
            a = self._pad_to_rgb(a, ha, unified_w, pad_rgb)
            b = self._pad_to_rgb(b, hb, unified_w, pad_rgb)

        _, ha2, wa2, _ = a.shape
        spacer = self._make_strip(
            spacing_width,
            wa2,
            space_rgb,
            axis="h",
            dtype=a.dtype,
            device=a.device,
            batch_size=bs,
        )

        if direction == "bottom":
            out = torch.cat([a, spacer, b], dim=1)
        else:  # top
            out = torch.cat([b, spacer, a], dim=1)

        return out

    def _parse_color_string(self, color_str):
        """
        解析任意颜色字符串为 0..1 RGB，
        支持灰度、RGB(0..1/0..255)、十六进制、颜色名与单字母。
        """
        try:
            from .image import ImageSolid  # 延迟导入避免循环依赖
            parser = ImageSolid()
            rgb = parser.parse_color(color_str)
            r = rgb[0] / 255.0
            g = rgb[1] / 255.0
            b = rgb[2] / 255.0
            return (r, g, b)
        except Exception:
            return (1.0, 1.0, 1.0)

    def _resize_keep_ratio(self, img, target_w, target_h):
        """按目标宽或高等比缩放，空参数保持原尺寸。"""
        b, h, w, c = img.shape
        if target_w is None:
            # 依据目标高计算宽
            scale = target_h / max(h, 1)
            new_w = max(int(round(w * scale)), 1)
            new_h = target_h
        elif target_h is None:
            # 依据目标宽计算高
            scale = target_w / max(w, 1)
            new_h = max(int(round(h * scale)), 1)
            new_w = target_w
        else:
            new_w, new_h = target_w, target_h

        img_nchw = img.permute(0, 3, 1, 2)
        out = F.interpolate(
            img_nchw,
            size=(new_h, new_w),
            mode="bicubic",
            align_corners=False,
        ).permute(0, 2, 3, 1)
        return out

    def _pad_to_rgb(self, img, target_h, target_w, fill_rgb):
        """居中填充到指定尺寸，支持 RGB 填充。"""
        b, h, w, c = img.shape
        out = torch.zeros(
            (b, target_h, target_w, c), dtype=img.dtype, device=img.device
        )
        fill_t = torch.tensor(
            fill_rgb, dtype=img.dtype, device=img.device
        )
        out[:] = fill_t
        top = max((target_h - h) // 2, 0)
        left = max((target_w - w) // 2, 0)
        out[:, top : top + h, left : left + w, :] = img
        return out

    def _make_strip(self, h, w, fill_rgb, axis, dtype, device, batch_size=1):
        """
        生成间隔条。
        axis='v' 垂直条（随宽度拼接），axis='h' 水平条。
        """
        if w <= 0:
            return torch.zeros((batch_size, h, 0, 3), dtype=dtype, device=device)
        if axis == "v":
            out = torch.zeros((batch_size, h, w, 3), dtype=dtype, device=device)
            out[:] = torch.tensor(fill_rgb, dtype=dtype, device=device)
            return out
        out = torch.zeros((batch_size, w, h, 3), dtype=dtype, device=device)
        out[:] = torch.tensor(fill_rgb, dtype=dtype, device=device)
        return out.permute(0, 2, 1, 3)

    def _broadcast_image(self, img, batch_size):
        """将单图或小批次广播到 batch_size，保持内容复制。"""
        b = img.shape[0]
        if b == batch_size:
            return img
        if b == 1:
            return img.repeat(batch_size, 1, 1, 1)
        # 其他情况：循环重复到指定批次
        reps = int(math.ceil(batch_size / b))
        tiled = img.repeat(reps, 1, 1, 1)[:batch_size]
        return tiled


NODE_CLASS_MAPPINGS = {
    "1hew_MultiStringJoin": MultiStringJoin,
    "1hew_MultiImageBatch": MultiImageBatch,
    "1hew_MultiMaskBatch": MultiMaskBatch,
    "1hew_MultiImageStitch": MultiImageStitch,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "1hew_MultiStringJoin": "Multi String Join",
    "1hew_MultiImageBatch": "Multi Image Batch",
    "1hew_MultiMaskBatch": "Multi Mask Batch",
    "1hew_MultiImageStitch": "Multi Image Stitch",
}