from comfy_api.latest import io, ui
import asyncio
import math
import numpy as np
import os
import re
import torch
from PIL import Image, ImageDraw, ImageFont
from typing import List, Tuple


WRAP_CACHE: dict = {}
MEASURE_CACHE: dict = {}
FONT_METRICS_CACHE: dict = {}


class ImageAddLabel(io.ComfyNode):
    """
    为图像添加标签文本 - 支持比例缩放的标签
    标签大小会根据图像尺寸自动调整，确保同比例不同尺寸的图片在缩放后标签大小保持一致
    支持批量图像和批量标签，支持动态引用输入值
    支持 -- 分隔符功能，当存在只包含连字符的行时，-- 之间的内容作为完整标签
    自动选择最佳缩放模式，根据标签方向智能优化
    """
    @classmethod
    def define_schema(cls) -> io.Schema:
        package_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
        font_dir = os.path.join(package_root, "fonts")
        font_files = []
        if os.path.exists(font_dir):
            for file in os.listdir(font_dir):
                if file.lower().endswith((".ttf", ".otf")):
                    font_files.append(file)
        if not font_files:
            font_files = ["FreeMono.ttf"]
        preferred = "Alibaba-PuHuiTi-Regular.otf"
        default_font = preferred if preferred in font_files else font_files[0]
        return io.Schema(
            node_id="1hew_ImageAddLabel",
            display_name="Image Add Label",
            category="1hewNodes/image",
            inputs=[
                io.Image.Input("image"),
                io.Int.Input("height_pad", default=24, min=1, max=1024, step=1),
                io.Int.Input("font_size", default=36, min=1, max=256, step=1),
                io.Boolean.Input("invert_color", default=True),
                io.Combo.Input("font", options=font_files, default=default_font),
                io.String.Input("text", default="", multiline=True, placeholder="-- splits override separator\nelse use newline."),
                io.Combo.Input("direction", options=["top", "bottom", "left", "right"], default="top"),
                io.String.Input("input1", default=""),
                io.String.Input("input2", default=""),
            ],
            outputs=[io.Image.Output(display_name="image")],
        )

    @classmethod
    async def execute(
        cls,
        image: torch.Tensor,
        height_pad: int,
        font_size: int,
        invert_color: bool,
        font: str,
        text: str,
        direction: str,
        input1: str,
        input2: str,
    ) -> io.NodeOutput:
        # 设置颜色，根据invert_color决定黑白配色
        if invert_color:
            font_color = "black"
            label_color = "white"
        else:
            font_color = "white"
            label_color = "black"

        result = []
        total_batches = len(image)
        
        # 缓存字体对象和文本尺寸
        font_cache = {}
        text_size_cache = {}
        scale_factor_cache = {}
        
        # 预处理所有文本，获取每张图片对应的文本
        all_current_texts = []
        all_scale_factors = []
        all_font_sizes = []
        all_height_pads = []
        all_selected_modes = []
        
        for i in range(total_batches):
            parsed_text = cls.parse_text_with_inputs(text, input1, input2, i, total_batches)
            
            text_lines = cls.parse_text_list(parsed_text)
            current_text = text_lines[i % len(text_lines)] if text_lines else ""
            all_current_texts.append(current_text)
            
            # 获取当前图像尺寸并计算缩放因子
            img_data = 255. * image[i].cpu().numpy()
            img_pil = Image.fromarray(np.clip(img_data, 0, 255).astype(np.uint8))
            width, height = img_pil.size
            
            # 使用缓存的缩放因子计算
            size_key = (width, height, direction)
            if size_key not in scale_factor_cache:
                selected_mode = cls.auto_select_scale_mode(width, height, direction)
                scale_factor = cls.calculate_scale_factor(width, height, font_size, selected_mode, direction)
                scale_factor_cache[size_key] = (selected_mode, scale_factor)
            else:
                selected_mode, scale_factor = scale_factor_cache[size_key]
                
            all_scale_factors.append(scale_factor)
            all_selected_modes.append(selected_mode)
            
            # 计算缩放后的字体大小和高度填充
            scaled_font_size = max(8, int(font_size * scale_factor))
            scaled_height_pad = max(4, int(height_pad * scale_factor))
            
            all_font_sizes.append(scaled_font_size)
            all_height_pads.append(scaled_height_pad)
        
        # 批量转换图像为PIL格式，减少重复转换
        pil_images = []
        for i, img in enumerate(image):
            img_data = 255. * img.cpu().numpy()
            img_pil = Image.fromarray(np.clip(img_data, 0, 255).astype(np.uint8))
            pil_images.append(img_pil)
        
        def _render_single(i):
            img_pil = pil_images[i]
            current_text = all_current_texts[i]
            scaled_font_size = all_font_sizes[i]
            scaled_height_pad = all_height_pads[i]
            font_key = (font, scaled_font_size)
            if font_key not in font_cache:
                font_cache[font_key] = cls._load_font(font, scaled_font_size)
            font_obj = font_cache[font_key]
            width, orig_height = img_pil.size
            scale_factor = all_scale_factors[i]
            text_margin = max(10, int(10 * scale_factor))
            if direction in ["top", "bottom"]:
                max_content_width = max(1, width - 2 * text_margin)
            else:
                max_content_width = max(1, orig_height - 2 * text_margin)
            wrap_key = (current_text, id(font_obj), int(max_content_width))
            if wrap_key in WRAP_CACHE:
                wrapped_text = WRAP_CACHE[wrap_key]
            else:
                wrapped_text = cls._wrap_text_to_width(current_text, font_obj, max_content_width)
            text_key = (wrapped_text, font_key)
            if text_key not in text_size_cache:
                text_size_cache[text_key] = cls._calculate_text_size(wrapped_text, font_obj)
            text_width, text_height, text_top_offset, line_heights = text_size_cache[text_key]
            min_padding = max(scaled_height_pad, 4)
            label_height = text_height + min_padding
            if direction in ["top", "bottom"]:
                label_img = Image.new("RGB", (width, label_height), label_color)
                draw = ImageDraw.Draw(label_img)
                text_x = text_margin
                text_y = min_padding // 2 + text_top_offset
                cls._draw_multiline_text(draw, wrapped_text, text_x, text_y, font_obj, font_color, line_heights)
                if direction == "top":
                    new_img = Image.new("RGB", (width, orig_height + label_height))
                    new_img.paste(label_img, (0, 0))
                    new_img.paste(img_pil, (0, label_height))
                else:
                    new_img = Image.new("RGB", (width, orig_height + label_height))
                    new_img.paste(img_pil, (0, 0))
                    new_img.paste(label_img, (0, orig_height))
            else:
                temp_label_img = Image.new("RGB", (orig_height, label_height), label_color)
                draw = ImageDraw.Draw(temp_label_img)
                text_x = text_margin
                text_y = min_padding // 2 + text_top_offset
                cls._draw_multiline_text(draw, wrapped_text, text_x, text_y, font_obj, font_color, line_heights)
                if direction == "left":
                    label_img = temp_label_img.rotate(90, expand=True)
                    new_img = Image.new("RGB", (width + label_height, orig_height))
                    new_img.paste(label_img, (0, 0))
                    new_img.paste(img_pil, (label_height, 0))
                else:
                    label_img = temp_label_img.rotate(270, expand=True)
                    new_img = Image.new("RGB", (width + label_height, orig_height))
                    new_img.paste(img_pil, (0, 0))
                    new_img.paste(label_img, (width, 0))
            img_np = np.array(new_img).astype(np.float32) / 255.0
            img_tensor = torch.from_numpy(img_np)[None,]
            return img_tensor

        tasks = [asyncio.to_thread(_render_single, i) for i in range(len(pil_images))]
        rendered = await asyncio.gather(*tasks)
        for img_tensor in rendered:
            result.append(img_tensor)
    
        out_t = torch.cat(result, dim=0).to(torch.float32)
        out_t = out_t.clamp(0.0, 1.0).to(image.device)
        return io.NodeOutput(out_t)


    @classmethod
    def auto_select_scale_mode(cls, image_width, image_height, direction="top"):
        """
        自动选择最佳的缩放模式
        根据图像的宽高比、尺寸特征和标签放置方向来智能选择
        """
        aspect_ratio = image_width / image_height
        
        # 根据 direction 优先选择合适的缩放模式
        if direction in ["top", "bottom"]:
            # 顶部/底部标签：标签高度相对于图像高度的比例应该保持一致
            # 优先使用 height 模式，确保标签高度占图像高度的比例在等比例缩放时保持一致
            
            if aspect_ratio > 3.0:
                # 极宽图像：使用高度模式，避免标签过大
                return "height"
            elif aspect_ratio < 0.3:
                # 极高图像：可能需要考虑面积，避免标签过小
                return "area" if min(image_width, image_height) < 512 else "height"
            else:
                # 大多数情况使用高度模式
                return "height"
                
        elif direction in ["left", "right"]:
            # 左侧/右侧标签：标签宽度相对于图像宽度的比例应该保持一致
            # 优先使用 width 模式，确保标签宽度占图像宽度的比例在等比例缩放时保持一致
            
            if aspect_ratio < 0.33:
                # 极高图像：使用宽度模式，避免标签过大
                return "width"
            elif aspect_ratio > 3.0:
                # 极宽图像：可能需要考虑面积，避免标签过小
                return "area" if min(image_width, image_height) < 512 else "width"
            else:
                # 大多数情况使用宽度模式
                return "width"
        
        # 回退到原有逻辑（用于其他未知的 direction 值）
        if 0.8 <= aspect_ratio <= 1.25:
            # 接近正方形的图像 (宽高比在 0.8-1.25 之间)
            return "area"
        elif aspect_ratio > 2.0:
            # 极宽图像
            return "height"
        elif aspect_ratio < 0.5:
            # 极高图像
            return "width"
        else:
            # 默认使用面积模式
            return "area"

    @classmethod
    def calculate_scale_factor(cls, image_width, image_height, base_font_size, scale_mode=None, direction="top"):
        """
        根据不同的缩放模式计算缩放因子
        如果未指定 scale_mode，则自动选择最佳模式
        """
        if scale_mode is None:
            scale_mode = cls.auto_select_scale_mode(image_width, image_height, direction)
        
        # 使用1024作为基准分辨率
        base_resolution = 1024
        
        if scale_mode == "area":
            # 基于面积的缩放 - 推荐模式，确保同比例图像标签大小一致
            base_area = base_resolution * base_resolution
            current_area = image_width * image_height
            scale_factor = math.sqrt(current_area / base_area)
        elif scale_mode == "width":
            # 基于宽度的缩放
            scale_factor = image_width / base_resolution
        elif scale_mode == "height":
            # 基于高度的缩放
            scale_factor = image_height / base_resolution
        elif scale_mode == "min_side":
            # 基于最短边的缩放
            min_side = min(image_width, image_height)
            scale_factor = min_side / base_resolution
        elif scale_mode == "max_side":
            # 基于最长边的缩放
            max_side = max(image_width, image_height)
            scale_factor = max_side / base_resolution
        else:
            scale_factor = 1.0
            
        return scale_factor

    @classmethod
    def parse_text_with_inputs(cls, text, input1=None, input2=None, batch_index=None, total_batches=None):
        """
        解析文本中的输入引用，支持变量和简单数学运算
        """
        
        parsed_text = text
        
        # 替换 {input1} 引用
        if input1 is not None and input1 != "":
            parsed_text = parsed_text.replace("{input1}", str(input1))
        
        # 替换 {input2} 引用
        if input2 is not None and input2 != "":
            parsed_text = parsed_text.replace("{input2}", str(input2))
        
        # 处理索引相关变量和运算 - 批量标注时生效
        if batch_index is not None and total_batches is not None:
            # 定义变量值
            variables = {
                'index': batch_index,      # 从0开始
                'idx': batch_index,        # 从0开始
                'range': batch_index       # 从0开始
            }
            
            # 使用正则表达式匹配 {变量名} 或 {变量名+数字} 或 {变量名-数字} 等表达式
            pattern = r'\{((?:index|idx|range)(?:[+\-*/]\d+)?)\}'
            
            def replace_expression(match):
                expression = match.group(1)
                try:
                    # 解析表达式
                    if '+' in expression:
                        var_name, operand = expression.split('+')
                        result = variables[var_name.strip()] + int(operand.strip())
                    elif '-' in expression:
                        var_name, operand = expression.split('-')
                        result = variables[var_name.strip()] - int(operand.strip())
                    elif '*' in expression:
                        var_name, operand = expression.split('*')
                        result = variables[var_name.strip()] * int(operand.strip())
                    elif '/' in expression:
                        var_name, operand = expression.split('/')
                        result = variables[var_name.strip()] // int(operand.strip())  # 整数除法
                    else:
                        # 纯变量名
                        result = variables[expression.strip()]
                    
                    # 对于 range 变量，保持补零格式
                    if expression.strip() == 'range' or (expression.startswith('range') and result >= 0):
                        if total_batches >= 100:
                            return f"{result:03d}"
                        elif total_batches >= 10:
                            return f"{result:02d}"
                        else:
                            return str(result)
                    else:
                        # index 和 idx 默认不补零
                        return str(result)
                        
                except (ValueError, KeyError, ZeroDivisionError):
                    # 如果解析失败，返回原始表达式
                    return match.group(0)
            
            parsed_text = re.sub(pattern, replace_expression, parsed_text)
            
        return parsed_text

    @classmethod
    def parse_text_list(cls, text):
        """
        解析文本列表，支持连字符分割和换行分割
        当有只包含连字符的行时，只按 -- 进行分割，其他分割方式失效
        否则按照换行符(\n) 分割
        """
        
        if not text.strip():
            return [""]
        
        # 检查是否有只包含连字符的行
        lines = text.split('\n')
        has_dash_separator = any(line.strip() and all(c == '-' for c in line.strip()) for line in lines)
        
        if has_dash_separator:
            # 按连字符分割，其他分割方式失效（包括换行符）
            sections = re.split(r'^\s*-+\s*$', text, flags=re.MULTILINE)
            all_lists = []
            
            for section in sections:
                section = section.strip()
                if not section:
                    continue
                    
                # 当有连字符分割时，每个段落作为一个完整项目，保留内部换行
                # 移除引号
                if (section.startswith('"') and section.endswith('"')) or (section.startswith("'") and section.endswith("'")):
                    section = section[1:-1]
                if section:
                    all_lists.append(str(section))
            
            return all_lists if all_lists else [""]
        else:
            # 按传统方式分割（换行符）
            text_lines = text.strip().split('\n')
            # 过滤空行
            text_lines = [line.strip() for line in text_lines if line.strip()]
            return text_lines if text_lines else [""]

    @classmethod
    def _calculate_text_size(cls, text, font_obj):
        """
        计算文本的尺寸，支持多行文本
        使用固定行高确保一致性
        """
        # 获取固定行高与顶部偏移（带缓存）
        fixed_line_height, text_top_offset = cls._get_line_metrics(font_obj)

        # 创建绘制对象测量宽度
        temp_img = Image.new("RGB", (1, 1))
        temp_draw = ImageDraw.Draw(temp_img)

        # 分割文本为多行
        lines = text.split('\n')
        max_width = 0

        # 计算每行的实际宽度，但使用固定行高
        line_heights = []
        for line in lines:
            try:
                text_bbox = temp_draw.textbbox((0, 0), line, font=font_obj)
                line_width = text_bbox[2] - text_bbox[0]
            except AttributeError:
                line_width, _ = temp_draw.textsize(line, font=font_obj)

            max_width = max(max_width, line_width)
            line_heights.append(fixed_line_height)  # 使用固定行高

        total_height = fixed_line_height * len(lines)  # 总高度 = 固定行高 × 行数

        return max_width, total_height, text_top_offset, line_heights

    @classmethod
    def _get_line_metrics(cls, font_obj):
        """获取字体固定行高与顶部偏移，并做缓存。"""
        key = id(font_obj)
        if key in FONT_METRICS_CACHE:
            return FONT_METRICS_CACHE[key]

        # 优先使用字体度量
        try:
            ascent, descent = font_obj.getmetrics()
            fixed_line_height = ascent + descent
            text_top_offset = 0
            FONT_METRICS_CACHE[key] = (fixed_line_height, text_top_offset)
            return FONT_METRICS_CACHE[key]
        except AttributeError:
            pass

        # 回退：用标准字符测量高度
        temp_img = Image.new("RGB", (1, 1))
        temp_draw = ImageDraw.Draw(temp_img)
        standard_chars = (
            "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"  # noqa: E501
            "!@#$%^&*()_+-=[]{}|;':,.<>?/~`"  # noqa: E501
            "中文测试样本汉字字符高度宽度计算标准参考"  # noqa: E501
        )
        try:
            text_bbox = temp_draw.textbbox((0, 0), standard_chars, font=font_obj)
            fixed_line_height = text_bbox[3] - text_bbox[1]
            text_top_offset = -text_bbox[1]
        except AttributeError:
            _, fixed_line_height = temp_draw.textsize(standard_chars, font=font_obj)
            text_top_offset = 0

        FONT_METRICS_CACHE[key] = (fixed_line_height, text_top_offset)
        return FONT_METRICS_CACHE[key]

    @classmethod
    def _wrap_text_to_width(cls, text, font_obj, max_width):
        """
        根据最大像素宽度自动换行，兼顾中英文与无空格文本。

        按行处理：
        - 优先按空格分词保持单词完整；
        - 对无空格或超长连续字符回退为逐字符包裹。
        """
        if max_width <= 0:
            return text

        # 包裹缓存：同一字体与宽度重复文本直接复用
        font_id = id(font_obj)
        wrap_key = (text, font_id, int(max_width))
        if wrap_key in WRAP_CACHE:
            return WRAP_CACHE[wrap_key]

        temp_img = Image.new("RGB", (1, 1))
        draw = ImageDraw.Draw(temp_img)

        # 宽度测量缓存，减少重复 textbbox 调用
        if font_id not in MEASURE_CACHE:
            MEASURE_CACHE[font_id] = {}

        def measure(s):
            cache = MEASURE_CACHE[font_id]
            if s in cache:
                return cache[s]
            try:
                bbox = draw.textbbox((0, 0), s, font=font_obj)
                w = bbox[2] - bbox[0]
            except AttributeError:
                w, _ = draw.textsize(s, font=font_obj)
            cache[s] = w
            return w

        def max_fit_index(line):
            """二分查找可容纳的最大子串长度。"""
            lo, hi = 0, len(line)
            while lo < hi:
                mid = (lo + hi + 1) // 2
                if measure(line[:mid]) <= max_width:
                    lo = mid
                else:
                    hi = mid - 1
            return lo

        wrapped_lines = []
        for raw_line in text.split("\n"):
            if not raw_line:
                wrapped_lines.append("")
                continue

            # 如果整行已在宽度内，直接添加
            if measure(raw_line) <= max_width:
                wrapped_lines.append(raw_line)
                continue

            # 优先基于空格分词
            if " " in raw_line or "\t" in raw_line:
                tokens = re.findall(r"\S+\s*", raw_line)
                current = ""
                for token in tokens:
                    candidate = current + token
                    if measure(candidate) <= max_width:
                        current = candidate
                    else:
                        if current.strip():
                            wrapped_lines.append(current.rstrip())
                            current = token.lstrip()
                            # 若新行上的 token 仍超宽，立即按字符切分
                            if measure(current) > max_width:
                                idx = max_fit_index(current)
                                if idx > 0:
                                    wrapped_lines.append(current[:idx])
                                    current = current[idx:]
                                # 如果仍有剩余且再次超宽，循环处理
                                while current and measure(current) > max_width:
                                    idx = max_fit_index(current)
                                    if idx == 0:
                                        # 极端情况，至少放一个字符
                                        wrapped_lines.append(current[0])
                                        current = current[1:]
                                    else:
                                        wrapped_lines.append(current[:idx])
                                        current = current[idx:]
                        else:
                            # 单个token过长，回退字符级拆分
                            rest = token
                            while rest:
                                idx = max_fit_index(rest)
                                if idx == 0:
                                    # 至少输出一个字符，避免死循环
                                    if current:
                                        wrapped_lines.append(current)
                                        current = ""
                                    wrapped_lines.append(rest[0])
                                    rest = rest[1:]
                                else:
                                    cand = current + rest[:idx]
                                    if measure(cand) <= max_width:
                                        current = cand
                                    else:
                                        if current:
                                            wrapped_lines.append(current)
                                        wrapped_lines.append(rest[:idx])
                                        current = ""
                                    rest = rest[idx:]
                if current:
                    wrapped_lines.append(current)
            else:
                # 无空格文本，逐字符包裹（适合中文等情况）
                line = raw_line
                while line:
                    idx = max_fit_index(line)
                    if idx == 0:
                        # 至少输出一个字符
                        wrapped_lines.append(line[0])
                        line = line[1:]
                    else:
                        wrapped_lines.append(line[:idx])
                        line = line[idx:]
        wrapped = "\n".join(wrapped_lines)
        WRAP_CACHE[wrap_key] = wrapped
        return wrapped

    @classmethod
    def _draw_multiline_text(cls, draw, text, x, y, font_obj, font_color, line_heights):
        """
        绘制多行文本，使用固定行高
        """
        lines = text.split('\n')
        current_y = y
        
        for i, line in enumerate(lines):
            draw.text((x, current_y), line, fill=font_color, font=font_obj)
            if i < len(line_heights):
                current_y += line_heights[i]  # 使用固定行高间距

    @classmethod
    def _load_font(cls, font, font_size):
        """
        加载字体对象
        """
        try:
            package_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
            font_path = os.path.join(package_root, "fonts", font)
            font_obj = ImageFont.truetype(font_path, font_size)
        except Exception as e:
            print(f"无法加载字体 {font}: {e}，使用默认字体")
            font_obj = ImageFont.load_default()
        return font_obj

    @classmethod
    def validate_inputs(
        cls,
        image: torch.Tensor,
        height_pad: int,
        font_size: int,
        invert_color: bool,
        font: str,
        text: str,
        direction: str,
        input1: str,
        input2: str,
    ):
        if direction not in {"top", "bottom", "left", "right"}:
            return "invalid direction"
        if font_size < 1 or height_pad < 1:
            return "invalid font_size or height_pad"
        return True

    @classmethod
    def fingerprint_inputs(
        cls,
        image: torch.Tensor,
        height_pad: int,
        font_size: int,
        invert_color: bool,
        font: str,
        text: str,
        direction: str,
        input1: str,
        input2: str,
    ):
        b = int(image.shape[0]) if isinstance(image, torch.Tensor) else 0
        h = int(image.shape[1]) if isinstance(image, torch.Tensor) else 0
        w = int(image.shape[2]) if isinstance(image, torch.Tensor) else 0
        return f"{b}x{h}x{w}|{height_pad}|{font_size}|{invert_color}|{font}|{direction}|{hash(text)}|{hash(input1)}|{hash(input2)}"