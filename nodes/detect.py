# YOLO目标检测独立节点 for ComfyUI
# 支持YOLOv5, YOLOv8, YOLOv9, YOLOv10, YOLOv11等多个版本
# 这是一个独立版本，可以在其他仓库中使用

# 使用说明和要求
"""
要求:
- 将YOLO模型文件(.pt)放置在 ComfyUI/models/yolo/ 目录中

模型下载:
您可以从以下位置下载YOLO模型:
- https://github.com/ultralytics/assets/releases
- 官方YOLO发布版本

"""

import copy
import os
import glob
import numpy as np
import torch
import cv2
from PIL import Image, ImageChops
import folder_paths


class DetectYolo:
    """
    YOLO目标检测节点
    使用YOLO模型检测图像中的目标并生成遮罩
    支持YOLOv5, YOLOv8, YOLOv9, YOLOv10, YOLOv11等多个版本
    支持阈值控制、索引标注和遮罩选择功能
    """
    
    def __init__(self):
        self.NODE_NAME = 'DetectYolo'

    @classmethod
    def INPUT_TYPES(cls):
        """定义节点的输入类型"""
        model_ext = [".pt"]
        model_path = cls.get_model_path()
        FILES_DICT = cls.get_files(model_path, model_ext)
        FILE_LIST = list(FILES_DICT.keys())
        
        # 如果没有找到模型，添加默认选项
        if not FILE_LIST:
            FILE_LIST = ["No models found - Please add .pt files to models/yolo/"]
        
        return {
            "required": {
                "image": ("IMAGE", ),
                "yolo_model": (FILE_LIST,),
                "threshold": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01}),
                "mask_index": ("STRING", {"default": "-1"}),
                "label": ("BOOLEAN", {"default": True}),
                "label_size": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 5.0, "step": 0.1}),
            },
            "optional": {
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("plot_image", "mask")
    FUNCTION = 'yolo_detect'
    CATEGORY = "1hewNodes/detect"

    def yolo_detect(self, image, yolo_model, threshold, mask_index, label, label_size):
        """
        执行YOLO目标检测
        
        参数:
            image: 输入图像张量
            yolo_model: YOLO模型文件名
            threshold: 检测置信度阈值 (0.0-1.0)
            mask_index: 遮罩选择的逗号分隔索引 (空值表示全部)
            
        返回:
            元组: (选中的遮罩, 绘制图像)
        """
        ret_masks = []
        ret_plot_images = []

        try:
            from ultralytics import YOLO
        except ImportError:
            self.log("ultralytics package is required. Please install it with: pip install ultralytics", message_type='error')
            raise ImportError("ultralytics package not found")

        # 获取模型路径并加载YOLO模型
        model_path = self.get_model_path()
        FILES_DICT = self.get_files(model_path, [".pt"])
        
        # 从文件字典中获取完整路径
        if yolo_model in FILES_DICT:
            yolo_model_path = FILES_DICT[yolo_model]
        else:
            # 备选方案：直接拼接路径（向后兼容）
            yolo_model_path = os.path.join(model_path, yolo_model)
        
        # 检查模型文件是否存在
        if not os.path.exists(yolo_model_path):
            self.log(f"Model file not found: {yolo_model_path}", message_type='error')
            self.log(f"Please place YOLO model files (.pt) in: {model_path}", message_type='warning')
            raise FileNotFoundError(f"Model file not found: {yolo_model_path}")
            
        yolo_model = YOLO(yolo_model_path)
        
        # 解析遮罩索引
        selected_indices = self.parse_mask_indices(mask_index)
        show_all = -1 in selected_indices  # 是否显示所有目标

        for i in image:
            i = torch.unsqueeze(i, 0)
            _image = self.tensor2pil(i)
            
            # 使用阈值运行YOLO检测
            results = yolo_model(_image, retina_masks=True, conf=threshold)
            
            image_masks = []  # 存储此图像的所有遮罩
            
            for result in results:
                # 创建原始图像的副本用于绘制
                plot_image = np.array(_image)
                
                # 添加自定义标签格式：[index]类别名 置信度
                if result.boxes is not None and len(result.boxes) > 0:
                    for index, box in enumerate(result.boxes):
                        # 检查是否应该显示此索引的目标
                        if not show_all and index not in selected_indices:
                            continue
                            
                        # 获取边界框坐标
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        
                        # 获取类别名称和置信度
                        class_id = int(box.cls[0].cpu().numpy())
                        confidence = float(box.conf[0].cpu().numpy())
                        class_name = yolo_model.names[class_id] if class_id in yolo_model.names else f"class_{class_id}"
                        
                        # 绘制边界框
                        box_color = (4, 42, 255)  # RGB格式的蓝色
                        box_thickness = max(1, int(3 * label_size))  # 根据label_size调整边界框粗细
                        cv2.rectangle(plot_image, (x1, y1), (x2, y2), box_color, box_thickness)
                        
                        # 创建自定义标签格式：[index] 类别名 置信度
                        if label:
                            custom_label = f"[{index}] {class_name} {confidence:.2f}"
                            font_scale = 0.6 * label_size  # 使用label_size参数控制字体大小
                            font_thickness = max(1, int(2 * label_size))  # 根据label_size调整线条粗细
                            font_color = (255, 255, 255)
                            bg_color = (4, 42, 255)  # RGB格式的蓝色背景 
                            
                            # 获取文本尺寸
                            (text_width, text_height), baseline = cv2.getTextSize(
                                custom_label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
                            )
                            
                            # 计算标签位置 - 以左下角为基准点，确保标签贴合检测框
                            # 标签的左下角固定在检测框的左上角位置
                            label_x = x1  # 标签左边缘与检测框左边缘对齐
                            
                            # 标签底部距离检测框上边缘的固定距离（不受缩放影响）
                            fixed_gap = 5  # 固定间隙5像素
                            label_y = y1 - fixed_gap  # 标签底部位置
                            
                            # 确保标签不会超出图像边界
                            if label_y - text_height < 0:
                                # 如果上方空间不够，放在检测框内部上方
                                label_y = y1 + text_height + fixed_gap
                            
                            # 绘制标签背景矩形 - 基于标签的实际尺寸
                            bg_padding = max(2, int(2 * label_size))  # 背景内边距随label_size缩放
                            cv2.rectangle(plot_image, 
                                        (label_x - bg_padding, label_y - text_height - bg_padding), 
                                        (label_x + text_width + bg_padding, label_y + baseline + bg_padding), 
                                        bg_color, -1)
                            
                            # 绘制黑色自定义标签文本
                            cv2.putText(plot_image, custom_label, 
                                      (label_x, label_y), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 
                                      font_scale, font_color, font_thickness)
                
                ret_plot_images.append(self.pil2tensor(Image.fromarray(plot_image)))
                
                # 如果有遮罩则处理遮罩
                if result.masks is not None and len(result.masks) > 0:
                    masks_data = result.masks.data
                    for index, mask in enumerate(masks_data):
                        _mask = mask.cpu().numpy() * 255
                        _mask = self.np2pil(_mask).convert("L")
                        image_masks.append(self.image2mask(_mask))
                        
                # 如果没有遮罩则处理边界框
                elif result.boxes is not None and len(result.boxes.xyxy) > 0:
                    white_image = Image.new('L', _image.size, "white")
                    for index, box in enumerate(result.boxes):
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        _mask = Image.new('L', _image.size, "black")
                        _mask.paste(white_image.crop((x1, y1, x2, y2)), (x1, y1))
                        image_masks.append(self.image2mask(_mask))
                        
                # 未找到检测结果
                else:
                    self.log(f"{self.NODE_NAME} mask or box not detected.")

            # 根据mask_index选择并合并遮罩
            if len(image_masks) > 0:
                if show_all:  # -1表示所有遮罩
                    # 合并所有遮罩
                    _mask = image_masks[0]
                    for i in range(1, len(image_masks)):
                        _mask = self.add_mask(_mask, image_masks[i])
                    ret_masks.append(_mask)
                else:
                    # 选择特定索引
                    selected_masks = []
                    for idx in selected_indices:
                        if idx < len(image_masks):
                            selected_masks.append(image_masks[idx])
                    
                    if selected_masks:
                        # 合并选定的遮罩
                        _mask = selected_masks[0]
                        for i in range(1, len(selected_masks)):
                            _mask = self.add_mask(_mask, selected_masks[i])
                        ret_masks.append(_mask)
                    else:
                        # 没有有效索引，创建空遮罩
                        ret_masks.append(torch.zeros((1, _image.size[1], _image.size[0]), dtype=torch.float32))
            else:
                # 如果没有检测结果，创建空遮罩
                ret_masks.append(torch.zeros((1, _image.size[1], _image.size[0]), dtype=torch.float32))

        self.log(f"{self.NODE_NAME} Processed {len(ret_masks)} image(s).", message_type='finish')
        
        return (
            torch.cat(ret_plot_images, dim=0) if ret_plot_images else torch.zeros((1, 512, 512, 3), dtype=torch.float32),
            torch.cat(ret_masks, dim=0) if ret_masks else torch.zeros((1, 512, 512), dtype=torch.float32),
        )

    @staticmethod
    def log(message: str, message_type: str = 'info'):
        """简洁的日志消息输出"""
        type_prefix = {
            'error': '[ERROR]',
            'warning': '[WARNING]', 
            'finish': '[SUCCESS]',
            'info': '[INFO]'
        }.get(message_type, '[INFO]')
        
        print(f"# DetectYolo: {type_prefix} {message}")

    @staticmethod
    def pil2tensor(image: Image.Image) -> torch.Tensor:
        """将PIL图像转换为张量"""
        return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

    @staticmethod
    def np2pil(np_image: np.ndarray) -> Image.Image:
        """将numpy数组转换为PIL图像"""
        return Image.fromarray(np_image)

    @staticmethod
    def tensor2pil(t_image: torch.Tensor) -> Image.Image:
        """将张量转换为PIL图像"""
        return Image.fromarray(np.clip(255.0 * t_image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

    @staticmethod
    def image2mask(image: Image.Image) -> torch.Tensor:
        """将PIL图像转换为遮罩张量"""
        if image.mode == 'L':
            return torch.tensor([DetectYolo.pil2tensor(image)[0, :, :].tolist()])
        else:
            image = image.convert('RGB').split()[0]
            return torch.tensor([DetectYolo.pil2tensor(image)[0, :, :].tolist()])

    @staticmethod
    def chop_image(background_image: Image.Image, layer_image: Image.Image, blend_mode: str, opacity: int) -> Image.Image:
        """使用指定混合模式混合两个图像"""
        ret_image = background_image
        
        if blend_mode == 'normal':
            ret_image = copy.deepcopy(layer_image)
        elif blend_mode == 'multiply':
            ret_image = ImageChops.multiply(background_image, layer_image)
        elif blend_mode == 'screen':
            ret_image = ImageChops.screen(background_image, layer_image)
        elif blend_mode == 'add':
            ret_image = ImageChops.add(background_image, layer_image, 1, 0)
        elif blend_mode == 'subtract':
            ret_image = ImageChops.subtract(background_image, layer_image, 1, 0)
        elif blend_mode == 'difference':
            ret_image = ImageChops.difference(background_image, layer_image)
        elif blend_mode == 'darker':
            ret_image = ImageChops.darker(background_image, layer_image)
        elif blend_mode == 'lighter':
            ret_image = ImageChops.lighter(background_image, layer_image)
        
        # 应用透明度
        if opacity == 0:
            ret_image = background_image
        elif opacity < 100:
            alpha = 1.0 - float(opacity) / 100
            ret_image = Image.blend(ret_image, background_image, alpha)
        
        return ret_image

    @staticmethod
    def add_mask(masks_a: torch.Tensor, masks_b: torch.Tensor) -> torch.Tensor:
        """将两个遮罩相加"""
        mask = DetectYolo.chop_image(DetectYolo.tensor2pil(masks_a), DetectYolo.tensor2pil(masks_b), blend_mode='add', opacity=100)
        return DetectYolo.image2mask(mask)

    @staticmethod
    def get_files(model_path: str, file_ext_list: list) -> dict:
        """从目录中递归获取指定扩展名的文件，支持子文件夹"""
        file_list = []
        for ext in file_ext_list:
            # 递归搜索：使用 ** 模式匹配所有子目录
            file_list.extend(glob.glob(os.path.join(model_path, '**', '*' + ext), recursive=True))
        
        files_dict = {}
        for file_path in file_list:
            # 获取相对于model_path的相对路径作为显示名称
            rel_path = os.path.relpath(file_path, model_path)
            # 将路径分隔符统一为 / 以保持一致性
            display_name = rel_path.replace(os.sep, '/')
            files_dict[display_name] = file_path
        
        return files_dict

    @staticmethod
    def parse_mask_indices(mask_index: str) -> list:
        """
        解析遮罩索引字符串为整数列表
        
        参数:
            mask_index: 字符串格式如 "0,1,2" 或 "0，1，2" 或 "0 1 2" 或 "-1" 或空字符串
            
        返回:
            整数列表，空列表或包含-1表示所有索引
        """
        if not mask_index or mask_index.strip() == "":
            return [-1]  # 空值表示全部
            
        # 将中文逗号和空格替换为标准逗号
        mask_index = mask_index.replace('，', ',').replace(' ', ',')
        
        # 按逗号分割并过滤空字符串
        indices = []
        for idx in mask_index.split(','):
            idx = idx.strip()
            if idx == "-1":
                return [-1]  # -1表示全部
            elif idx.isdigit():
                indices.append(int(idx))
        
        return indices if indices else [-1]

    @classmethod
    def get_model_path(cls):
        """获取YOLO模型路径，提供多种备选方案"""
        try:
            # 主要路径：ComfyUI模型目录
            primary_path = os.path.join(folder_paths.models_dir, 'yolo')
            if os.path.exists(primary_path):
                return primary_path
        except:
            pass
        
        try:
            # 备选方案1：使用base_path（如果可用）
            fallback_path = os.path.join(folder_paths.base_path, 'models', 'yolo')
            if os.path.exists(fallback_path):
                return fallback_path
        except:
            pass
        
        # 备选方案2：在当前工作目录创建目录
        current_dir_path = os.path.join(os.getcwd(), 'models', 'yolo')
        os.makedirs(current_dir_path, exist_ok=True)
        return current_dir_path


class DetectGuideLine:
    """引导线检测"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "canny_low": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.01}),
                "canny_high": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.01}),
                "seg_min_len": ("INT", {"default": 40, "min": 1, "max": 300, "step": 1}),
                "seg_max_gap": ("INT", {"default": 8, "min": 1, "max": 100, "step": 1}),
                "guide_filter": ("FLOAT", {"default": 0.6, "min": 0.1, "max": 1.0, "step": 0.1}),
                "guide_width": ("INT", {"default": 2, "min": 1, "max": 100, "step": 1}),
                "cluster_eps": ("INT", {"default": 30, "min": 1, "max": 100, "step": 5}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "IMAGE", "MASK")
    RETURN_NAMES = ("image", "line_image", "line_mask")
    FUNCTION = "detect_guide_line"
    CATEGORY = "1hewNodes/detect"

    def detect_guide_line(self, image, canny_low, canny_high, seg_min_len, seg_max_gap, guide_filter, guide_width, cluster_eps):
        # 1. 图像格式转换
        img_np = (image[0].cpu().numpy() * 255).astype(np.uint8)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        img_rgb = img_np.copy()
        h, w = img_rgb.shape[:2]
        lines_only = np.zeros((h, w, 3), dtype=np.uint8)
        line_mask = np.zeros((h, w), dtype=np.uint8)
        red = (255, 0, 0)
        default_vp = (w//2, h//2)  # 默认消失点（图像中心）
        
        # 2. 边缘检测（带降噪，阈值采用0-1映射）
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)
        low = int(np.clip(float(canny_low), 0.0, 1.0) * 255)
        high = int(np.clip(float(canny_high), 0.0, 1.0) * 255)
        high = max(high, low)  # 确保高阈值不低于低阈值
        edges = cv2.Canny(gray_blur, low, high)
        
        # 3. 线段提取
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=40,
            minLineLength=seg_min_len,
            maxLineGap=seg_max_gap
        )
        
        if lines is None:
            return (
                image,
                torch.from_numpy(lines_only.astype(np.float32) / 255.0).unsqueeze(0),
                torch.from_numpy(line_mask.astype(np.float32) / 255.0).unsqueeze(0),
            )
        
        # 4. 计算延长线交点
        intersections = []
        for i in range(len(lines)):
            x1, y1, x2, y2 = lines[i][0]
            dx, dy = x2 - x1, y2 - y1
            x1_ext = x1 - 0.1*dx
            y1_ext = y1 - 0.1*dy
            x2_ext = x2 + 0.1*dx
            y2_ext = y2 + 0.1*dy
            
            for j in range(i + 1, len(lines)):
                x3, y3, x4, y4 = lines[j][0]
                dx2, dy2 = x4 - x3, y4 - y3
                x3_ext = x3 - 0.1*dx2
                y3_ext = y3 - 0.1*dy2
                x4_ext = x4 + 0.1*dx2
                y4_ext = y4 + 0.1*dy2
                
                den = (x1_ext - x2_ext) * (y3_ext - y4_ext) - (y1_ext - y2_ext) * (x3_ext - x4_ext)
                if den != 0:
                    t_num = (x1_ext - x3_ext) * (y3_ext - y4_ext) - (y1_ext - y3_ext) * (x3_ext - x4_ext)
                    u_num = -((x1_ext - x2_ext) * (y1_ext - y3_ext) - (y1_ext - y2_ext) * (x1_ext - x3_ext))
                    t = t_num / den
                    u = u_num / den
                    x = x1_ext + t * (x2_ext - x1_ext)
                    y = y1_ext + t * (y2_ext - y1_ext)
                    if (-w <= x <= 2*w) and (-h <= y <= 2*h):
                        intersections.append((x, y))
        
        if not intersections:
            return (
                image,
                torch.from_numpy(lines_only.astype(np.float32) / 255.0).unsqueeze(0),
                torch.from_numpy(line_mask.astype(np.float32) / 255.0).unsqueeze(0),
            )
        
        # 5. 聚类优化（核心修复部分）
        intersections_np = np.array(intersections)
        try:
            dbscan = DBSCAN(eps=cluster_eps, min_samples=5)
            clusters = dbscan.fit_predict(intersections_np)
            
            # 过滤噪声点（-1）
            valid_clusters = clusters[clusters != -1]
            if len(valid_clusters) == 0:  # 全是噪声点
                vanishing_point = default_vp
            else:
                # 找到最大聚类
                cluster_mode = mode(valid_clusters)
                if len(cluster_mode) == 0 or len(cluster_mode[0]) == 0:
                    largest_cluster_idx = 0
                else:
                    largest_cluster_idx = cluster_mode[0][0]
                
                # 确保聚类索引存在
                if largest_cluster_idx not in clusters:
                    vanishing_point = default_vp
                else:
                    cluster_points = intersections_np[clusters == largest_cluster_idx]
                    if len(cluster_points) == 0:
                        vanishing_point = default_vp
                    else:
                        vanishing_point = (int(np.mean(cluster_points[:,0])), int(np.mean(cluster_points[:,1])))
        except:
            # 任何异常都返回默认消失点
            vanishing_point = default_vp
        
        # 6. 线条筛选
        line_scores = []
        vx, vy = vanishing_point
        for line in lines:
            x1, y1, x2, y2 = line[0]
            dir1 = np.array([x1 - vx, y1 - vy])
            dir2 = np.array([x2 - vx, y2 - vy])
            line_dir = np.array([x2 - x1, y2 - y1])
            if np.linalg.norm(line_dir) < 1e-8:
                line_scores.append(0.0)
                continue
            line_dir_norm = line_dir / np.linalg.norm(line_dir)
            score1 = np.abs(np.dot(dir1 / (np.linalg.norm(dir1)+1e-8), line_dir_norm)) if np.linalg.norm(dir1) > 1e-8 else 0
            score2 = np.abs(np.dot(dir2 / (np.linalg.norm(dir2)+1e-8), line_dir_norm)) if np.linalg.norm(dir2) > 1e-8 else 0
            line_scores.append(max(score1, score2))
        
        # 避免空列表导致的percentile错误
        if len(line_scores) == 0:
            filtered_lines = []
        else:
            threshold = np.percentile(line_scores, 100 - (guide_filter * 60))
            filtered_lines = [line for i, line in enumerate(lines) if line_scores[i] >= threshold]
        
        # 7. 绘制线条
        for line in filtered_lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(img_rgb, (x1, y1), vanishing_point, red, guide_width)
            cv2.line(img_rgb, (x2, y2), vanishing_point, red, guide_width)
            cv2.line(lines_only, (x1, y1), vanishing_point, red, guide_width)
            cv2.line(lines_only, (x2, y2), vanishing_point, red, guide_width)
            cv2.line(line_mask, (x1, y1), vanishing_point, 255, guide_width)
        
        # 标记消失点
        cv2.circle(img_rgb, vanishing_point, max(5, guide_width), red, -1)
        cv2.circle(lines_only, vanishing_point, max(5, guide_width), red, -1)
        cv2.circle(line_mask, vanishing_point, max(5, guide_width), 255, -1)
        
        # 转换格式
        result_with_lines = torch.from_numpy(img_rgb.astype(np.float32) / 255.0).unsqueeze(0)
        result_lines_only = torch.from_numpy(lines_only.astype(np.float32) / 255.0).unsqueeze(0)
        result_line_mask = torch.from_numpy(line_mask.astype(np.float32) / 255.0).unsqueeze(0)
        
        return (result_with_lines, result_lines_only, result_line_mask)


# ComfyUI节点注册
NODE_CLASS_MAPPINGS = {
    "1hew_DetectYolo": DetectYolo,
    "1hew_DetectGuideLine": DetectGuideLine
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "1hew_DetectYolo": "Detect Yolo",
    "1hew_DetectGuideLine": "Detect Guide Line"
}
