import asyncio
from comfy_api.latest import io, ui
import cv2
import folder_paths
import glob
import numpy as np
import os
from PIL import Image
import torch
from ultralytics import YOLO


class DetectYolo(io.ComfyNode):
    NODE_NAME = "DetectYolo"
    MODEL_CACHE = {}
    @classmethod
    def define_schema(cls) -> io.Schema:
        model_dir = cls.get_model_path()
        files = cls.get_files(model_dir, [".pt"])
        options = list(files.keys()) or ["No models found - Please add .pt files to models/yolo/"]
        return io.Schema(
            node_id="1hew_DetectYolo",
            display_name="Detect Yolo",
            category="1hewNodes/detect",
            inputs=[
                io.Image.Input("image"),
                io.Combo.Input("yolo_model", options=options, default=options[0]),
                io.Float.Input("threshold", default=0.3, min=0.0, max=1.0, step=0.01),
                io.String.Input("mask_index", default="-1"),
                io.Boolean.Input("label", default=True),
                io.Float.Input("label_size", default=1.0, min=0.1, max=5.0, step=0.01),
            ],
            outputs=[
                io.Image.Output(display_name="plot_image"),
                io.Mask.Output(display_name="mask"),
            ],
        )

    @classmethod
    async def execute(
        cls,
        image: torch.Tensor,
        yolo_model: str,
        threshold: float,
        mask_index: str,
        label: bool,
        label_size: float,
    ) -> io.NodeOutput:
        ret_masks = []
        ret_plot_images = []

        model_path = cls.get_model_path()
        files_dict = cls.get_files(model_path, [".pt"])
        if yolo_model in files_dict:
            yolo_model_path = files_dict[yolo_model]
        else:
            yolo_model_path = os.path.join(model_path, yolo_model)
        if not os.path.exists(yolo_model_path):
            cls.log(f"Model file not found: {yolo_model_path}", message_type="error")
            cls.log(
                f"Please place YOLO model files (.pt) in: {model_path}",
                message_type="warning",
            )
            raise FileNotFoundError(f"Model file not found: {yolo_model_path}")

        if yolo_model_path in cls.MODEL_CACHE:
            model = cls.MODEL_CACHE[yolo_model_path]
        else:
            model = await asyncio.to_thread(YOLO, yolo_model_path)
            cls.MODEL_CACHE[yolo_model_path] = model

        selected_indices = cls.parse_mask_indices(mask_index)
        show_all = -1 in selected_indices

        sem = asyncio.Semaphore(1)
        tasks = []
        for i in image:
            async def run_one(x=i):
                async with sem:
                    return await asyncio.to_thread(
                        cls._process_image,
                        model,
                        x,
                        selected_indices,
                        show_all,
                        threshold,
                        label,
                        label_size,
                    )
            tasks.append(run_one())

        results = await asyncio.gather(*tasks)
        for plot_tensor, mask_tensor in results:
            ret_plot_images.append(plot_tensor)
            ret_masks.append(mask_tensor)

        cls.log(
            f"{cls.NODE_NAME} Processed {len(ret_masks)} image(s).",
            message_type="finish",
        )

        out_image = (
            torch.cat(ret_plot_images, dim=0)
            if ret_plot_images
            else torch.zeros((1, 512, 512, 3), dtype=torch.float32)
        )
        out_mask = (
            torch.cat(ret_masks, dim=0)
            if ret_masks
            else torch.zeros((1, 512, 512), dtype=torch.float32)
        )

        return io.NodeOutput(out_image, out_mask)

    @staticmethod
    def log(message: str, message_type: str = "info"):
        """简洁的日志消息输出"""
        type_prefix = {
            "error": "[ERROR]",
            "warning": "[WARNING]",
            "finish": "[SUCCESS]",
            "info": "[INFO]",
        }.get(message_type, "[INFO]")

        print(f"# DetectYolo: {type_prefix} {message}")

    @staticmethod
    def pil2tensor(image: Image.Image) -> torch.Tensor:
        """将PIL图像转换为张量"""
        return (
            torch.from_numpy(np.array(image).astype(np.float32) / 255.0)
            .unsqueeze(0)
        )

    @staticmethod
    def np2pil(np_image: np.ndarray) -> Image.Image:
        """将numpy数组转换为PIL图像"""
        return Image.fromarray(np_image)

    @staticmethod
    def tensor2pil(t_image: torch.Tensor) -> Image.Image:
        """将张量转换为PIL图像"""
        return Image.fromarray(
            np.clip(255.0 * t_image.cpu().numpy().squeeze(), 0, 255).astype(
                np.uint8
            )
        )

    @staticmethod
    def image2mask(image: Image.Image) -> torch.Tensor:
        """将PIL图像转换为遮罩张量"""
        if image.mode != "L":
            image = image.convert("L")
        return DetectYolo.pil2tensor(image)

    @staticmethod
    def add_mask(masks_a: torch.Tensor, masks_b: torch.Tensor) -> torch.Tensor:
        masks_a = masks_a.float()
        masks_b = masks_b.float()
        return torch.clamp(masks_a + masks_b, 0.0, 1.0)

    @staticmethod
    def get_files(model_path: str, file_ext_list: list) -> dict:
        file_list = []
        for ext in file_ext_list:
            file_list.extend(
                glob.glob(os.path.join(model_path, "**", "*" + ext), recursive=True)
            )

        files_dict = {}
        for file_path in file_list:
            rel_path = os.path.relpath(file_path, model_path)
            display_name = rel_path.replace(os.sep, "/")
            files_dict[display_name] = file_path

        return files_dict

    @staticmethod
    def parse_mask_indices(mask_index: str) -> list:
        if not mask_index or mask_index.strip() == "":
            return [-1]
        mask_index = mask_index.replace("，", ",").replace(" ", ",")
        indices = []
        for idx in mask_index.split(","):
            idx = idx.strip()
            if idx == "-1":
                return [-1]
            elif idx.isdigit():
                indices.append(int(idx))
        return indices if indices else [-1]

    @classmethod
    def _process_image(
        cls,
        yolo_model,
        i: torch.Tensor,
        selected_indices: list,
        show_all: bool,
        threshold: float,
        label: bool,
        label_size: float,
    ):
        i = torch.unsqueeze(i, 0)
        _image = cls.tensor2pil(i)
        results = yolo_model(_image, retina_masks=True, conf=threshold)
        plot_image = np.array(_image)
        image_masks = []
        box_coords = []
        for result in results:
            if result.boxes is not None and len(result.boxes) > 0:
                for index, box in enumerate(result.boxes):
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    box_coords.append((x1, y1, x2, y2))
                    if not show_all and index not in selected_indices:
                        continue
                    class_id = int(box.cls[0].cpu().numpy())
                    confidence = float(box.conf[0].cpu().numpy())
                    class_name = (
                        yolo_model.names[class_id]
                        if class_id in yolo_model.names
                        else f"class_{class_id}"
                    )
                    box_color = (4, 42, 255)
                    box_thickness = max(1, int(3 * label_size))
                    cv2.rectangle(
                        plot_image, (x1, y1), (x2, y2), box_color, box_thickness
                    )
                    if label:
                        custom_label = f"[{index}] {class_name} {confidence:.2f}"
                        font_scale = 0.6 * label_size
                        font_thickness = max(1, int(2 * label_size))
                        font_color = (255, 255, 255)
                        bg_color = (4, 42, 255)
                        (text_width, text_height), baseline = cv2.getTextSize(
                            custom_label,
                            cv2.FONT_HERSHEY_SIMPLEX,
                            font_scale,
                            font_thickness,
                        )
                        label_x = x1
                        fixed_gap = 5
                        label_y = y1 - fixed_gap
                        if label_y - text_height < 0:
                            label_y = y1 + text_height + fixed_gap
                        bg_padding = max(2, int(2 * label_size))
                        cv2.rectangle(
                            plot_image,
                            (
                                label_x - bg_padding,
                                label_y - text_height - bg_padding,
                            ),
                            (
                                label_x + text_width + bg_padding,
                                label_y + baseline + bg_padding,
                            ),
                            bg_color,
                            -1,
                        )
                        cv2.putText(
                            plot_image,
                            custom_label,
                            (label_x, label_y),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            font_scale,
                            font_color,
                            font_thickness,
                        )

            if result.masks is not None and len(result.masks) > 0:
                masks_data = result.masks.data
                for index, mask in enumerate(masks_data):
                    _mask = mask.cpu().numpy()
                    _mask = cls.np2pil((_mask * 255).astype(np.uint8)).convert("L")
                    if _mask.size != _image.size:
                        _mask = _mask.resize(_image.size, Image.NEAREST)
                    image_masks.append(cls.image2mask(_mask))
            if not image_masks and result.boxes is not None and len(result.boxes.xyxy) > 0:
                white_image = Image.new("L", _image.size, "white")
                for index, box in enumerate(result.boxes):
                    if not show_all and index not in selected_indices:
                        continue
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    _mask = Image.new("L", _image.size, "black")
                    _mask.paste(white_image.crop((x1, y1, x2, y2)), (x1, y1))
                    image_masks.append(cls.image2mask(_mask))

        ret_plot = cls.pil2tensor(Image.fromarray(plot_image))
        if len(image_masks) > 0:
            if show_all:
                _mask = image_masks[0]
                for j in range(1, len(image_masks)):
                    _mask = cls.add_mask(_mask, image_masks[j])
                ret_mask = _mask
            else:
                selected_masks = []
                for idx in selected_indices:
                    if idx < len(image_masks):
                        selected_masks.append(image_masks[idx])
                if selected_masks:
                    _mask = selected_masks[0]
                    for j in range(1, len(selected_masks)):
                        _mask = cls.add_mask(_mask, selected_masks[j])
                    ret_mask = _mask
                else:
                    white_image = Image.new("L", _image.size, "white")
                    fallback_masks = []
                    for idx in selected_indices:
                        if 0 <= idx < len(box_coords):
                            x1, y1, x2, y2 = box_coords[idx]
                            _m = Image.new("L", _image.size, "black")
                            _m.paste(white_image.crop((x1, y1, x2, y2)), (x1, y1))
                            fallback_masks.append(cls.image2mask(_m))
                    if fallback_masks:
                        _mask = fallback_masks[0]
                        for j in range(1, len(fallback_masks)):
                            _mask = cls.add_mask(_mask, fallback_masks[j])
                        ret_mask = _mask
                    else:
                        ret_mask = torch.zeros(
                            (1, _image.size[1], _image.size[0]),
                            dtype=torch.float32,
                        )
        else:
            ret_mask = torch.zeros(
                (1, _image.size[1], _image.size[0]), dtype=torch.float32
            )
        return ret_plot, ret_mask

    

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