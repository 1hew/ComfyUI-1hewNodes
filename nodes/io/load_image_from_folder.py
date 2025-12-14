from __future__ import annotations
import os
import torch
import numpy as np
import torch.nn.functional as F
from PIL import Image, ImageOps
from comfy_api.latest import io, ui
from server import PromptServer
from aiohttp import web
import math
from io import BytesIO

import hashlib

class LoadImageFromFolder(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="1hew_LoadImageFromFolder",
            display_name="Load Image From Folder",
            category="1hewNodes/io",
            inputs=[
                io.Image.Input("get_image_size", optional=True),
                io.String.Input("folder", default=""),
                io.Int.Input("index", default=0, min=-8192, max=8192, step=1),
                io.Boolean.Input("include_subfolder", default=True),
                io.Boolean.Input("all", default=False),
            ],
            outputs=[
                io.Image.Output(display_name="image"),
                io.Mask.Output(display_name="mask"),
            ],
        )

    @staticmethod
    def load_image(path):
        img = Image.open(path)
        img = ImageOps.exif_transpose(img)
        
        mask = None
        if 'A' in img.getbands():
            mask_np = np.array(img.getchannel('A')).astype(np.float32) / 255.0
            mask = torch.from_numpy(mask_np)
        else:
            mask = torch.ones((img.height, img.width), dtype=torch.float32)
            
        img = img.convert("RGB")
        return img, mask

    @staticmethod
    def pil2tensor(image):
        return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

    @staticmethod
    def tensor2pil(image):
        return Image.fromarray(np.clip(255.0 * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

    @staticmethod
    def crop_and_resize(tensor, target_h, target_w):
        """
        先按比例裁剪（Center Crop），再缩放到目标尺寸。
        输入 tensor shape: [1, H, W, C]
        """
        if tensor.shape[1] == target_h and tensor.shape[2] == target_w:
            return tensor

        # 转换为 PIL 以利用 Pillow 的处理逻辑，或者手动计算 tensor 操作
        # 这里为了简单复用逻辑，可以手动计算 tensor 的切片
        
        curr_h, curr_w = tensor.shape[1], tensor.shape[2]
        curr_ratio = curr_w / curr_h
        target_ratio = target_w / target_h

        # Permute to [B, C, H, W] for processing
        t = tensor.permute(0, 3, 1, 2)
        
        if curr_ratio > target_ratio:
            # 原图更宽，以高为基准，裁剪宽度
            # new_w = curr_h * target_ratio
            # 实际上我们需要保留的高度就是 curr_h
            # 需要保留的宽度
            crop_w = int(curr_h * target_ratio)
            crop_h = curr_h
        else:
            # 原图更高，以宽为基准，裁剪高度
            crop_w = curr_w
            crop_h = int(curr_w / target_ratio)

        # Center Crop
        start_x = (curr_w - crop_w) // 2
        start_y = (curr_h - crop_h) // 2
        
        # [B, C, H, W]
        t = t[:, :, start_y:start_y+crop_h, start_x:start_x+crop_w]
        
        # Resize
        t = F.interpolate(t, size=(target_h, target_w), mode="bilinear", align_corners=False)
        
        # Permute back to [B, H, W, C]
        return t.permute(0, 2, 3, 1)

    @staticmethod
    def get_image_paths(folder, include_subfolder):
        if not os.path.isdir(folder):
            return []
            
        valid_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.webp', '.tiff', '.gif'}
        image_paths = []

        if include_subfolder:
            for root, dirs, files in os.walk(folder):
                for file in files:
                    if os.path.splitext(file)[1].lower() in valid_extensions:
                        image_paths.append(os.path.join(root, file))
        else:
            for file in os.listdir(folder):
                path = os.path.join(folder, file)
                if os.path.isfile(path) and os.path.splitext(file)[1].lower() in valid_extensions:
                    image_paths.append(path)

        # Case-insensitive sort for cross-platform consistency
        image_paths.sort(key=lambda x: x.lower())
        return image_paths

    @classmethod
    def IS_CHANGED(cls, folder, include_subfolder, **kwargs):
        if not os.path.isdir(folder):
            return float("nan")
            
        image_paths = cls.get_image_paths(folder, include_subfolder)
        m = hashlib.sha256()
        for path in image_paths:
            try:
                mtime = os.path.getmtime(path)
                m.update(f"{path}:{mtime}".encode())
            except OSError:
                continue
                
        return m.hexdigest()

    @staticmethod
    def crop_and_resize_pil(pil_img, target_w, target_h):
        """
        PIL 版本的 crop_and_resize，保持与 tensor 版本逻辑一致。
        """
        curr_w, curr_h = pil_img.size
        if curr_w == target_w and curr_h == target_h:
            return pil_img
            
        curr_ratio = curr_w / curr_h
        target_ratio = target_w / target_h
        
        if curr_ratio > target_ratio:
            # 原图更宽，裁剪宽度
            crop_w = int(curr_h * target_ratio)
            crop_h = curr_h
        else:
            # 原图更高，裁剪高度
            crop_w = curr_w
            crop_h = int(curr_w / target_ratio)
            
        # Center Crop
        left = (curr_w - crop_w) // 2
        top = (curr_h - crop_h) // 2
        right = left + crop_w
        bottom = top + crop_h
        
        img_cropped = pil_img.crop((left, top, right, bottom))
        img_resized = img_cropped.resize((target_w, target_h), Image.Resampling.BILINEAR)
        return img_resized

    @classmethod
    async def execute(cls, folder: str, index: int, all: bool, include_subfolder: bool, get_image_size: torch.Tensor | None = None) -> io.NodeOutput:
        image_paths = cls.get_image_paths(folder, include_subfolder)
        count = len(image_paths)

        if count == 0:
            return io.NodeOutput(None, None)

        target_h = 0
        target_w = 0

        # 如果提供了参考图片，优先使用参考图片的尺寸
        if get_image_size is not None:
            target_h, target_w = get_image_size.shape[1:3]

        if all:
            images_tensors = []
            masks_tensors = []
            
            # 如果没有参考图片，加载第一张图片来确定基准尺寸
            start_idx = 0
            if target_h == 0 or target_w == 0:
                try:
                    first_img, first_mask = cls.load_image(image_paths[0])
                    first_tensor = cls.pil2tensor(first_img)
                    first_mask_tensor = first_mask.unsqueeze(0).unsqueeze(-1) # [1, H, W, 1]
                    
                    target_h, target_w = first_tensor.shape[1:3]
                    images_tensors.append(first_tensor)
                    masks_tensors.append(first_mask_tensor)
                    start_idx = 1
                except Exception as e:
                    print(f"Error loading first image {image_paths[0]}: {e}")
                    return io.NodeOutput(None, None)
            
            for path in image_paths[start_idx:]:
                try:
                    img, mask = cls.load_image(path)
                    tensor = cls.pil2tensor(img)
                    mask_tensor = mask.unsqueeze(0).unsqueeze(-1)
                    
                    # 裁剪并缩放
                    tensor = cls.crop_and_resize(tensor, target_h, target_w)
                    mask_tensor = cls.crop_and_resize(mask_tensor, target_h, target_w)
                    
                    images_tensors.append(tensor)
                    masks_tensors.append(mask_tensor)
                except Exception as e:
                    print(f"Error loading image {path}: {e}")
            
            if not images_tensors:
                return io.NodeOutput(None, None)
                
            output_image = torch.cat(images_tensors, dim=0)
            output_mask = torch.cat(masks_tensors, dim=0).squeeze(-1) # [B, H, W]

        else:
            idx = index % count
            path = image_paths[idx]
            try:
                img, mask = cls.load_image(path)
                tensor = cls.pil2tensor(img)
                mask_tensor = mask.unsqueeze(0).unsqueeze(-1)
                
                # 如果有参考尺寸，单张图片也进行裁剪缩放
                if target_h > 0 and target_w > 0:
                    tensor = cls.crop_and_resize(tensor, target_h, target_w)
                    mask_tensor = cls.crop_and_resize(mask_tensor, target_h, target_w)
                
                output_image = tensor
                output_mask = mask_tensor.squeeze(-1) # [1, H, W]
            except Exception as e:
                print(f"Error loading image {path}: {e}")
                return io.NodeOutput(None, None)

        return io.NodeOutput(output_image, output_mask, ui=ui.PreviewImage(output_image, cls=cls))


@PromptServer.instance.routes.get("/1hew/view_image_from_folder")
async def view_image_from_folder(request):
    folder = request.query.get("folder")
    index_str = request.query.get("index", "0")
    include_subfolder = request.query.get("include_subfolder", "true").lower() == "true"
    all_images = request.query.get("all", "false").lower() == "true"
    return_list = request.query.get("return_list", "false").lower() == "true"
    
    if not folder or not os.path.isdir(folder):
        return web.Response(status=404)
        
    try:
        index = int(index_str)
    except ValueError:
        index = 0

    image_paths = LoadImageFromFolder.get_image_paths(folder, include_subfolder)
    if not image_paths:
        return web.Response(status=404)
        
    if return_list:
         # 返回图片列表信息
         return web.json_response({
             "count": len(image_paths),
             "paths": image_paths
         })

    if all_images:
        # 批量预览模式：生成网格图
        # 限制预览数量，避免过慢
        max_preview = 200 # 增加到 200 张
        paths_to_show = image_paths[:max_preview]
        
        if not paths_to_show:
             return web.Response(status=404)

        images = []
        
        # 确定基准尺寸：读取第一张图
        try:
            first_img, _ = LoadImageFromFolder.load_image(paths_to_show[0])
            # 计算预览用的目标尺寸，限制最大边长，但保持长宽比
            base_w, base_h = first_img.size
            max_side = 256
            scale = min(max_side / base_w, max_side / base_h)
            # 至少为 1
            if scale > 1: scale = 1 
            
            target_w = int(base_w * scale)
            target_h = int(base_h * scale)
            
            # 处理第一张图
            images.append(LoadImageFromFolder.crop_and_resize_pil(first_img, target_w, target_h))
            
            # 处理后续图片
            for p in paths_to_show[1:]:
                try:
                    img, _ = LoadImageFromFolder.load_image(p)
                    # 使用与 execute 相同的 crop_and_resize 逻辑
                    processed_img = LoadImageFromFolder.crop_and_resize_pil(img, target_w, target_h)
                    images.append(processed_img)
                except Exception:
                    continue
                    
        except Exception as e:
             print(f"Preview error: {e}")
             return web.Response(status=404)
                
        if not images:
            return web.Response(status=404)
            
        # 计算网格行列
        count = len(images)
        cols = math.ceil(math.sqrt(count))
        rows = math.ceil(count / cols)
        
        # 创建网格背景
        # 既然所有图片都 resize 到了 target_w, target_h，这里直接用
        cell_w = target_w
        cell_h = target_h
        # ComfyUI 原生 Preview Image 如果是 batch，通常是紧密排列或有很小间距
        # 这里为了对齐效果，我们不加额外大间距，或者只加一点点
        spacing = 0 
        
        grid_w = cols * cell_w + (cols - 1) * spacing
        grid_h = rows * cell_h + (rows - 1) * spacing
        
        # 使用黑色背景，类似 ComfyUI
        grid_img = Image.new('RGB', (grid_w, grid_h), (0, 0, 0))
        
        for i, img in enumerate(images):
            r = i // cols
            c = i % cols
            x = c * (cell_w + spacing)
            y = r * (cell_h + spacing)
            grid_img.paste(img, (x, y))
            
        # 保存到内存流
        stream = BytesIO()
        grid_img.save(stream, format="JPEG", quality=85)
        return web.Response(body=stream.getvalue(), content_type='image/jpeg')

    # 单张预览模式
    idx = index % len(image_paths)
    path = image_paths[idx]
    
    return web.FileResponse(path)
