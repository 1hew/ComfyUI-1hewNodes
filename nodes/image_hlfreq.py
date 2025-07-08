import torch
import numpy as np
import cv2
import colorsys

# 模糊系数常量
BLUR_COEFFICIENT = 1

class ImageHLFreqSeparate:
    """
    高级频率分离节点 - 支持RGB、HSV、IGBI三种方法
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "method": (["rgb", "hsv", "igbi"], {"default": "rgb"}),
                "blur_radius": ("FLOAT", {
                    "default": 10.0,
                    "min": 0.0,
                    "max": 100.0,
                    "step": 0.01
                })
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("high_freq", "low_freq", "combine")
    FUNCTION = "separate_frequency"
    CATEGORY = "1hewNodes/image/hlfreq"

    def _apply_levels_numpy(self, image_array, black_point, white_point, gray_point=1.0, output_black_point=0, output_white_point=255):
        """应用色阶调整（numpy版本）"""
        # 确保值在0-255范围内
        black_point = max(0, min(255, black_point))
        white_point = max(0, min(255, white_point))
        output_black_point = max(0, min(255, output_black_point))
        output_white_point = max(0, min(255, output_white_point))
        
        # 分别处理每个通道
        result_array = np.zeros_like(image_array)
        
        for i in range(image_array.shape[2]):
            channel = image_array[:, :, i].astype(float)
            
            # 应用黑白点调整
            channel = np.clip(channel, black_point, white_point)
            channel = (channel - black_point) / (white_point - black_point) * 255.0
            
            # 应用灰度点调整（gamma校正）
            if gray_point != 1.0:
                channel = 255.0 * (channel / 255.0) ** (1.0 / gray_point)
            
            # 应用输出黑白点调整
            channel = (channel / 255.0) * (output_white_point - output_black_point) + output_black_point
            
            # 确保值在有效范围内
            result_array[:, :, i] = np.clip(channel, 0, 255).astype(np.uint8)
        
        return result_array

    def separate_frequency(self, image, method, blur_radius):
        """
        频率分离处理
        """
        batch_size, height, width, channels = image.shape
        
        # 应用统一的模糊系数
        effective_blur_radius = blur_radius * BLUR_COEFFICIENT
        
        # 转换为numpy数组进行处理
        if image.is_cuda:
            image_np = image.cpu().numpy()
        else:
            image_np = image.numpy()
        
        result_high_freq = []
        result_low_freq = []
        result_combine = []
        
        for i in range(batch_size):
            img = image_np[i]
            
            if method == "igbi":
                # IGBI模式：直接分离
                high_freq, low_freq = self._igbi_separation(img, effective_blur_radius)
                # IGBI重组使用混合+色阶调整方式
                combine = self._igbi_recombine(high_freq, low_freq)
            else:
                # RGB/HSV模式：单次分离
                high_freq, low_freq = self._single_separation(img, method, effective_blur_radius)
                # 根据方法选择重组方式
                if method == "hsv":
                    combine = self._recombine_hsv(high_freq, low_freq)
                else:  # rgb
                    combine = self._recombine_linear_light(high_freq, low_freq)
            
            result_high_freq.append(torch.from_numpy(high_freq))
            result_low_freq.append(torch.from_numpy(low_freq))
            result_combine.append(torch.from_numpy(combine))
        
        # 合并批次
        high_freq_tensor = torch.stack(result_high_freq)
        low_freq_tensor = torch.stack(result_low_freq)
        combine_tensor = torch.stack(result_combine)
        
        return (high_freq_tensor, low_freq_tensor, combine_tensor)
    
    def _igbi_separation(self, image, blur_radius):
        """
        IGBI方法频率分离 - 反转高斯混合再反转
        """
        # 统一使用cv2.GaussianBlur确保模糊度一致
        blur_radius = self._ensure_odd_radius(blur_radius)
        
        # 转换为uint8格式用于cv2处理
        image_uint8 = (image * 255).astype(np.uint8)
        
        # 低频：直接高斯模糊
        low_freq = cv2.GaussianBlur(image_uint8, (blur_radius, blur_radius), 0) / 255.0
        
        # 高频：完整的IGBI处理流程
        # 1. 图像反转
        inverted = 255 - image_uint8
        # 2. 图像高斯模糊
        blurred = cv2.GaussianBlur(image_uint8, (blur_radius, blur_radius), 0)
        # 3. 反转图像与模糊图像混合（normal模式，50%混合）
        mixed = (inverted.astype(np.float32) * 0.5 + blurred.astype(np.float32) * 0.5).astype(np.uint8)
        # 4. 再次反转得到高频
        high_freq = (255 - mixed) / 255.0
        
        return high_freq, low_freq
    
    def _igbi_recombine(self, high_freq, low_freq):
        """
        IGBI重组方式 - 使用混合+色阶调整
        """
        # 转换为uint8格式
        high_freq_uint8 = (high_freq * 255).astype(np.uint8)
        low_freq_uint8 = (low_freq * 255).astype(np.uint8)
        
        # 混合高频和低频图像（65%高频 + 35%低频）
        mixed = (high_freq_uint8.astype(np.float32) * 0.65 + low_freq_uint8.astype(np.float32) * 0.35).astype(np.uint8)
        
        # 应用色阶调整：黑点83，白点172
        result = self._apply_levels_numpy(mixed, 83, 172, 1.0, 0, 255)
        
        return result.astype(np.float32) / 255.0
    
    def _recombine_hsv(self, high_freq, low_freq):
        """HSV重组方式 - 修正为标准算法"""
        # 转换低频到HSV空间
        low_hsv = cv2.cvtColor((low_freq * 255).astype(np.uint8), cv2.COLOR_RGB2HSV)
        h, s, v_low = cv2.split(low_hsv)
        
        # 在V通道进行线性光混合
        high_v = high_freq[:, :, 0]  # 使用高频的第一个通道作为V通道
        v_low_norm = v_low.astype(np.float32) / 255.0
        v_combined = (2 * v_low_norm + high_v - 1)
        v_combined = np.clip(v_combined * 255, 0, 255).astype(np.uint8)
        
        # 重组HSV并转回RGB
        combined_hsv = cv2.merge([h, s, v_combined])
        combined_rgb = cv2.cvtColor(combined_hsv, cv2.COLOR_HSV2RGB)
        
        return combined_rgb.astype(np.float32) / 255.0

        
        return result_array
    
    def _ensure_odd_radius(self, radius):
        """确保半径为奇数（RGB/HSV模式需要）"""
        radius_int = int(radius)
        if radius_int % 2 == 0:
            radius_int += 1
        return max(radius_int, 3)
    
    def _single_separation(self, image, method, blur_radius):
        """单次频率分离"""
        if method == "rgb":
            return self._rgb_separation(image, blur_radius)
        else:  # hsv
            return self._hsv_separation(image, blur_radius)
    
    def _rgb_separation(self, image, blur_radius):
        """RGB方法频率分离 - 修正为标准算法"""
        # RGB/HSV模式需要奇数半径
        blur_radius = self._ensure_odd_radius(blur_radius)
        
        # 低频层：直接对原图进行高斯模糊
        low_freq = cv2.GaussianBlur(image, (blur_radius, blur_radius), 0)
        
        # 高频层：基于灰度信息计算
        gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        blur_gray = cv2.GaussianBlur(gray, (blur_radius, blur_radius), 0)
        high_freq_gray = (gray.astype(np.float32) - blur_gray.astype(np.float32)) / 255.0 + 0.5
        
        # 确保高频层在正确范围内
        high_freq_gray = np.clip(high_freq_gray, 0, 1)
        
        # 将高频层扩展到RGB通道
        high_freq = np.stack([high_freq_gray] * 3, axis=-1)
        
        return high_freq, low_freq
    
    def _hsv_separation(self, image, blur_radius):
        """HSV方法频率分离 - 修正为标准算法"""
        # RGB/HSV模式需要奇数半径
        blur_radius = self._ensure_odd_radius(blur_radius)
        
        # 转换为HSV
        hsv = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(hsv)
        
        # 对V通道进行高斯模糊创建低频V通道
        v_blur = cv2.GaussianBlur(v, (blur_radius, blur_radius), 0)
        
        # 创建高频层：V通道减去模糊V通道
        high_freq_v = (v.astype(np.float32) - v_blur.astype(np.float32)) / 255.0 + 0.5
        high_freq_v = np.clip(high_freq_v, 0, 1)
        
        # 将高频V通道扩展到RGB通道
        high_freq = np.stack([high_freq_v] * 3, axis=-1)
        
        # 创建低频层：用模糊的V通道替换原V通道，然后转回RGB
        low_freq_hsv = hsv.copy()
        low_freq_hsv[..., 2] = v_blur
        low_freq = cv2.cvtColor(low_freq_hsv, cv2.COLOR_HSV2RGB) / 255.0
        
        return high_freq, low_freq
    

    
    def _recombine_linear_light(self, high_freq, low_freq):
        """Linear Light重组高低频图像"""
        # 使用 linear light 公式：2 * high_freq + low_freq - 1
        result = 2 * high_freq + low_freq - 1
        return np.clip(result, 0, 1)
    
    def _apply_levels_numpy(self, image_array, black_point, white_point, gray_point=1.0, output_black_point=0, output_white_point=255):
        """应用色阶调整（numpy版本）"""
        # 确保值在0-255范围内
        black_point = max(0, min(255, black_point))
        white_point = max(0, min(255, white_point))
        output_black_point = max(0, min(255, output_black_point))
        output_white_point = max(0, min(255, output_white_point))
        
        # 分别处理每个通道
        result_array = np.zeros_like(image_array)
        
        for i in range(image_array.shape[2]):
            channel = image_array[:, :, i].astype(float)
            
            # 应用黑白点调整
            channel = np.clip(channel, black_point, white_point)
            channel = (channel - black_point) / (white_point - black_point) * 255.0
            
            # 应用灰度点调整（gamma校正）
            if gray_point != 1.0:
                channel = 255.0 * (channel / 255.0) ** (1.0 / gray_point)
            
            # 应用输出黑白点调整
            channel = (channel / 255.0) * (output_white_point - output_black_point) + output_black_point
            
            # 确保值在有效范围内
            result_array[:, :, i] = np.clip(channel, 0, 255).astype(np.uint8)
        
        return result_array
    
    def _apply_levels_numpy(self, image_array, black_point, white_point, gray_point=1.0, output_black_point=0, output_white_point=255):
        """应用色阶调整（numpy版本）"""
        # 确保值在0-255范围内
        black_point = max(0, min(255, black_point))
        white_point = max(0, min(255, white_point))
        output_black_point = max(0, min(255, output_black_point))
        output_white_point = max(0, min(255, output_white_point))
        
        # 分别处理每个通道
        result_array = np.zeros_like(image_array)
        
        for i in range(image_array.shape[2]):
            channel = image_array[:, :, i].astype(float)
            
            # 应用黑白点调整
            channel = np.clip(channel, black_point, white_point)
            channel = (channel - black_point) / (white_point - black_point) * 255.0
            
            # 应用灰度点调整（gamma校正）
            if gray_point != 1.0:
                channel = 255.0 * (channel / 255.0) ** (1.0 / gray_point)
            
            # 应用输出黑白点调整
            channel = (channel / 255.0) * (output_white_point - output_black_point) + output_black_point
            
            # 确保值在有效范围内
            result_array[:, :, i] = np.clip(channel, 0, 255).astype(np.uint8)
        
        return result_array


class ImageHLFreqCombine:
    """高级频率重组节点 - 支持多种混合模式"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "high_freq": ("IMAGE",),
                "low_freq": ("IMAGE",),
                "method": (["rgb", "hsv", "igbi"], {"default": "rgb"}),
                "high_strength": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.01
                }),
                "low_strength": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.01
                })
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "combine_frequency"
    CATEGORY = "1hewNodes/image/hlfreq"

    def combine_frequency(self, high_freq, low_freq, method, high_strength, low_strength):
        """高级频率重组处理"""
        # 处理批量不匹配：自动复制较少的批次来匹配较多的批次
        high_batch_size = high_freq.shape[0]
        low_batch_size = low_freq.shape[0]
        max_batch_size = max(high_batch_size, low_batch_size)
        
        # 如果批次数量不匹配，复制较少的批次
        if high_batch_size < max_batch_size:
            # 复制 high_freq 到匹配的批次数量
            repeat_times = max_batch_size // high_batch_size
            remainder = max_batch_size % high_batch_size
            if remainder > 0:
                high_freq = torch.cat([high_freq.repeat(repeat_times, 1, 1, 1), high_freq[:remainder]], dim=0)
            else:
                high_freq = high_freq.repeat(repeat_times, 1, 1, 1)
        
        if low_batch_size < max_batch_size:
            # 复制 low_freq 到匹配的批次数量
            repeat_times = max_batch_size // low_batch_size
            remainder = max_batch_size % low_batch_size
            if remainder > 0:
                low_freq = torch.cat([low_freq.repeat(repeat_times, 1, 1, 1), low_freq[:remainder]], dim=0)
            else:
                low_freq = low_freq.repeat(repeat_times, 1, 1, 1)
        
        # 调整频率强度
        if method in ["rgb", "hsv"]:
            adjusted_high_freq = (high_freq - 0.5) * high_strength + 0.5
        else:  # igbi
            adjusted_high_freq = high_freq * high_strength
            
        adjusted_low_freq = low_freq * low_strength
        
        # 确保数值范围
        adjusted_high_freq = torch.clamp(adjusted_high_freq, 0, 1)
        adjusted_low_freq = torch.clamp(adjusted_low_freq, 0, 1)
        
        # 转换为numpy进行混合
        if adjusted_high_freq.is_cuda:
            high_np = adjusted_high_freq.cpu().numpy()
            low_np = adjusted_low_freq.cpu().numpy()
        else:
            high_np = adjusted_high_freq.numpy()
            low_np = adjusted_low_freq.numpy()
        
        batch_size = max_batch_size
        result_images = []
        
        for i in range(batch_size):
            # 根据方法选择重组方式
            if method == "rgb":
                result = self._recombine_linear_light(high_np[i], low_np[i])
            elif method == "hsv":
                result = self._recombine_hsv(high_np[i], low_np[i])
            else:  # igbi
                result = self._recombine_igbi(high_np[i], low_np[i])
            result_images.append(torch.from_numpy(result))
        
        result_tensor = torch.stack(result_images)
        return (result_tensor,)
    
    def _recombine_linear_light(self, high_freq, low_freq):
        """Linear Light重组高低频图像（RGB模式）"""
        # 使用 linear light 公式：2 * high_freq + low_freq - 1
        result = 2 * high_freq + low_freq - 1
        return np.clip(result, 0, 1)
    
    def _recombine_hsv(self, high_freq, low_freq):
        """HSV重组方式 - 修正为标准算法"""
        # 转换低频到HSV空间
        low_hsv = cv2.cvtColor((low_freq * 255).astype(np.uint8), cv2.COLOR_RGB2HSV)
        h, s, v_low = cv2.split(low_hsv)
        
        # 在V通道进行线性光混合
        high_v = high_freq[:, :, 0]  # 使用高频的第一个通道作为V通道
        v_low_norm = v_low.astype(np.float32) / 255.0
        v_combined = (2 * v_low_norm + high_v - 1)
        v_combined = np.clip(v_combined * 255, 0, 255).astype(np.uint8)
        
        # 重组HSV并转回RGB
        combined_hsv = cv2.merge([h, s, v_combined])
        combined_rgb = cv2.cvtColor(combined_hsv, cv2.COLOR_HSV2RGB)
        
        return combined_rgb.astype(np.float32) / 255.0
    
    def _recombine_igbi(self, high_freq, low_freq):
        """IGBI重组方式 - 使用混合+色阶调整"""
        # 转换为uint8格式
        high_freq_uint8 = (high_freq * 255).astype(np.uint8)
        low_freq_uint8 = (low_freq * 255).astype(np.uint8)
        
        # 混合高频和低频图像（65%高频 + 35%低频）
        mixed = (high_freq_uint8.astype(np.float32) * 0.65 + low_freq_uint8.astype(np.float32) * 0.35).astype(np.uint8)
        
        # 应用色阶调整：黑点83，白点172
        result = self._apply_levels_numpy(mixed, 83, 172, 1.0, 0, 255)
        
        return result.astype(np.float32) / 255.0
    
    def _apply_levels_numpy(self, image_array, black_point, white_point, gray_point=1.0, output_black_point=0, output_white_point=255):
        """应用色阶调整（numpy版本）"""
        # 确保值在0-255范围内
        black_point = max(0, min(255, black_point))
        white_point = max(0, min(255, white_point))
        output_black_point = max(0, min(255, output_black_point))
        output_white_point = max(0, min(255, output_white_point))
        
        # 分别处理每个通道
        result_array = np.zeros_like(image_array)
        
        for i in range(image_array.shape[2]):
            channel = image_array[:, :, i].astype(float)
            
            # 应用黑白点调整
            channel = np.clip(channel, black_point, white_point)
            channel = (channel - black_point) / (white_point - black_point) * 255.0
            
            # 应用灰度点调整（gamma校正）
            if gray_point != 1.0:
                channel = 255.0 * (channel / 255.0) ** (1.0 / gray_point)
            
            # 应用输出黑白点调整
            channel = (channel / 255.0) * (output_white_point - output_black_point) + output_black_point
            
            # 确保值在有效范围内
            result_array[:, :, i] = np.clip(channel, 0, 255).astype(np.uint8)
        
        return result_array


class ImageHLFreqTransform:
    """高级细节迁移节点 - 支持RGB/HSV/IGBI三种频率分离方法"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "generate_image": ("IMAGE",),
                "detail_image": ("IMAGE",),
                "method": (["rgb", "hsv", "igbi"], {"default": "igbi"}),
                "blur_radius": ("FLOAT", {
                    "default": 10.0,
                    "min": 0.0,
                    "max": 100.0,
                    "step": 0.01
                })
            },
            "optional": {
                "detail_mask": ("MASK",),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("image", "high_freq", "low_freq")
    FUNCTION = "transfer_detail_advanced"
    CATEGORY = "1hewNodes/image/hlfreq"



    def transfer_detail_advanced(self, generate_image, detail_image, method, blur_radius, detail_mask=None):
        """高级细节迁移处理 - 支持三种频率分离方法"""
        # 处理批量不匹配：自动复制较少的批次来匹配较多的批次
        generate_batch_size = generate_image.shape[0]
        detail_batch_size = detail_image.shape[0]
        max_batch_size = max(generate_batch_size, detail_batch_size)
        
        # 如果批次数量不匹配，复制较少的批次
        if generate_batch_size < max_batch_size:
            # 复制 generate_image 到匹配的批次数量
            repeat_times = max_batch_size // generate_batch_size
            remainder = max_batch_size % generate_batch_size
            if remainder > 0:
                generate_image = torch.cat([generate_image.repeat(repeat_times, 1, 1, 1), generate_image[:remainder]], dim=0)
            else:
                generate_image = generate_image.repeat(repeat_times, 1, 1, 1)
        
        if detail_batch_size < max_batch_size:
            # 复制 detail_image 到匹配的批次数量
            repeat_times = max_batch_size // detail_batch_size
            remainder = max_batch_size % detail_batch_size
            if remainder > 0:
                detail_image = torch.cat([detail_image.repeat(repeat_times, 1, 1, 1), detail_image[:remainder]], dim=0)
            else:
                detail_image = detail_image.repeat(repeat_times, 1, 1, 1)
        
        # 处理 detail_mask 的批量不匹配
        if detail_mask is not None:
            mask_batch_size = detail_mask.shape[0]
            if mask_batch_size < max_batch_size:
                # 复制 detail_mask 到匹配的批次数量
                repeat_times = max_batch_size // mask_batch_size
                remainder = max_batch_size % mask_batch_size
                if remainder > 0:
                    detail_mask = torch.cat([detail_mask.repeat(repeat_times, 1, 1), detail_mask[:remainder]], dim=0)
                else:
                    detail_mask = detail_mask.repeat(repeat_times, 1, 1)
        
        batch_size, height, width, channels = detail_image.shape
        
        # 应用统一的模糊系数
        effective_blur_radius = blur_radius * BLUR_COEFFICIENT
        
        # 转换为numpy数组进行处理
        if detail_image.is_cuda:
            detail_np = detail_image.cpu().numpy()
            generated_np = generate_image.cpu().numpy()
            if detail_mask is not None:
                mask_np = detail_mask.cpu().numpy()
        else:
            detail_np = detail_image.numpy()
            generated_np = generate_image.numpy()
            if detail_mask is not None:
                mask_np = detail_mask.numpy()
        
        # 处理mask：如果没有提供mask，创建全白mask
        if detail_mask is None:
            mask_3ch = np.ones((batch_size, height, width, 3), dtype=np.float32)
        else:
            # 扩展mask到3通道
            if len(mask_np.shape) == 3:  # [batch, height, width]
                mask_3ch = np.expand_dims(mask_np, axis=-1)
                mask_3ch = np.repeat(mask_3ch, 3, axis=-1)
            else:  # [batch, height, width, 1]
                mask_3ch = np.repeat(mask_np, 3, axis=-1)
        
        result_images = []
        high_freq_images = []
        low_freq_images = []
        
        for i in range(batch_size):
            detail_img = detail_np[i]
            generated_img = generated_np[i]
            current_mask = mask_3ch[i]
            
            if method == "igbi":
                # IGBI模式：直接处理
                result, high_freq, low_freq = self._igbi_detail_transfer(
                    generated_img, detail_img, current_mask, effective_blur_radius
                )
            else:
                # RGB/HSV模式：单次处理
                result, high_freq, low_freq = self._traditional_detail_transfer(
                    generated_img, detail_img, current_mask, method, effective_blur_radius
                )
            
            result_images.append(torch.from_numpy(result))
            high_freq_images.append(torch.from_numpy(high_freq))
            low_freq_images.append(torch.from_numpy(low_freq))
        
        # 合并批次
        result_tensor = torch.stack(result_images)
        high_freq_tensor = torch.stack(high_freq_images)
        low_freq_tensor = torch.stack(low_freq_images)
        
        return (result_tensor, high_freq_tensor, low_freq_tensor)
    
    def _igbi_detail_transfer(self, generate_img, detail_img, mask, blur_radius):
        """
        IGBI模式细节迁移 - 修正版本
        低频图片：生成图像高斯模糊后的结果
        高频图片：生成图像和细节图像反转混合再反转后通过mask混合的图像
        """
        # 转换为uint8格式用于cv2处理
        generate_uint8 = (generate_img * 255).astype(np.uint8)
        detail_uint8 = (detail_img * 255).astype(np.uint8)
        mask_single = mask[:,:,0]  # 使用单通道mask
        
        # 确保blur_radius为奇数
        blur_radius = self._ensure_odd_radius(blur_radius)
        
        # === 低频图片：生成图像高斯模糊后的结果 ===
        low_freq_uint8 = cv2.GaussianBlur(generate_uint8, (blur_radius, blur_radius), 0)
        
        # === 高频图片计算流程 ===
        # 生成图像处理流程：
        # 1. 生成图像反转
        generate_inverted = 255 - generate_uint8
        # 2. 生成图像高斯模糊
        generate_blurred = cv2.GaussianBlur(generate_uint8, (blur_radius, blur_radius), 0)
        # 3. 反转图像与模糊图像混合（normal模式，50%混合）
        generate_mixed = (generate_inverted.astype(np.float32) * 0.5 + generate_blurred.astype(np.float32) * 0.5).astype(np.uint8)
        # 4. 再次反转得到图像a
        image_a = 255 - generate_mixed
        
        # 细节图像处理流程：
        # 1. 细节图像反转
        detail_inverted = 255 - detail_uint8
        # 2. 细节图像高斯模糊
        detail_blurred = cv2.GaussianBlur(detail_uint8, (blur_radius, blur_radius), 0)
        # 3. 反转图像与模糊图像混合（normal模式，50%混合）
        detail_mixed = (detail_inverted.astype(np.float32) * 0.5 + detail_blurred.astype(np.float32) * 0.5).astype(np.uint8)
        # 4. 再次反转得到图像b
        image_b = 255 - detail_mixed
        
        # 5. 通过mask混合图像a和图像b得到高频图像
        # 扩展mask到3通道
        mask_3ch = np.stack([mask_single] * 3, axis=-1)
        high_freq_uint8 = (image_a.astype(np.float32) * (1 - mask_3ch) + image_b.astype(np.float32) * mask_3ch).astype(np.uint8)
        
        # === 最终结果计算 ===
        # 将高频图像与低频图像混合得到最终结果（65%高频 + 35%低频）
        result_mixed = (high_freq_uint8.astype(np.float32) * 0.65 + low_freq_uint8.astype(np.float32) * 0.35).astype(np.uint8)
        
        # 应用色阶调整得到最终结果
        result_uint8 = self._apply_levels_numpy(result_mixed, 83, 172, 1.0, 0, 255)
        
        # 转换回float32格式
        result = result_uint8.astype(np.float32) / 255.0
        high_freq = high_freq_uint8.astype(np.float32) / 255.0  # 高频图像
        low_freq = low_freq_uint8.astype(np.float32) / 255.0    # 低频图像（生成图像模糊结果）
        
        return result, high_freq, low_freq
    

    

    
    def _traditional_detail_transfer(self, generated_img, detail_img, mask, method, base_radius):
        """传统RGB/HSV模式细节迁移"""
        # RGB/HSV模式需要确保blur_radius为奇数
        blur_radius = self._ensure_odd_radius(base_radius)
        generated_blur_radius = self._ensure_odd_radius(base_radius * 1.5)
        
        # 单次处理
        detail_high_freq, _ = self._single_separation(detail_img, method, blur_radius)
        generated_high_freq, generated_low_freq = self._single_separation(
            generated_img, method, generated_blur_radius
        )
        
        # 在高频层进行mask控制的细节迁移
        mixed_high_freq = generated_high_freq * (1 - mask) + detail_high_freq * mask
        
        # 重组图像
        if method == "hsv":
            result = self._recombine_hsv(mixed_high_freq, generated_low_freq)
        else:
            result = self._recombine_linear_light(mixed_high_freq, generated_low_freq)
        result = np.clip(result, 0, 1)
        
        return result, mixed_high_freq, generated_low_freq
    
    def _recombine_hsv(self, high_freq, low_freq):
        """HSV重组方式 - 修正为标准算法"""
        # 转换低频到HSV空间
        low_hsv = cv2.cvtColor((low_freq * 255).astype(np.uint8), cv2.COLOR_RGB2HSV)
        h, s, v_low = cv2.split(low_hsv)
        
        # 在V通道进行线性光混合
        high_v = high_freq[:, :, 0]  # 使用高频的第一个通道作为V通道
        v_low_norm = v_low.astype(np.float32) / 255.0
        v_combined = (2 * v_low_norm + high_v - 1)
        v_combined = np.clip(v_combined * 255, 0, 255).astype(np.uint8)
        
        # 重组HSV并转回RGB
        combined_hsv = cv2.merge([h, s, v_combined])
        combined_rgb = cv2.cvtColor(combined_hsv, cv2.COLOR_HSV2RGB)
        
        return combined_rgb.astype(np.float32) / 255.0
    
    def _ensure_odd_radius(self, radius):
        """确保半径为奇数"""
        radius_int = int(radius)
        if radius_int % 2 == 0:
            radius_int += 1
        return max(radius_int, 3)
    
    def _single_separation(self, image, method, blur_radius):
        """单次频率分离"""
        if method == "rgb":
            return self._rgb_separation(image, blur_radius)
        else:  # hsv
            return self._hsv_separation(image, blur_radius)
    
    def _rgb_separation(self, image, blur_radius):
        """RGB方法频率分离 - 修正为标准算法"""
        # RGB/HSV模式需要奇数半径
        blur_radius = self._ensure_odd_radius(blur_radius)
        
        # 低频层：直接对原图进行高斯模糊
        low_freq = cv2.GaussianBlur(image, (blur_radius, blur_radius), 0)
        
        # 高频层：基于灰度信息计算
        gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        blur_gray = cv2.GaussianBlur(gray, (blur_radius, blur_radius), 0)
        high_freq_gray = (gray.astype(np.float32) - blur_gray.astype(np.float32)) / 255.0 + 0.5
        
        # 确保高频层在正确范围内
        high_freq_gray = np.clip(high_freq_gray, 0, 1)
        
        # 将高频层扩展到RGB通道
        high_freq = np.stack([high_freq_gray] * 3, axis=-1)
        
        return high_freq, low_freq
    
    def _hsv_separation(self, image, blur_radius):
        """HSV方法频率分离 - 修正为标准算法"""
        # RGB/HSV模式需要奇数半径
        blur_radius = self._ensure_odd_radius(blur_radius)
        
        # 转换为HSV
        hsv = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(hsv)
        
        # 对V通道进行高斯模糊创建低频V通道
        v_blur = cv2.GaussianBlur(v, (blur_radius, blur_radius), 0)
        
        # 创建高频层：V通道减去模糊V通道
        high_freq_v = (v.astype(np.float32) - v_blur.astype(np.float32)) / 255.0 + 0.5
        high_freq_v = np.clip(high_freq_v, 0, 1)
        
        # 将高频V通道扩展到RGB通道
        high_freq = np.stack([high_freq_v] * 3, axis=-1)
        
        # 创建低频层：用模糊的V通道替换原V通道，然后转回RGB
        low_freq_hsv = hsv.copy()
        low_freq_hsv[..., 2] = v_blur
        low_freq = cv2.cvtColor(low_freq_hsv, cv2.COLOR_HSV2RGB) / 255.0
        
        return high_freq, low_freq
    

    
    def _recombine_linear_light(self, high_freq, low_freq):
        """Linear Light重组高低频图像"""
        # 使用 linear light 公式：2 * high_freq + low_freq - 1
        result = 2 * high_freq + low_freq - 1
        return np.clip(result, 0, 1)
    
    def _apply_levels_numpy(self, image_array, black_point, white_point, gray_point=1.0, output_black_point=0, output_white_point=255):
        """应用色阶调整（numpy版本）"""
        # 确保值在0-255范围内
        black_point = max(0, min(255, black_point))
        white_point = max(0, min(255, white_point))
        output_black_point = max(0, min(255, output_black_point))
        output_white_point = max(0, min(255, output_white_point))
        
        # 分别处理每个通道
        result_array = np.zeros_like(image_array)
        
        for i in range(image_array.shape[2]):
            channel = image_array[:, :, i].astype(float)
            
            # 应用黑白点调整
            channel = np.clip(channel, black_point, white_point)
            channel = (channel - black_point) / (white_point - black_point) * 255.0
            
            # 应用灰度点调整（gamma校正）
            if gray_point != 1.0:
                channel = 255.0 * (channel / 255.0) ** (1.0 / gray_point)
            
            # 应用输出黑白点调整
            channel = (channel / 255.0) * (output_white_point - output_black_point) + output_black_point
            
            # 确保值在有效范围内
            result_array[:, :, i] = np.clip(channel, 0, 255).astype(np.uint8)
        
        return result_array


# 更新节点注册
NODE_CLASS_MAPPINGS = {
    "ImageHLFreqSeparate": ImageHLFreqSeparate,
    "ImageHLFreqCombine": ImageHLFreqCombine,
    "ImageHLFreqTransform": ImageHLFreqTransform,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageHLFreqSeparate": "Image HL Freq Separate",
    "ImageHLFreqCombine": "Image HL Freq Combine",
    "ImageHLFreqTransform": "Image HL Freq Transform",
}