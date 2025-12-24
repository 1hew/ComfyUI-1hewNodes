import asyncio
import os
import torch
from comfy_api.latest import io


class MatchBrightnessContrast(io.ComfyNode):
    """
    匹配亮度与对比度
    
    调整源图像(source_image)的亮度和对比度以匹配参考图像(reference_image)。
    可选择仅使用边缘区域进行统计计算，以忽略中心内容的变化。
    """
    
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="1hew_MatchBrightnessContrast",
            display_name="Match Brightness Contrast",
            category="1hewNodes/color",
            inputs=[
                io.Image.Input("source_image"),
                io.Image.Input("reference_image"),
                io.Float.Input("edge_amount", default=0.2, min=0.0, max=8192.0, step=0.01, display_mode=io.NumberDisplay.number),
                io.Combo.Input("consistency", options=["lock_first", "frame_match"], default="lock_first"),
                io.Combo.Input("method", options=["standard", "histogram"], default="histogram"),
            ],
            outputs=[
                io.Image.Output(display_name="image"),
            ],
        )

    @classmethod
    async def execute(cls, source_image, reference_image, edge_amount, method="histogram", consistency="lock_first"):
        # source_image, reference_image: [B, H, W, C]
        
        # 1. 全局设备对齐：确保参考图与源图在同一设备，避免循环内重复搬运
        if reference_image.device != source_image.device:
            reference_image = reference_image.to(source_image.device)
            
        src_batch = source_image.shape[0]
        ref_batch = reference_image.shape[0]
        
        # 2. 全局参数计算：Batch 内图像尺寸一致，只需计算一次 Margin
        h_src, w_src = source_image.shape[1:3]
        margin_src = cls._calculate_margin(edge_amount, h_src, w_src)
        
        h_ref, w_ref = reference_image.shape[1:3]
        margin_ref = cls._calculate_margin(edge_amount, h_ref, w_ref)
        
        # 预计算第一帧的统计信息或映射表 (如果开启了时序一致性)
        locked_params = None
        if consistency == "lock_first" and src_batch > 0:
            # 使用第一帧源图像和对应的参考图像计算参数
            ref_idx = 0
            ref_img = reference_image[ref_idx % ref_batch]
            src_img = source_image[0]
            
            if method == "histogram":
                locked_params = cls._calculate_histogram_luts(src_img, ref_img, margin_src, margin_ref)
            else:
                locked_params = cls._calculate_standard_stats(src_img, ref_img, margin_src, margin_ref)

        # 确定并发度 (CPU 核心数或批次大小的较小值)
        concurrency = max(1, min(src_batch, os.cpu_count() or 1))
        sem = asyncio.Semaphore(concurrency)
        tasks = []

        for i in range(src_batch):
            # 获取对应的参考图像 (处理批次不匹配情况)
            ref_idx = i % ref_batch
            ref_img = reference_image[ref_idx]
            src_img = source_image[i]

            async def run_one(s_img, r_img, m_src, m_ref, params=None):
                async with sem:
                    # 将计算密集型任务放入线程池执行，避免阻塞事件循环
                    return await asyncio.to_thread(
                        cls._process_one_image,
                        s_img,
                        r_img,
                        m_src,
                        m_ref,
                        method,
                        params
                    )
            
            tasks.append(run_one(src_img, ref_img, margin_src, margin_ref, locked_params))
            
        # 并发执行所有任务
        results = await asyncio.gather(*tasks)
        
        result = torch.stack(results)
        result = torch.clamp(result, 0.0, 1.0)
        
        return io.NodeOutput(result)


    @classmethod
    def _calculate_margin(cls, amount: float, height: int, width: int) -> int:
        """
        计算实际边缘像素宽度。
        amount <= 0: 0 (使用全图)
        0 < amount < 1.0: 短边百分比
        amount >= 1.0: 像素值
        """
        if amount <= 0:
            return 0
        
        # 百分比模式
        if amount < 1.0:
            base = min(height, width)
            return int(base * amount)
            
        # 像素模式
        return int(amount)

    @classmethod
    def _extract_area(cls, image_tensor: torch.Tensor, margin: int):
        """
        提取相关区域（全图或边缘）作为平铺的张量列表。
        返回: [N, C] 像素张量
        """
        h, w, c = image_tensor.shape
        
        if margin <= 0 or margin * 2 >= h or margin * 2 >= w:
            return image_tensor.reshape(-1, c)

        # 提取边缘 (上, 下, 左中, 右中)
        top = image_tensor[:margin, :, :]
        bottom = image_tensor[h-margin:, :, :]
        left = image_tensor[margin:h-margin, :margin, :]
        right = image_tensor[margin:h-margin, w-margin:, :]
        
        return torch.cat([
            top.reshape(-1, c),
            bottom.reshape(-1, c),
            left.reshape(-1, c),
            right.reshape(-1, c)
        ], dim=0)

    @classmethod
    def _get_stats(cls, image_tensor: torch.Tensor, margin: int):
        pixels = cls._extract_area(image_tensor, margin)
        mean = torch.mean(pixels, dim=0)
        std = torch.std(pixels, dim=0)
        return mean, std

    @classmethod
    def _calculate_histogram_luts(cls, src_img, ref_img, margin_src, margin_ref):
        """计算直方图映射表 (LUTs)"""
        # 1. 提取像素用于直方图计算
        src_pixels = cls._extract_area(src_img, margin_src) # [N_src, 3]
        ref_pixels = cls._extract_area(ref_img, margin_ref) # [N_ref, 3]
        
        # 2. 转换为 0-255 uint8 用于直方图
        src_p_255 = (src_pixels * 255).clamp(0, 255).to(torch.uint8)
        ref_p_255 = (ref_pixels * 255).clamp(0, 255).to(torch.uint8)
        
        C = src_img.shape[-1]
        luts = []
        for c in range(C):
            s_chan = src_p_255[:, c].float()
            r_chan = ref_p_255[:, c].float()
            
            s_hist = torch.histc(s_chan, bins=256, min=0, max=255)
            r_hist = torch.histc(r_chan, bins=256, min=0, max=255)
            
            s_cdf = torch.cumsum(s_hist, dim=0)
            s_cdf = s_cdf / (s_cdf[-1] + 1e-6)
            
            r_cdf = torch.cumsum(r_hist, dim=0)
            r_cdf = r_cdf / (r_cdf[-1] + 1e-6)
            
            # 映射
            matches = torch.searchsorted(r_cdf, s_cdf).clamp(0, 255)
            luts.append(matches.float() / 255.0)
            
        return luts

    @classmethod
    def _apply_histogram_luts(cls, src_img, luts):
        """应用直方图映射表"""
        C = src_img.shape[-1]
        out = torch.empty_like(src_img)
        src_full_255 = (src_img * 255).clamp(0, 255).long()
        
        for c in range(C):
            channel_indices = src_full_255[..., c]
            channel_lut = luts[c]
            out[..., c] = channel_lut[channel_indices]
            
        return out

    @classmethod
    def _calculate_standard_stats(cls, src_img, ref_img, margin_src, margin_ref):
        """计算标准模式的统计信息 (mu_src, std_src, mu_ref, std_ref)"""
        mu_src, std_src = cls._get_stats(src_img, margin_src)
        mu_ref, std_ref = cls._get_stats(ref_img, margin_ref)
        
        # 避免除以零
        std_src = torch.where(std_src < 1e-6, torch.ones_like(std_src), std_src)
        
        return (mu_src, std_src, mu_ref, std_ref)

    @classmethod
    def _apply_standard_stats(cls, src_img, stats):
        """应用标准模式的统计信息"""
        mu_src, std_src, mu_ref, std_ref = stats
        
        # 颜色迁移: (x - mu_src) * (std_ref / std_src) + mu_ref
        res = (src_img - mu_src) * (std_ref / std_src) + mu_ref
        return res

    @classmethod
    def _process_one_image(cls, src_img, ref_img, margin_src, margin_ref, method, params=None):
        """处理单张图像的辅助函数"""
        
        if params is not None:
            # 使用预计算的参数 (锁定模式)
            if method == "histogram":
                return cls._apply_histogram_luts(src_img, params)
            else:
                return cls._apply_standard_stats(src_img, params)
        
        # 实时计算模式
        if method == "histogram":
            luts = cls._calculate_histogram_luts(src_img, ref_img, margin_src, margin_ref)
            return cls._apply_histogram_luts(src_img, luts)
        else:
            stats = cls._calculate_standard_stats(src_img, ref_img, margin_src, margin_ref)
            return cls._apply_standard_stats(src_img, stats)

