
from comfy_api.latest import io
import asyncio
import cv2
import numpy as np
import re
import torch


class VideoCutGroup(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="1hew_VideoCutGroup",
            display_name="Video Cut Group",
            category="1hewNodes/batch",
            inputs=[
                io.Image.Input("image"),
                io.Float.Input("threshold_base", default=0.8, min=0.0, max=1.0, step=0.01),
                io.Float.Input("threshold_range", default=0.05, min=0.01, max=0.2, step=0.01),
                io.Int.Input("threshold_count", default=2, min=1, max=10, step=1),
                io.String.Input("kernel", default="3, 7, 11"),
                io.Int.Input("min_frame_count", default=10, min=1, max=1000, step=1),
                io.Int.Input("max_frame_count", default=0, min=0, max=10000, step=1),
                io.Boolean.Input("fast", default=False),
                io.String.Input("add_frame", default="", optional=True),
                io.String.Input("delete_frame", default="", optional=True),
            ],
            outputs=[
                io.Image.Output(display_name="image"),
                io.Int.Output(display_name="group_total"),
                io.MultiType.Output(display_name="start_index", is_output_list=True),
                io.MultiType.Output(display_name="batch_count", is_output_list=True),
            ],
        )


    @classmethod
    async def execute(
        cls,
        image: torch.Tensor,
        threshold_base: float,
        threshold_range: float,
        threshold_count: int,
        kernel: str,
        min_frame_count: int,
        max_frame_count: int,
        fast: bool,
        add_frame: str = "",
        delete_frame: str = "",
    ) -> io.NodeOutput:
        B = int(image.shape[0])
        if B <= 1:
            return io.NodeOutput(image, 1, [0], [B])

        kernel_configs = cls.parse_custom_kernels(kernel)

        if fast:
            cut_points = cls.fast_mode_detection(
                image, threshold_base, min_frame_count, max_frame_count
            )
        else:
            images_np = image.detach().cpu().numpy()
            all_results = await cls.batch_detection_all_features_async(
                images_np,
                threshold_base,
                min_frame_count,
                max_frame_count,
                threshold_range,
                threshold_count,
                kernel_configs,
            )
            cut_points = cls.unified_voting_fusion(
                all_results, B, min_frame_count, max_frame_count
            )

        add_list = cls.parse_user_frames(add_frame)
        del_list = cls.parse_user_frames(delete_frame)
        s = set(cut_points)
        for a in add_list:
            if 0 <= a <= B:
                s.add(a)
        for d in del_list:
            if d in s and d != 0:
                s.remove(d)
        if 0 not in s:
            s.add(0)
        cut_points = sorted(list(s))
        cut_points = cls._apply_final_grouping_rules(
            cut_points, min_frame_count, max_frame_count, B
        )

        starts = cut_points
        counts = []
        for i in range(len(starts)):
            start = starts[i]
            if i == len(starts) - 1:
                cnt = max(0, B - start)
            else:
                cnt = max(0, starts[i + 1] - start)
            counts.append(cnt)

        selected = image[starts]
        return io.NodeOutput(selected, len(starts), starts, counts)

    @staticmethod
    def parse_user_frames(frame_string):
        """
        解析用户输入的帧索引字符串，支持逗号分隔，智能处理中英文逗号和空格
        """
        if not frame_string or not frame_string.strip():
            return []
        
        # 替换中文逗号为英文逗号，移除多余空格
        normalized = re.sub(r'[，,]\s*', ',', frame_string.strip())
        
        # 分割并转换为整数
        frame_indices = []
        for item in normalized.split(','):
            item = item.strip()
            if item and item.isdigit():
                frame_indices.append(int(item))
        
        return sorted(list(set(frame_indices)))  # 去重并排序

    @staticmethod
    def parse_custom_kernels(kernel_string):
        """
        解析用户输入的kernel配置字符串，支持逗号分隔
        格式: "3,7,11" 或 "3,5,7,9,11,13,15"
        返回: [(kernel_size, sigma), ...] 格式的列表
        """
        if not kernel_string or not kernel_string.strip():
            # 如果为空，返回默认配置
            return [(3, 0.6), (7, 1.0), (11, 1.5)]
        
        # 替换中文逗号为英文逗号，移除多余空格
        normalized = re.sub(r'[，,]\s*', ',', kernel_string.strip())
        
        # 分割并转换为整数
        kernel_sizes = []
        for item in normalized.split(','):
            item = item.strip()
            if item and item.isdigit():
                size = int(item)
                # 验证kernel大小必须是奇数且大于等于3
                if size >= 3 and size % 2 == 1:
                    kernel_sizes.append(size)
        
        # 去重并排序
        kernel_sizes = sorted(list(set(kernel_sizes)))
        
        # 如果没有有效的kernel，返回默认配置
        if not kernel_sizes:
            return [(3, 0.6), (7, 1.0), (11, 1.5)]
        
        # 为每个kernel大小生成对应的sigma值
        # sigma = kernel_size * 0.2 (经验公式)
        kernel_configs = []
        for size in kernel_sizes:
            sigma = size * 0.2
            kernel_configs.append((size, sigma))
        
        return kernel_configs

    @staticmethod
    def simple_ssim(img1, img2, C1=0.01**2, C2=0.03**2):
        """
        简易SSIM计算方法，用于快速模式
        """
        # 如果是彩色图像，先转为灰度
        if img1.shape[-1] == 3:
            img1 = 0.299 * img1[..., 0] + 0.587 * img1[..., 1] + 0.114 * img1[..., 2]
            img2 = 0.299 * img2[..., 0] + 0.587 * img2[..., 1] + 0.114 * img2[..., 2]
        
        # 计算均值
        mu1 = img1.mean()
        mu2 = img2.mean()
        
        # 计算方差
        sigma1 = ((img1 - mu1) ** 2).mean()
        sigma2 = ((img2 - mu2) ** 2).mean()
        
        # 计算协方差
        sigma12 = ((img1 - mu1) * (img2 - mu2)).mean()
        
        # 计算SSIM
        ssim = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / ((mu1 ** 2 + mu2 ** 2 + C1) * (sigma1 + sigma2 + C2))
        
        # 保证结果在0~1之间
        return max(0.0, min(1.0, ssim.item() if hasattr(ssim, 'item') else float(ssim)))

    @staticmethod
    def preprocess_images_batch(images_np):
        """
        批量预处理所有图像，避免重复转换
        """
        # 确保数据格式正确
        if images_np.dtype != np.float32:
            images_np = images_np.astype(np.float32)
        
        # 如果图像值在 [0, 1] 范围内，转换为 [0, 255] 以便SSIM计算
        if images_np.max() <= 1.0:
            images_np = images_np * 255.0
        
        # 转换为灰度图像以提高检测效果
        if len(images_np.shape) == 4 and images_np.shape[3] == 3:
            # RGB转灰度：0.299*R + 0.587*G + 0.114*B
            images_np = np.dot(images_np[..., :3], [0.299, 0.587, 0.114])
        elif len(images_np.shape) == 4 and images_np.shape[3] == 1:
            # 已经是单通道，去掉最后一个维度
            images_np = images_np.squeeze(-1)
        
        return images_np

    @staticmethod
    def batch_calculate_ssim_matrix(processed_images, kernel_configs):
        """
        批量计算所有相邻帧的SSIM值矩阵，使用固定的核配置和模糊模式
        """
        B = processed_images.shape[0]
        if B <= 1:
            return {}
        
        ssim_matrix = {}
        
        # 为每个核配置计算模糊SSIM
        for kernel_idx, kernel_config in enumerate(kernel_configs):
            kernel_size, sigma = kernel_config
            ssim_values = np.zeros(B - 1, dtype=np.float32)
            
            # 计算所有相邻帧的模糊SSIM
            for i in range(B - 1):
                ssim_val = VideoCutGroup._blur_pixel_ssim(
                    processed_images[i], processed_images[i + 1], kernel_size, sigma
                )
                ssim_values[i] = ssim_val
            
            ssim_matrix[kernel_idx] = ssim_values
        
        return ssim_matrix

    @staticmethod
    async def batch_calculate_ssim_matrix_async(processed_images, kernel_configs):
        B = processed_images.shape[0]
        if B <= 1:
            return {}
        async def task(idx, config):
            kernel_size, sigma = config
            ssim_values = np.zeros(B - 1, dtype=np.float32)
            def calc():
                for i in range(B - 1):
                    s = VideoCutGroup._blur_pixel_ssim(
                        processed_images[i], processed_images[i + 1], kernel_size, sigma
                    )
                    ssim_values[i] = s
                return ssim_values
            values = await asyncio.to_thread(calc)
            return idx, values
        tasks = [task(i, kernel_configs[i]) for i in range(len(kernel_configs))]
        results = await asyncio.gather(*tasks)
        ssim_matrix = {idx: values for idx, values in results}
        return ssim_matrix

    @staticmethod
    def _blur_pixel_ssim(img1, img2, kernel_size, sigma):
        """模糊像素SSIM计算，优化内存使用"""
        # 应用高斯模糊
        ksize = (kernel_size, kernel_size)
        img1_blur = cv2.GaussianBlur(img1, ksize, sigma)
        img2_blur = cv2.GaussianBlur(img2, ksize, sigma)
        
        # 计算均值
        mu1 = cv2.boxFilter(img1_blur, -1, (kernel_size, kernel_size))
        mu2 = cv2.boxFilter(img2_blur, -1, (kernel_size, kernel_size))
        
        mu1_sq = mu1 * mu1
        mu2_sq = mu2 * mu2
        mu1_mu2 = mu1 * mu2
        
        # 计算方差和协方差
        sigma1_sq = cv2.boxFilter(img1_blur * img1_blur, -1, (kernel_size, kernel_size)) - mu1_sq
        sigma2_sq = cv2.boxFilter(img2_blur * img2_blur, -1, (kernel_size, kernel_size)) - mu2_sq
        sigma12 = cv2.boxFilter(img1_blur * img2_blur, -1, (kernel_size, kernel_size)) - mu1_mu2
        
        # SSIM常数
        C1 = (0.01) ** 2
        C2 = (0.03) ** 2
        
        # 计算SSIM
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        
        return float(np.mean(ssim_map))

    @staticmethod
    def generate_dynamic_thresholds(threshold_base, threshold_range=0.05, threshold_count=2):
        """
        生成动态数量的阈值
        threshold_count=1: 只使用threshold_base
        threshold_count=2: 范围两端 [base-range, base+range]
        threshold_count=3: 两端+中间 [base-range, base, base+range]
        threshold_count=4+: 在范围内均匀分布
        """
        if threshold_count == 1:
            # 只使用基础阈值
            thresholds = [threshold_base]
        elif threshold_count == 2:
            # 范围两端
            thresholds = [
                threshold_base - threshold_range,  # 下限
                threshold_base + threshold_range   # 上限
            ]
        else:
            # 3个或更多：在范围内均匀分布
            min_threshold = threshold_base - threshold_range
            max_threshold = threshold_base + threshold_range
            
            if threshold_count == 3:
                # 特殊处理：两端+中间
                thresholds = [min_threshold, threshold_base, max_threshold]
            else:
                # 4个或更多：均匀分布
                step = (max_threshold - min_threshold) / (threshold_count - 1)
                thresholds = [min_threshold + i * step for i in range(threshold_count)]
        
        # 确保阈值在合理范围内
        thresholds = [max(0.0, min(1.0, t)) for t in thresholds]
        
        return sorted(thresholds)

    @staticmethod
    def optimized_single_threshold_detection(ssim_matrix, user_threshold, kernel_idx, 
                                           min_frame_count, max_frame_count, total_frames):
        """
        基于预计算SSIM矩阵的纯粹阈值检测方法
        修正阈值逻辑：用户阈值越大，检测越严格，画面越少
        
        注意：此方法只进行纯粹的阈值检测，不应用分组规则
        分组规则应该在所有阈值检测完成后，在最终合并阶段统一应用
        """
        B = total_frames
        if B <= 1:
            return [0]
        
        # 获取对应的SSIM值数组
        if kernel_idx not in ssim_matrix:
            return [0]
        
        ssim_values = ssim_matrix[kernel_idx]
        
        # 纯粹的阈值检测：找出所有满足阈值条件的硬切点
        threshold_cuts = [0]  # 第一组总是从0开始
        for i in range(len(ssim_values)):
            ssim_val = ssim_values[i]
            # 直接比较：(1-ssim)大于用户阈值时，认为是硬切
            if (1.0 - ssim_val) > user_threshold:
                cut_point = i + 1  # 硬切后的帧作为新组的起始
                if cut_point < B and cut_point not in threshold_cuts:
                    threshold_cuts.append(cut_point)
        
        return threshold_cuts


    @staticmethod
    def batch_detection_all_features(images_np, threshold_base, min_frame_count, max_frame_count, threshold_range=0.05, threshold_count=2, kernel_configs=None):
        """
        批量检测所有特征组合，使用固定的核配置和参数
        """
        total_frames = len(images_np)
        
        if total_frames < 2:
            return [[0]]
        
        # 预处理图像
        processed_images = VideoCutGroup.preprocess_images_batch(images_np)
        
        # 批量计算SSIM矩阵
        ssim_matrix = VideoCutGroup.batch_calculate_ssim_matrix(processed_images, kernel_configs)
        
        # 生成动态数量的阈值
        user_thresholds = VideoCutGroup.generate_dynamic_thresholds(threshold_base, threshold_range, threshold_count)
        
        # 打印检测任务概览
        print()
        print("=== 🚀 VideoCutGroup 多核模糊模式检测 启动 ===")
        print(f"threshold: {[f'{t:.3f}' for t in user_thresholds]}")
        kernel_list = [str(k[0]) for k in kernel_configs]
        print(f"kernel: [{','.join(kernel_list)}]")
        print()
        
        total_groups = len(kernel_configs) * len(user_thresholds)
        print(f"📈 {total_groups} 组检测任务详情")
        
        # 对每个核和每个阈值进行检测
        all_detection_results = []
        group_num = 1
        
        for kernel_idx in range(len(kernel_configs)):
            kernel_size, sigma = kernel_configs[kernel_idx]
            
            for user_threshold in user_thresholds:
                # 使用优化的检测方法
                result = VideoCutGroup.optimized_single_threshold_detection(
                    ssim_matrix, user_threshold, kernel_idx, min_frame_count, max_frame_count, total_frames
                )
                all_detection_results.append(result)
                
                # 获取阈值详细信息用于日志 - 基于实际检测结果
                threshold_details = []
                if kernel_idx in ssim_matrix and len(result) > 1:
                    ssim_values = ssim_matrix[kernel_idx]
                    # 遍历实际检测到的切点（排除起始点0）
                    for cut_point in result[1:]:  # 跳过起始点0
                        if cut_point > 0 and cut_point <= len(ssim_values):
                            # 切点对应的是前一帧与当前帧的比较
                            ssim_val = ssim_values[cut_point - 1]
                            threshold_val = 1.0 - ssim_val
                            threshold_details.append(f"{cut_point}:{threshold_val:.3f}")
                
                # 格式化日志输出（排除起始点0）
                actual_cut_points = len(result) - 1 if result and result[0] == 0 else len(result)
                print(f"🔍 第{group_num}组： threshold={user_threshold:.3f}，kernel = {kernel_size}")
                print(f"- 检测切点：{actual_cut_points} 个 [index：threshold]")
                if threshold_details:
                    print(f"- [{', '.join(threshold_details)}]")
                print()
                
                group_num += 1
        
        # 存储检测结果用于后续汇总
        # 摘要由打印输出表示
        
        return all_detection_results

    @staticmethod
    async def batch_detection_all_features_async(images_np, threshold_base, min_frame_count, max_frame_count, threshold_range=0.05, threshold_count=2, kernel_configs=None):
        total_frames = len(images_np)
        if total_frames < 2:
            return [[0]]
        processed_images = VideoCutGroup.preprocess_images_batch(images_np)
        ssim_matrix = await VideoCutGroup.batch_calculate_ssim_matrix_async(processed_images, kernel_configs)
        user_thresholds = VideoCutGroup.generate_dynamic_thresholds(threshold_base, threshold_range, threshold_count)
        detect_tasks = []
        for kernel_idx in range(len(kernel_configs)):
            for user_threshold in user_thresholds:
                def detect():
                    return VideoCutGroup.optimized_single_threshold_detection(
                        ssim_matrix, user_threshold, kernel_idx, min_frame_count, max_frame_count, total_frames
                    )
                detect_tasks.append(asyncio.to_thread(detect))
        results = await asyncio.gather(*detect_tasks)
        return results

    @staticmethod
    def unified_voting_fusion(all_detection_results, total_frames, min_frame_count=10, max_frame_count=0):
        """
        统一融合方法，整合所有检测结果并应用分组规则
        
        重要：在此阶段统一应用min_frame_count和max_frame_count规则，
        确保所有阈值组合都使用相同的分组策略
        """
        if not all_detection_results:
            return [0]
        
        # 打印最终检测结果汇总
        print()
        print("✅ 最终检测结果")
        
        # 收集所有切点（去重）
        all_cut_points = set([0])  # 起始帧
        for result in all_detection_results:
            for frame in result[1:]:  # 跳过起始帧0
                all_cut_points.add(frame)
        
        # 排序
        raw_cut_points = sorted(list(all_cut_points))
        
        # 显示合并前的结果
        print(f"合并检测结果：{len(raw_cut_points)} 个")
        print(f"{raw_cut_points}")
        print()
        
        # 应用分组规则
        final_cut_points = VideoCutGroup._apply_final_grouping_rules(raw_cut_points, min_frame_count, max_frame_count, total_frames)
        
        # 显示最终结果
        print(f"min_frame_count={min_frame_count}, max_frame_count={max_frame_count} ，分组规则处理后：{len(final_cut_points)} 个")
        print(f"{final_cut_points}")
        print()
        
        return final_cut_points
    
    @staticmethod
    def _apply_final_grouping_rules(cut_points, min_frame_count, max_frame_count, total_frames):
        """
        在最终阶段应用分组规则
        """
        if not cut_points or len(cut_points) <= 1:
            return [0]
        
        # 应用min_frame_count规则：合并过近的切点
        filtered_points = [cut_points[0]]  # 保留起始点0
        
        for point in cut_points[1:]:
            if point - filtered_points[-1] >= min_frame_count:
                filtered_points.append(point)
            # 如果距离太近，跳过这个切点
        
        # 应用max_frame_count规则：拆分过长的段
        if max_frame_count > 0:
            final_points = [0]
            
            for i in range(1, len(filtered_points)):
                start = filtered_points[i-1]
                end = filtered_points[i]
                segment_length = end - start
                
                # 如果段长度超过限制，需要拆分
                if segment_length > max_frame_count:
                    # 在这个段内按max_frame_count间隔插入切点
                    current = start
                    while current + max_frame_count < end:
                        current += max_frame_count
                        final_points.append(current)
                
                # 添加原始切点
                final_points.append(end)
            
            # 处理最后一段（到视频结尾）
            if len(filtered_points) > 0:
                last_point = filtered_points[-1]
                if last_point < total_frames:
                    remaining_length = total_frames - last_point
                    if remaining_length > max_frame_count:
                        current = last_point
                        while current + max_frame_count < total_frames:
                            current += max_frame_count
                            final_points.append(current)
            
            return sorted(list(set(final_points)))
        else:
            return filtered_points

    @staticmethod
    def sequential_detection(images, threshold_base, min_frame_count, max_frame_count, threshold_range=0.05, threshold_count=2, kernel_configs=None):
        """
        序列检测方法，根据参数配置进行视频硬切检测
        """
        # 转换图像格式
        if hasattr(images, 'cpu'):
            images_np = images.cpu().numpy()
        else:
            images_np = images
        
        # 批量检测所有特征组合
        all_detection_results = VideoCutGroup.batch_detection_all_features(
            images_np, threshold_base, min_frame_count, max_frame_count, threshold_range, threshold_count, kernel_configs
        )
        
        # 统一投票融合
        final_split_points = VideoCutGroup.unified_voting_fusion(all_detection_results, len(images_np), min_frame_count, max_frame_count)
        
        return final_split_points

    @staticmethod
    def fast_mode_detection(images, threshold_base, min_frame_count, max_frame_count):
        """
        快速模式检测，使用简化的SSIM计算方法
        注意：这里的threshold_base已经经过1-处理，需要转换回原始阈值逻辑
        """
        if hasattr(images, 'cpu'):
            images_np = images.cpu().numpy()
        else:
            images_np = images
        B = int(images_np.shape[0])
        if B < 2:
            return [0]

        threshold = float(max(0.0, min(1.0, threshold_base)))
        processed = VideoCutGroup.preprocess_images_batch(images_np)

        ssim_list = []
        for i in range(B - 1):
            ssim_val = VideoCutGroup.simple_ssim(processed[i], processed[i + 1])
            ssim_list.append(ssim_val)

        cut_points = [0]
        for i, ssim_val in enumerate(ssim_list):
            if (1.0 - float(ssim_val)) > threshold:
                cp = i + 1
                if cp < B:
                    cut_points.append(cp)

        cut_points = sorted(list(set(cut_points)))
        final_points = VideoCutGroup._apply_final_grouping_rules(
            cut_points, min_frame_count, max_frame_count, B
        )
        return final_points


    def apply_user_modifications(self, cut_points, add_frame, delete_frame, total_frames):
        """
        应用用户自定义的添加和删除帧修改
        """
        # 解析用户输入
        add_frames = self.parse_user_frames(add_frame)
        delete_frames = self.parse_user_frames(delete_frame)
        
        # 应用修改
        modified_cut_points = list(cut_points)
        
        # 记录实际添加和删除的帧
        actually_added = []
        actually_deleted = []
        
        # 添加用户指定的帧
        for frame in add_frames:
            if 0 <= frame < total_frames and frame not in modified_cut_points:
                modified_cut_points.append(frame)
                actually_added.append(frame)
        
        # 删除用户指定的帧（但保留起始帧0）
        for frame in delete_frames:
            if frame in modified_cut_points and frame != 0:
                modified_cut_points.remove(frame)
                actually_deleted.append(frame)
        
        # 打印用户设定的所有添加和删除信息（不是过滤后的）
        if add_frames:
            print(f"➕ 用户添加帧: {add_frames}")
        if delete_frames:
            print(f"➖ 用户删除帧: {delete_frames}")
        
        # 排序并返回
        final_cut_points = sorted(list(set(modified_cut_points)))
        
        if add_frames or delete_frames:
            print(f"🔧 后期增减帧处理后：{len(final_cut_points)} 个")
            print(f"{final_cut_points}")
        
        return final_cut_points