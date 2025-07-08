class ImageGetSize:
    """
    获取图像的宽度和高度信息
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            }
        }
    
    RETURN_TYPES = ("INT", "INT")
    RETURN_NAMES = ("width", "height")
    FUNCTION = "get_image_size"
    CATEGORY = "1hewNodes/util"
    
    def get_image_size(self, image):
        """
        获取图像的宽度和高度
        
        Args:
            image: 输入的图像张量，格式为 (batch, height, width, channels)
            
        Returns:
            tuple: (width, height) 图像的宽度和高度
        """
        # 确保输入是正确的维度
        if image.dim() == 3:
            image = image.unsqueeze(0)
        
        # 获取图像尺寸 (batch, height, width, channels)
        batch_size, height, width, channels = image.shape
        
        # 返回宽度和高度（整数类型）
        return (int(width), int(height))


class RangeMapping:
    """范围映射
    滑动条的数值会根据min和max_value的修改实时变化
    rounding参数控制小数位数精度
    """
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "value": ("FLOAT", {
                        "default": 1.0, 
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.001,
                        "display": "slider"
                    }),
                    "min": ("FLOAT", {
                        "default": 0.0, 
                        "min": -0xffffffffffffffff, 
                        "max": 0xffffffffffffffff,
                        "step": 0.001, 
                        "display": "number"  
                    }),
                    "max": ("FLOAT", {
                        "default": 1.0, 
                        "min": -0xffffffffffffffff,
                        "max": 0xffffffffffffffff,
                        "step": 0.001, 
                        "display": "number"  
                    }),
                    "rounding": ("INT", {
                        "default": 3, 
                        "min": 0,
                        "max": 10,
                        "step": 1, 
                        "display": "number"  
                    }),
                },
            }
    
    RETURN_TYPES = ("FLOAT","INT") 
    RETURN_NAMES = ('float','int')
    FUNCTION = "range_mapping"

    CATEGORY = "1hewNodes/util"

    def range_mapping(self, value, min, max, rounding):
        # 将0-1范围的滑动条值映射到 min 和 max 之间
        actual_value = min + value * (max - min)
        
        # 根据rounding参数设置小数位数精度
        if rounding > 0:
            actual_value = round(actual_value, rounding)
        else:
            actual_value = int(actual_value)
            
        return (actual_value, int(actual_value))


class PathBuild:
    """
    路径构建 - 提供一个层级结构的路径选择下拉框，并允许添加自定义路径扩展
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        # 定义路径选项
        paths = []
        
        # 一级字段
        level1 = ["Wan_kijai", "Wan_org", "Flux"]
        # 二级字段映射
        level2_mapping = {
            "Wan_kijai": ["FusionX_VACE",],
            "Wan_org": ["VACE", "FLF2V", "Fun"],
            "Flux": ["ACE_Redux", "ACE", "TTP"]
        }
        # 三级字段映射
        level3_mapping = {
            "Wan_kijai": ["FLFControl", "Control", "FLF", "Inpaint", "Outpaint"],
            "Wan_org": ["FLFControl", "Control", "FLF", "Inpaint"],
            "Flux": []  # Flux 使用 additional_path 作为三级字段，所以这里为空
        }
        
        # 生成所有可能的路径组合（遍历模式）
        for l1 in level1:
            for l2 in level2_mapping[l1]:
                if l1 == "Flux":
                    # 对于 Flux，直接添加二级路径
                    paths.append(f"{l1}/{l2}")
                else:
                    # 对于其他一级字段，添加三级路径
                    for l3 in level3_mapping[l1]:
                        paths.append(f"{l1}/{l2}/{l3}")
        
        # 添加固定的自定义路径（固定模式）
        custom_paths = [
            # 可以在这里添加更多固定路径
            # "Custom/Path1",
            "Wan_kijai/FusionX_Phantom",
            # ... 方便后面补充更多固定路径
        ]
        
        # 合并遍历生成的路径和固定路径
        paths.extend(custom_paths)
        
        return {
            "required": {
                "preset_path": (paths, {"default": paths[0]}),
                "additional_path": ("STRING", {"default": "", "multiline": False})
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("full_path",)
    FUNCTION = "build_path"
    CATEGORY = "1hewNodes/util"

    def build_path(self, preset_path, additional_path):
        # 如果提供了额外路径，则添加到预设路径中
        if additional_path and additional_path.strip():
            full_path = f"{preset_path}/{additional_path.strip()}"
        else:
            full_path = preset_path
            
        return (full_path,)
  

# 在NODE_CLASS_MAPPINGS中更新节点映射
NODE_CLASS_MAPPINGS = {
    "ImageGetSize": ImageGetSize,
    "RangeMapping": RangeMapping,
    "PathBuild": PathBuild,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageGetSize": "Image Get Size",
    "RangeMapping": "Range Mapping",
    "PathBuild": "Path Build",
}