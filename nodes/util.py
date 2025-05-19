import torch
import json
import numpy as np
import os

class CoordinateExtractor:
    """坐标提取器
    [
        {
            "x": 0,
            "y": 512
        },
        {
            "x": 59,
            "y": 510
        }
     ]
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "coordinates_json": ("STRING", {"multiline": True}),
            }
        }

    RETURN_TYPES = ("FLOAT", "FLOAT")
    RETURN_NAMES = ("x", "y")
    OUTPUT_IS_LIST = (True, True)
    FUNCTION = "coordinate_extract"
    
    CATEGORY = "1hewNodes/util"

    def coordinate_extract(self, coordinates_json):
        try:
            # 解析 JSON 字符串为 Python 对象
            if isinstance(coordinates_json, str):
                points = json.loads(coordinates_json)
            else:
                points = coordinates_json
            
            # 提取 x 和 y 列表
            x_list = [float(point["x"]) for point in points]
            y_list = [float(point["y"]) for point in points]
            
            return (x_list, y_list)
        except Exception as e:
            print(f"Error extracting coordinates: {e}")
            return ([], [])


class SliderValueRangeMapping:
    """滑动条数值范围映射
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
    FUNCTION = "slider_value_range_mapping"

    CATEGORY = "1hewNodes/util"

    def slider_value_range_mapping(self, value, min, max, rounding):
        # 将0-1范围的滑动条值映射到 min 和 max 之间
        actual_value = min + value * (max - min)
        
        # 根据rounding参数设置小数位数精度
        if rounding > 0:
            actual_value = round(actual_value, rounding)
        else:
            actual_value = int(actual_value)
            
        return (actual_value, int(actual_value))


class PathSelect:
    """
    路径选择 - 提供一个层级结构的路径选择下拉框，并允许添加第四级自定义字段
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        # 定义路径选项
        paths = []
        
        # 一级字段
        level1 = ["kijai_wan", "org_wan"]
        # 二级字段
        level2 = ["VACE", "FLF2V", "Fun"]
        # 三级字段
        level3 = ["FLFControl", "Control", "FLF", "Inpaint"]
        
        # 生成所有可能的路径组合
        for l1 in level1:
            for l2 in level2:
                for l3 in level3:
                    paths.append(f"{l1}/{l2}/{l3}")
        
        return {
            "required": {
                "path": (paths, {"default": paths[0], "label": "选择路径"}),
                "filename": ("STRING", {"default": "", "multiline": False, "label": "第四级字段"})
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("filename",)
    FUNCTION = "select_path"
    CATEGORY = "1hewNodes/util"

    def select_path(self, path, filename):
        # 如果提供了第四级字段，则添加到路径中
        if filename and filename.strip():
            full_path = f"{path}/{filename.strip()}"
        else:
            full_path = path
            
        return (full_path,)


# 在NODE_CLASS_MAPPINGS中添加新节点
NODE_CLASS_MAPPINGS = {
    "CoordinateExtractor": CoordinateExtractor,
    "SliderValueRangeMapping": SliderValueRangeMapping,
    "PathSelect": PathSelect,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CoordinateExtractor": "Coordinat Extractor",
    "SliderValueRangeMapping": "Slider Value Range Mapping",
    "PathSelect": "Path Select",
}
