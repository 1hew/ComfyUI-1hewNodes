import json
import torch
import numpy as np

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

# 注册节点
NODE_CLASS_MAPPINGS = {
    "CoordinateExtractor": CoordinateExtractor
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CoordinateExtractor": "Coordinat Extractor"
}