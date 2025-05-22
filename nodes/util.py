import torch
import json
import numpy as np
import os
import re

class CoordinateExtract:
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

class PromptExtract:
    """
    从文本中提取指定语言的内容
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {}),
                "language": (["en", "zh"], {"default": "en"}),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    FUNCTION = "prompt_extract"
    CATEGORY = "1hewNodes/util"
    
    def prompt_extract(self, text, language):
        # 尝试解析JSON文本
        try:
            # 如果输入是标准JSON格式
            data = json.loads(text)
        except json.JSONDecodeError:
            # 如果不是标准JSON，尝试解析为键值对格式
            try:
                # 清理文本，移除多余的空格和换行符
                cleaned_text = re.sub(r'\s+', ' ', text).strip()
                # 将文本转换为标准JSON格式
                if cleaned_text.startswith('{') and cleaned_text.endswith('}'):
                    cleaned_text = cleaned_text[1:-1]
                
                # 构建JSON字符串
                json_str = '{' + cleaned_text + '}'
                data = json.loads(json_str)
            except:
                # 如果仍然无法解析，尝试使用正则表达式提取
                pattern = r'"([^"]+)"\s*:\s*"([^"]+)"'
                matches = re.findall(pattern, text)
                data = {key: value for key, value in matches}
        
        # 根据选择的语言提取文本
        if language == "en":
            # 尝试查找英文相关的键
            english_keys = ["English", "english", "英文", "英语", "ENGLISH", "ENG", "eng"]
            for key in english_keys:
                if key in data:
                    return (data[key],)
        else:  # language == "zh"
            # 尝试查找中文相关的键
            chinese_keys = ["Chinese", "chinese", "中文", "China", "china", "CHINESE", "CHN", "chn", "ZH", "zh"]
            for key in chinese_keys:
                if key in data:
                    return (data[key],)
        
        # 如果没有找到匹配的键，返回错误信息
        return (f"未找到{language}语言的文本",)


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
        level1 = ["kijai_wan", "org_wan", "Flux"]
        # 二级字段映射
        level2_mapping = {
            "kijai_wan": ["VACE", "FLF2V", "Fun"],
            "org_wan": ["VACE", "FLF2V", "Fun"],
            "Flux": ["ACE_Redux", "ACE", "TTP"]
        }
        # 三级字段映射
        level3_mapping = {
            "kijai_wan": ["FLFControl", "Control", "FLF", "Inpaint"],
            "org_wan": ["FLFControl", "Control", "FLF", "Inpaint"],
            "Flux": []  # Flux 使用 filename 作为三级字段，所以这里为空
        }
        
        # 生成所有可能的路径组合
        for l1 in level1:
            for l2 in level2_mapping[l1]:
                if l1 == "Flux":
                    # 对于 Flux，直接添加二级路径
                    paths.append(f"{l1}/{l2}")
                else:
                    # 对于其他一级字段，添加三级路径
                    for l3 in level3_mapping[l1]:
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
    "CoordinateExtract": CoordinateExtract,
    "PromptExtract": PromptExtract,
    "SliderValueRangeMapping": SliderValueRangeMapping,
    "PathSelect": PathSelect,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CoordinateExtract": "Coordinat Extract",
    "PromptExtract": "Prompt Extract",
    "SliderValueRangeMapping": "Slider Value Range Mapping",
    "PathSelect": "Path Select",
}
