import json
import re

class TextCustomList:
    """
    自定义列表节点 - 生成自定义的文本/数值列表
    支持多种分隔符和引号包裹，可以直接与 xyAny 节点配合使用进行笛卡尔积组合
    自动检测并输出最适合的数据类型：int > float > string
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "custom_text": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "placeholder": "Separate with comma(,), semicolon(;), or newline."
                })
            }
        }
    
    RETURN_TYPES = ("*", "INT")
    RETURN_NAMES = ("list", "count")
    OUTPUT_IS_LIST = (True, False)
    CATEGORY = "1hewNodes/text"
    FUNCTION = "text_custom_list"

    
    def parse_text_list(self, text):
        """
        解析文本列表，支持多种分隔符和引号包裹
        支持的分隔符：逗号(,)、分号(;)、换行符(\n)
        支持双引号和单引号包裹的整体文本
        忽略只包含一个或多个连字符(-)的行
        """
        if not text.strip():
            return ["default"]
        
        # 简化的解析方法，避免复杂的正则表达式
        # 首先按换行符分割
        lines = text.split('\n')
        
        text_list = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # 忽略只包含一个或多个连字符的行
            if line and all(c == '-' for c in line):
                continue
                
            # 处理逗号和分号分隔的项目
            if ',' in line or ';' in line:
                # 替换分号为逗号，统一处理
                line = line.replace(';', ',')
                items = line.split(',')
                for item in items:
                    item = item.strip()
                    # 移除引号
                    if (item.startswith('"') and item.endswith('"')) or (item.startswith("'") and item.endswith("'")):
                        item = item[1:-1]
                    if item:
                        text_list.append(item)
            else:
                # 单个项目，移除引号
                if (line.startswith('"') and line.endswith('"')) or (line.startswith("'") and line.endswith("'")):
                    line = line[1:-1]
                if line:
                    text_list.append(line)
        
        # 如果解析结果为空，返回默认值
        if not text_list:
            text_list = ["default"]
            
        return text_list
    
    def detect_and_convert_type(self, text_list):
        """
        自动检测并转换为最适合的数据类型
        优先级：int > float > string
        如果包含小数点则保持为浮点数
        """
        if not text_list:
            return []
        
        # 检测是否所有项目都可以转换为数字
        all_int = True
        all_float = True
        
        converted_list = []
        
        for item in text_list:
            try:
                # 如果包含小数点，直接作为浮点数处理
                if '.' in str(item):
                    float_val = float(item)
                    converted_list.append(float_val)
                    all_int = False  # 包含小数点就不是整数类型
                else:
                    # 不包含小数点，尝试转换为整数
                    int_val = int(item)
                    converted_list.append(int_val)
            except (ValueError, TypeError):
                # 无法转换为数字，保持原始字符串
                all_int = False
                all_float = False
                converted_list.append(item)
        
        # 如果不能全部转换为数字，返回原始字符串列表
        if not all_float:
            return text_list
        
        return converted_list
    
    def text_custom_list(self, custom_text):
        """
        生成自动检测类型的自定义列表
        """
        text_list = self.parse_text_list(custom_text)
        value_list = self.detect_and_convert_type(text_list)
        return (value_list, len(value_list))


class TextCustomExtract:
    """
    文本自定义提取器 - 从JSON对象或数组中提取指定键的值
    支持精确匹配模式和增强匹配模式
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "json_data": ("STRING", {
                    "multiline": True,
                    "placeholder": "Enter JSON object or array data"
                }),
                "key": ("STRING", {
                    "default": "zh",
                    "placeholder": "Key name to extract"
                }),
                "precision_match": (["disabled", "enabled"], {"default": "disabled"})
            }
        }
    
    RETURN_TYPES = ("*",)
    RETURN_NAMES = ("value",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "text_custom_extract"
    CATEGORY = "1hewNodes/text"
    
    def parse_json_data(self, json_data):
        """
        解析JSON数据，支持多种格式
        """
        try:
            # 尝试直接解析JSON
            if isinstance(json_data, str):
                return json.loads(json_data)
            else:
                return json_data
        except json.JSONDecodeError:
            # 如果不是标准JSON，尝试解析为键值对格式
            try:
                # 清理文本，移除多余的空格和换行符
                cleaned_text = re.sub(r'\s+', ' ', json_data).strip()
                
                # 如果文本被大括号包围，移除它们
                if cleaned_text.startswith('{') and cleaned_text.endswith('}'):
                    cleaned_text = cleaned_text[1:-1]
                
                # 构建标准JSON字符串
                json_str = '{' + cleaned_text + '}'
                return json.loads(json_str)
            except:
                # 使用正则表达式提取键值对
                pattern = r'"([^"]+)"\s*:\s*"([^"]+)"'
                matches = re.findall(pattern, json_data)
                if matches:
                    return {key: value for key, value in matches}
                
                # 如果仍然无法解析，返回空字典
                return {}
    
    def get_enhanced_keys(self, key_name):
        """
        根据输入的键名生成增强匹配的键名列表
        包含 en/zh 语言增强匹配和常见变体
        """
        if not key_name.strip():
            return []
            
        key_lower = key_name.lower().strip()
        enhanced_keys = [key_name]  # 原始键名
        
        # 添加大小写变体
        enhanced_keys.extend([
            key_name.upper(),
            key_name.lower(),
            key_name.capitalize()
        ])
        
        # 改进的语言匹配逻辑
        # 英文相关键名
        english_variants = ["English", "english", "英文", "英语", "ENGLISH", "ENG", "eng", "en", "EN"]
        # 中文相关键名  
        chinese_variants = ["Chinese", "chinese", "中文", "China", "china", "CHINESE", "CHN", "chn", "ZH", "zh"]
        
        # 检查是否为英文相关键名
        if key_lower in ['en', 'english', 'eng'] or key_name in ['英文', '英语']:
            enhanced_keys.extend(english_variants)
        # 检查是否为中文相关键名
        elif key_lower in ['zh', 'chinese', 'chn', 'china'] or key_name in ['中文']:
            enhanced_keys.extend(chinese_variants)
        
        # 根据常见模式添加变体
        common_patterns = {
            'x': ['X', 'pos_x', 'position_x', 'coord_x', 'horizontal'],
            'y': ['Y', 'pos_y', 'position_y', 'coord_y', 'vertical'],
            'width': ['w', 'W', 'WIDTH', 'size_w'],
            'height': ['h', 'H', 'HEIGHT', 'size_h'],
            'id': ['ID', 'Id', 'identifier', 'key'],
            'name': ['NAME', 'Name', 'title', 'label'],
            'value': ['VALUE', 'Value', 'val', 'data'],
            'a': ['A', 'alpha', 'first'],
            'b': ['B', 'beta', 'second']
        }
        
        if key_lower in common_patterns:
            enhanced_keys.extend(common_patterns[key_lower])
        
        # 去重并保持顺序
        seen = set()
        unique_keys = []
        for key in enhanced_keys:
            if key not in seen:
                seen.add(key)
                unique_keys.append(key)
        
        return unique_keys
    
    def find_key_in_object(self, obj, key_name, precision_match):
        """
        在对象中查找键值，支持精确匹配模式
        precision_match="enabled": 精确匹配（只匹配完全相同的键名）
        precision_match="disabled": 增强匹配（支持多种键名变体和语言匹配）
        """
        if not isinstance(obj, dict):
            return None
        
        if not key_name.strip():
            return None
            
        if precision_match == "enabled":
            # 精确匹配模式：只匹配完全相同的键名
            return obj.get(key_name)
        else:
            # 增强匹配模式：尝试多种键名变体
            enhanced_keys = self.get_enhanced_keys(key_name)
            for key in enhanced_keys:
                if key in obj:
                    return obj[key]
            # 不区分大小写搜索
            for obj_key, obj_value in obj.items():
                if obj_key.lower() == key_name.lower():
                    return obj_value
        
        return None
    
    def auto_convert_value(self, value):
        """
        根据值的类型自动转换为对应类型
        """
        if value is None:
            return None
        
        # 如果已经是数字类型，直接返回
        if isinstance(value, (int, float)):
            return value
        
        # 如果是字符串，尝试转换为数字
        if isinstance(value, str):
            # 尝试转换为整数
            try:
                if '.' not in value and 'e' not in value.lower():
                    return int(value)
            except (ValueError, TypeError):
                pass
            
            # 尝试转换为浮点数
            try:
                return float(value)
            except (ValueError, TypeError):
                pass
            
            # 无法转换为数字，返回原字符串
            return value
        
        # 其他类型直接返回
        return value
    
    def text_custom_extract(self, json_data, key, precision_match):
        try:
            # 解析 JSON 数据（增强版）
            data = self.parse_json_data(json_data)
            
            values = []
            
            # 判断是单个对象还是数组
            if isinstance(data, list):
                # 数组形式：遍历每个对象
                for item in data:
                    value = self.find_key_in_object(item, key, precision_match)
                    if value is not None:
                        converted_value = self.auto_convert_value(value)
                        values.append(converted_value)
            elif isinstance(data, dict):
                # 单个对象：直接提取
                value = self.find_key_in_object(data, key, precision_match)
                if value is not None:
                    converted_value = self.auto_convert_value(value)
                    values.append(converted_value)
            
            # 如果没有找到任何值，返回空列表
            if not values:
                values = []
            
            return (values,)
            
        except Exception as e:
            print(f"Error in text custom extract: {e}")
            # 返回空列表
            return ([],)


# 节点映射
NODE_CLASS_MAPPINGS = {
    "TextCustomList": TextCustomList,
    "TextCustomExtract": TextCustomExtract,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TextCustomList": "Text Custom List",
    "TextCustomExtract": "Text Custom Extract",
}