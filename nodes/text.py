import json
import re
import random
import time


class TextCustomExtract:
    """
    文本自定义提取器 - 从JSON对象或数组中提取指定键的值
    支持精确匹配模式和增强匹配模式
    支持基于label值过滤并提取对应key的值
    输出格式：数组时分行展示，单个对象时直接输出字符串
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "json_data": ("STRING", {
                    "multiline": True,
                    "placeholder": "Enter json object or array data"
                }),
                "key": ("STRING", {
                    "default": "zh",
                    "placeholder": "Key name to extract"
                }),
                "precision_match": (["disabled", "enabled"], {"default": "disabled"})
            },
            "optional": {
                "label_filter": ("STRING", {
                    "default": "",
                    "placeholder": "Filter by label values (comma separated, supports partial match)"
                })
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("string",)
    FUNCTION = "text_custom_extract"
    CATEGORY = "1hewNodes/text"
    
    def clean_input_text(self, text: str) -> str:
        """清理输入文本，移除多余的格式标记（借鉴SimpleBBoxConverter）"""
        text = text.strip()
        
        # 移除代码块标记
        if "```json" in text:
            text = text.split("```json", 1)[1]
            text = text.split("```", 1)[0]
        elif "```" in text:
            parts = text.split("```")
            if len(parts) >= 3:
                text = parts[1]
        
        # 移除可能的前缀文字（保留JSON结构）
        # 查找第一个 [ 或 { 的位置
        start_pos = -1
        for i, char in enumerate(text):
            if char in '[{':
                start_pos = i
                break
        
        if start_pos != -1:
            text = text[start_pos:]
        
        # 查找最后一个 ] 或 } 的位置
        end_pos = -1
        for i in range(len(text) - 1, -1, -1):
            if text[i] in ']}':
                end_pos = i + 1
                break
        
        if end_pos != -1:
            text = text[:end_pos]
        
        return text
    
    def parse_json_data(self, json_data):
        """
        解析JSON数据，采用SimpleBBoxConverter的多层次解析策略
        """
        # 首先清理输入文本
        if isinstance(json_data, str):
            text = self.clean_input_text(json_data)
        else:
            text = str(json_data)
        
        if not text:
            print("清理后的文本为空")
            return {}
        
        print(f"清理后的文本: {text[:200]}...")
        
        try:
            # 尝试直接解析JSON
            data = json.loads(text)
            print(f"JSON解析成功，数据类型: {type(data)}")
            return data
        except json.JSONDecodeError as e:
            print(f"JSON解析失败: {e}")
            try:
                # 尝试使用ast.literal_eval
                import ast
                data = ast.literal_eval(text)
                print(f"ast.literal_eval解析成功，数据类型: {type(data)}")
                return data
            except (ValueError, SyntaxError) as e:
                print(f"ast.literal_eval解析失败: {e}")
                # 尝试修复常见格式问题
                try:
                    # 修复单引号问题
                    fixed_text = text.replace("'", '"')
                    data = json.loads(fixed_text)
                    print(f"修复单引号后解析成功，数据类型: {type(data)}")
                    return data
                except json.JSONDecodeError as e:
                    print(f"修复单引号后仍然解析失败: {e}")
                    # 最后尝试正则表达式提取
                    try:
                        pattern = r'"([^"]+)"\s*:\s*"([^"]+)"'
                        matches = re.findall(pattern, text)
                        if matches:
                            result = {key: value for key, value in matches}
                            print(f"正则表达式提取成功: {result}")
                            return result
                    except Exception as e:
                        print(f"正则表达式提取失败: {e}")
                    
                    print(f"所有解析方法都失败，返回空字典")
                    return {}
    
    def get_enhanced_keys(self, key_name):
        """
        根据输入的键名生成增强匹配的键名列表
        简化匹配逻辑，提高成功率
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
        
        # 常见键名映射
        key_mappings = {
            'bbox': ['bbox', 'BBOX', 'Bbox', 'box', 'bounding_box', 'coordinates', 'coord', 'bbox_2d'],
            'label': ['label', 'LABEL', 'Label', 'name', 'title', 'text', 'class'],
            'confidence': ['confidence', 'CONFIDENCE', 'conf', 'score', 'probability', 'prob'],
            'x': ['x', 'X', 'pos_x', 'position_x'],
            'y': ['y', 'Y', 'pos_y', 'position_y'],
            'width': ['width', 'w', 'W', 'WIDTH'],
            'height': ['height', 'h', 'H', 'HEIGHT'],
            'zh': ['zh', 'ZH', 'chinese', 'Chinese', 'CHINESE', '中文'],
            'en': ['en', 'EN', 'english', 'English', 'ENGLISH', '英文', '英语']
        }
        
        # 检查是否有对应的映射
        for base_key, variants in key_mappings.items():
            if key_lower == base_key or key_name in variants:
                enhanced_keys.extend(variants)
                break
        
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
        """
        if not isinstance(obj, dict):
            return None
        
        if not key_name.strip():
            return None
            
        print(f"在对象中查找键: {key_name}, 对象键: {list(obj.keys())}")
            
        if precision_match == "enabled":
            # 精确匹配模式：只匹配完全相同的键名
            result = obj.get(key_name)
            print(f"精确匹配结果: {result}")
            return result
        else:
            # 增强匹配模式：尝试多种键名变体
            enhanced_keys = self.get_enhanced_keys(key_name)
            print(f"增强匹配键列表: {enhanced_keys}")
            
            for key in enhanced_keys:
                if key in obj:
                    result = obj[key]
                    print(f"找到匹配键 '{key}': {result}")
                    return result
            
            # 如果还是没找到，尝试不区分大小写搜索
            key_lower = key_name.lower()
            for obj_key, obj_value in obj.items():
                if obj_key.lower() == key_lower:
                    print(f"不区分大小写找到匹配键 '{obj_key}': {obj_value}")
                    return obj_value
        
        print(f"未找到匹配的键")
        return None
    
    def format_value_as_string(self, value):
        """
        将值格式化为字符串
        """
        if value is None:
            return ""
        
        # 如果是列表或数组，转换为字符串表示
        if isinstance(value, list):
            # 如果是数字列表，格式化为逗号分隔
            if all(isinstance(x, (int, float)) for x in value):
                return ", ".join(map(str, value))
            else:
                return str(value)
        
        # 其他类型直接转换为字符串
        return str(value)
    
    def parse_label_filters(self, label_filter):
        """
        解析label过滤器字符串，支持中文逗号和英文逗号分隔的多个过滤条件
        """
        if not label_filter or not label_filter.strip():
            return []
        
        # 先将中文逗号替换为英文逗号，然后按英文逗号分割并清理空白
        normalized_filter = label_filter.replace('，', ',')
        filters = [f.strip() for f in normalized_filter.split(',') if f.strip()]
        return filters
    
    def matches_label_filter(self, label_value, filters):
        """
        检查label值是否匹配任一过滤条件（支持部分匹配）
        """
        if not filters:
            return True  # 没有过滤条件时，匹配所有
        
        if not label_value:
            return False
        
        label_str = str(label_value).lower()
        
        # 检查是否有任一过滤条件匹配
        for filter_str in filters:
            if filter_str.lower() in label_str:
                return True
        
        return False
    
    def text_custom_extract(self, json_data, key, precision_match, label_filter=""):
        try:
            print(f"开始提取，键名: {key}, 精确匹配: {precision_match}, label过滤: {label_filter}")
            print(f"输入数据: {json_data[:200]}...")
            
            # 解析 JSON 数据（使用改进的解析方法）
            data = self.parse_json_data(json_data)
            
            if not data:
                print("解析后的数据为空")
                return ("",)
            
            # 解析label过滤条件
            label_filters = self.parse_label_filters(label_filter)
            print(f"Label过滤条件: {label_filters}")
            
            values = []
            
            # 判断是单个对象还是数组
            if isinstance(data, list):
                print(f"处理数组，包含 {len(data)} 个元素")
                # 数组形式：遍历每个对象，提取键值
                for i, item in enumerate(data):
                    print(f"处理第 {i+1} 个元素: {type(item)}")
                    if isinstance(item, dict):
                        # 如果有label过滤条件，先检查label是否匹配
                        if label_filters:
                            # 查找label字段（支持多种label键名）
                            label_value = self.find_key_in_object(item, "label", "disabled")
                            if not self.matches_label_filter(label_value, label_filters):
                                print(f"第 {i+1} 个元素的label '{label_value}' 不匹配过滤条件，跳过")
                                continue
                            else:
                                print(f"第 {i+1} 个元素的label '{label_value}' 匹配过滤条件")
                        
                        # 提取指定key的值
                        value = self.find_key_in_object(item, key, precision_match)
                        if value is not None:
                            formatted_value = self.format_value_as_string(value)
                            values.append(formatted_value)
                            print(f"提取到值: {formatted_value}")
                
                # 数组格式：分行展示
                if values:
                    result = "\n".join(values)
                    print(f"最终结果 (数组): {result}")
                else:
                    result = ""
                    print("未找到任何匹配的值")
                    
            elif isinstance(data, dict):
                print(f"处理单个对象，键: {list(data.keys())}")
                
                # 如果有label过滤条件，先检查label是否匹配
                if label_filters:
                    label_value = self.find_key_in_object(data, "label", "disabled")
                    if not self.matches_label_filter(label_value, label_filters):
                        print(f"对象的label '{label_value}' 不匹配过滤条件")
                        result = ""
                    else:
                        print(f"对象的label '{label_value}' 匹配过滤条件")
                        # 单个对象：直接提取
                        value = self.find_key_in_object(data, key, precision_match)
                        if value is not None:
                            result = self.format_value_as_string(value)
                            print(f"提取到值: {result}")
                        else:
                            result = ""
                            print("未找到匹配的键")
                else:
                    # 没有label过滤，直接提取
                    value = self.find_key_in_object(data, key, precision_match)
                    if value is not None:
                        result = self.format_value_as_string(value)
                        print(f"提取到值: {result}")
                    else:
                        result = ""
                        print("未找到匹配的键")
            else:
                print(f"不支持的数据类型: {type(data)}")
                result = ""
            
            return (result,)
            
        except Exception as e:
            print(f"提取过程中发生错误: {e}")
            import traceback
            traceback.print_exc()
            return ("",)


class ListCustomInt:
    """
    自定义整数列表节点 - 生成整数类型的列表
    支持连字符分割和多种分隔符
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "custom_text": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "placeholder": '-- splits override separator\nelse use "," ";" or newline.'
                })
            }
        }
    
    RETURN_TYPES = ("INT", "INT")
    RETURN_NAMES = ("int_list", "count")
    OUTPUT_IS_LIST = (True, False)
    CATEGORY = "1hewNodes/text"
    FUNCTION = "list_custom_int"
    
    def parse_text_list(self, text):
        """
        解析文本列表，支持连字符分割和多种分隔符
        当有只包含连字符的行时，只按 -- 进行分割，其他分割方式失效
        否则按照逗号(,)、分号(;)、换行符(\n) 分割
        """
        if not text.strip():
            return [0]
        
        # 检查是否有只包含连字符的行
        lines = text.split('\n')
        has_dash_separator = any(line.strip() and all(c == '-' for c in line.strip()) for line in lines)
        
        if has_dash_separator:
            # 按连字符分割，其他分割方式失效（包括换行符）
            sections = re.split(r'^\s*-+\s*$', text, flags=re.MULTILINE)
            all_lists = []
            
            for section in sections:
                section = section.strip()
                if not section:
                    continue
                    
                # 当有连字符分割时，每个段落作为一个完整项目，不进行任何其他分割
                # 移除引号
                if (section.startswith('"') and section.endswith('"')) or (section.startswith("'") and section.endswith("'")):
                    section = section[1:-1]
                if section:
                    try:
                        # 转换为整数，如果是浮点数则取整
                        if '.' in section:
                            int_val = int(float(section))
                        else:
                            int_val = int(section)
                        all_lists.append(int_val)
                    except (ValueError, TypeError):
                        # 无法转换为整数，跳过
                        continue
            
            return all_lists if all_lists else [0]
        else:
            # 按传统方式分割
            return self._parse_section(text)
    
    def _parse_section(self, text):
        """
        解析单个文本段落并转换为整数，支持中英文逗号和分号
        """
        if not text.strip():
            return []
        
        lines = text.split('\n')
        int_list = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # 处理中英文逗号和分号分隔的项目
            if ',' in line or ';' in line or '，' in line or '；' in line:
                # 统一替换为英文逗号
                line = line.replace(';', ',').replace('，', ',').replace('；', ',')
                items = line.split(',')
                for item in items:
                    item = item.strip()
                    # 移除引号
                    if (item.startswith('"') and item.endswith('"')) or (item.startswith("'") and item.endswith("'")):
                        item = item[1:-1]
                    if item:
                        try:
                            # 转换为整数，如果是浮点数则取整
                            if '.' in item:
                                int_val = int(float(item))
                            else:
                                int_val = int(item)
                            int_list.append(int_val)
                        except (ValueError, TypeError):
                            # 无法转换为整数，跳过
                            continue
            else:
                # 单个项目，移除引号
                if (line.startswith('"') and line.endswith('"')) or (line.startswith("'") and line.endswith("'")):
                    line = line[1:-1]
                if line:
                    try:
                        # 转换为整数，如果是浮点数则取整
                        if '.' in line:
                            int_val = int(float(line))
                        else:
                            int_val = int(line)
                        int_list.append(int_val)
                    except (ValueError, TypeError):
                        # 无法转换为整数，跳过
                        continue
        
        return int_list if int_list else [0]
    
    def list_custom_int(self, custom_text):
        """
        生成整数类型的自定义列表
        """
        int_list = self.parse_text_list(custom_text)
        return (int_list, len(int_list))


class ListCustomFloat:
    """
    自定义浮点数列表节点 - 生成浮点数类型的列表
    支持连字符分割和多种分隔符
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "custom_text": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "placeholder": '-- splits override separator\nelse use "," ";" or newline.'
                })
            }
        }
    
    RETURN_TYPES = ("FLOAT", "INT")
    RETURN_NAMES = ("float_list", "count")
    OUTPUT_IS_LIST = (True, False)
    CATEGORY = "1hewNodes/text"
    FUNCTION = "list_custom_float"
    
    def parse_text_list(self, text):
        """
        解析文本列表，支持连字符分割和多种分隔符
        当有只包含连字符的行时，只按 -- 进行分割，其他分割方式失效
        否则按照逗号(,)、分号(;)、换行符(\n) 分割
        """
        if not text.strip():
            return [0.0]
        
        # 检查是否有只包含连字符的行
        lines = text.split('\n')
        has_dash_separator = any(line.strip() and all(c == '-' for c in line.strip()) for line in lines)
        
        if has_dash_separator:
            # 按连字符分割，其他分割方式失效（包括换行符）
            sections = re.split(r'^\s*-+\s*$', text, flags=re.MULTILINE)
            all_lists = []
            
            for section in sections:
                section = section.strip()
                if not section:
                    continue
                    
                # 当有连字符分割时，每个段落作为一个完整项目，不进行任何其他分割
                # 移除引号
                if (section.startswith('"') and section.endswith('"')) or (section.startswith("'") and section.endswith("'")):
                    section = section[1:-1]
                if section:
                    try:
                        float_val = float(section)
                        all_lists.append(float_val)
                    except (ValueError, TypeError):
                        # 无法转换为浮点数，跳过
                        continue
            
            return all_lists if all_lists else [0.0]
        else:
            # 按传统方式分割
            return self._parse_section(text)
    
    def _parse_section(self, text):
        """
        解析单个文本段落并转换为浮点数，支持中英文逗号和分号
        """
        if not text.strip():
            return []
        
        lines = text.split('\n')
        float_list = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # 处理中英文逗号和分号分隔的项目
            if ',' in line or ';' in line or '，' in line or '；' in line:
                # 统一替换为英文逗号
                line = line.replace(';', ',').replace('，', ',').replace('；', ',')
                items = line.split(',')
                for item in items:
                    item = item.strip()
                    # 移除引号
                    if (item.startswith('"') and item.endswith('"')) or (item.startswith("'") and item.endswith("'")):
                        item = item[1:-1]
                    if item:
                        try:
                            float_val = float(item)
                            float_list.append(float_val)
                        except (ValueError, TypeError):
                            # 无法转换为浮点数，跳过
                            continue
            else:
                # 单个项目，移除引号
                if (line.startswith('"') and line.endswith('"')) or (line.startswith("'") and line.endswith("'")):
                    line = line[1:-1]
                if line:
                    try:
                        float_val = float(line)
                        float_list.append(float_val)
                    except (ValueError, TypeError):
                        # 无法转换为浮点数，跳过
                        continue
        
        return float_list if float_list else [0.0]
    
    def list_custom_float(self, custom_text):
        """
        生成浮点数类型的自定义列表
        """
        float_list = self.parse_text_list(custom_text)
        return (float_list, len(float_list))


class ListCustomString:
    """
    自定义字符串列表节点 - 生成字符串类型的列表
    支持连字符分割和多种分隔符
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "custom_text": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "placeholder": '-- splits override separator\nelse use "," ";" or newline.'
                })
            }
        }
    
    RETURN_TYPES = ("STRING", "INT")
    RETURN_NAMES = ("string_list", "count")
    OUTPUT_IS_LIST = (True, False)
    CATEGORY = "1hewNodes/text"
    FUNCTION = "list_custom_string"
    
    def parse_text_list(self, text):
        """
        解析文本列表，支持连字符分割和多种分隔符
        当有只包含连字符的行时，只按 -- 进行分割，其他分割方式失效
        否则按照逗号(,)、分号(;)、换行符(\n) 分割
        """
        if not text.strip():
            return ["default"]
        
        # 检查是否有只包含连字符的行
        lines = text.split('\n')
        has_dash_separator = any(line.strip() and all(c == '-' for c in line.strip()) for line in lines)
        
        if has_dash_separator:
            # 按连字符分割，其他分割方式失效（包括换行符）
            sections = re.split(r'^\s*-+\s*$', text, flags=re.MULTILINE)
            all_lists = []
            
            for section in sections:
                section = section.strip()
                if not section:
                    continue
                    
                # 当有连字符分割时，每个段落作为一个完整项目，不进行任何其他分割
                # 移除引号
                if (section.startswith('"') and section.endswith('"')) or (section.startswith("'") and section.endswith("'")):
                    section = section[1:-1]
                if section:
                    all_lists.append(str(section))
            
            return all_lists if all_lists else ["default"]
        else:
            # 按传统方式分割（逗号、分号、换行符）
            return self._parse_section(text)
    
    def _parse_section(self, text):
        """
        解析单个文本段落，支持中英文逗号和分号
        """
        if not text.strip():
            return []
        
        lines = text.split('\n')
        text_list = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # 处理中英文逗号和分号分隔的项目
            if ',' in line or ';' in line or '，' in line or '；' in line:
                # 统一替换为英文逗号
                line = line.replace(';', ',').replace('，', ',').replace('；', ',')
                items = line.split(',')
                for item in items:
                    item = item.strip()
                    # 移除引号
                    if (item.startswith('"') and item.endswith('"')) or (item.startswith("'") and item.endswith("'")):
                        item = item[1:-1]
                    if item:
                        text_list.append(str(item))
            else:
                # 单个项目，移除引号
                if (line.startswith('"') and line.endswith('"')) or (line.startswith("'") and line.endswith("'")):
                    line = line[1:-1]
                if line:
                    text_list.append(str(line))
        
        return text_list if text_list else ["default"]
    
    def list_custom_string(self, custom_text):
        """
        生成字符串类型的自定义列表
        """
        text_list = self.parse_text_list(custom_text)
        # 确保所有项目都是字符串类型
        string_list = [str(item) for item in text_list]
        return (string_list, len(string_list))


class ListCustomSeed:
    """
    自定义种子列表节点 - 生成种子类型的列表
    基于输入种子生成指定数量的随机种子列表
    种子范围：0 到 1125899906842624
    保留control after generate功能，确保生成的种子不重复
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "seed": ("INT", {
                    "default": 42,
                    "min": 0,
                    "max": 1125899906842624,
                    "step": 1
                }),
                "count": ("INT", {
                    "default": 3,
                    "min": 1,
                    "max": 1000,
                    "step": 1
                })
            }
        }
    
    RETURN_TYPES = ("INT", "INT")
    RETURN_NAMES = ("seed_list", "count")
    OUTPUT_IS_LIST = (True, False)
    CATEGORY = "1hewNodes/text"
    FUNCTION = "list_custom_seed"
    
    def __init__(self):
        self.used_seeds = set()  # 用于跟踪已使用的随机种子
    
    def generate_unique_random_seeds(self, count):
        """
        生成不重复的随机种子列表
        """
        seeds = []
        max_attempts = count * 10  # 防止无限循环
        attempts = 0
        
        while len(seeds) < count and attempts < max_attempts:
            seed = random.randint(0, 1125899906842624)
            if seed not in self.used_seeds:
                seeds.append(seed)
                self.used_seeds.add(seed)
            attempts += 1
        
        # 如果无法生成足够的唯一种子，清空已使用集合并重新开始
        if len(seeds) < count:
            self.used_seeds.clear()
            while len(seeds) < count:
                seed = random.randint(0, 1125899906842624)
                if seed not in seeds:
                    seeds.append(seed)
        
        return seeds
    
    def clamp_seed(self, seed):
        """
        确保种子值在有效范围内
        """
        return max(0, min(seed, 1125899906842624))
    

    
    def list_custom_seed(self, seed, count):
        """
        生成种子类型的自定义列表
        基于输入种子生成指定数量的随机种子列表
        保留control after generate功能，确保生成的种子不重复
        """
        # 使用输入种子作为随机数生成器的种子
        random.seed(seed)
        
        # 生成不重复的随机种子列表
        seed_list = self.generate_unique_random_seeds(count)
        
        # 确保所有种子都在有效范围内
        seed_list = [self.clamp_seed(s) for s in seed_list]
        
        return (seed_list, len(seed_list))


# 节点映射
NODE_CLASS_MAPPINGS = {
    "TextCustomExtract": TextCustomExtract,
    "ListCustomInt": ListCustomInt,
    "ListCustomFloat": ListCustomFloat,
    "ListCustomString": ListCustomString,
    "ListCustomSeed": ListCustomSeed,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TextCustomExtract": "Text Custom Extract",
    "ListCustomInt": "List Custom Int", 
    "ListCustomFloat": "List Custom Float",
    "ListCustomString": "List Custom String",
    "ListCustomSeed": "List Custom Seed",
}