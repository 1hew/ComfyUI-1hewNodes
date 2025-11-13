import os
import time
import folder_paths
from datetime import datetime


class WorkflowName:
    """
    自动获取当前工作流文件名
    通过监控临时文件获取最近保存的工作流文件名
    支持路径控制、自定义前缀和日期前缀
    """
    
    # 类级别的缓存，用于存储上次读取的文件修改时间和内容
    _last_mtime = 0
    _cached_content = ""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "prefix": ("STRING", {"default": ""}),
                "suffix": ("STRING", {"default": ""}),
                "date_format": (["none", "yyyy-MM-dd", "yyyy/MM/dd", "yyyyMMdd", "yyyy-MM-dd HH:mm", "yyyy/MM/dd HH:mm", "yyyy-MM-dd HH:mm:ss", "MM-dd", "MM/dd", "MMdd", "dd", "HH:mm", "HH:mm:ss", "yyyy年MM月dd日", "MM月dd日", "yyyyMMdd_HHmm", "yyyyMMdd_HHmmss"], {"default": "yyyy-MM-dd"}),
                "full_path": ("BOOLEAN", {"default": True}),
                "strip_extension": ("BOOLEAN", {"default": True})
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("string",)
    FUNCTION = "get_workflow_name"
    CATEGORY = "1hewNodes/util"
    
    # 让ComfyUI知道每次都要重新执行
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # 返回当前时间戳，确保每次都重新执行
        return str(time.time())

    def get_workflow_name(self, full_path=True, date_format="none", prefix="", suffix="", strip_extension=False):
        """读取临时文件中的工作流文件名，支持路径控制、前缀、后缀和日期功能"""
        try:
            # 动态获取ComfyUI根目录路径
            comfyui_root = folder_paths.base_path
            # 临时文件路径
            temp_file_path = os.path.join(comfyui_root, "custom_nodes", "ComfyUI-1hewNodes", "utils", "workflow", "current_workflow.tmp")
            
            if not os.path.exists(temp_file_path):
                return ("未检测到临时文件（请启动监控脚本）",)
            
            # 读取文件内容（最多尝试3次，处理文件锁定）
            for attempt in range(3):
                try:
                    with open(temp_file_path, "r", encoding="utf-8") as f:
                        file_path = f.read().strip()
                    
                    if not file_path:
                        return ("临时文件为空（未保存工作流）",)
                    
                    # 处理路径和文件名
                    result = self._process_workflow_path(file_path, full_path, date_format, prefix, suffix, strip_extension)
                    return (result,)
                        
                except PermissionError:
                    if attempt < 2:
                        time.sleep(0.05)
                        continue
                    else:
                        return ("无法读取临时文件（文件被占用）",)
                except Exception as read_error:
                    if attempt < 2:
                        time.sleep(0.05)
                        continue
                    else:
                        return (f"读取错误：{str(read_error)}",)
            
            return ("无法读取临时文件（多次尝试失败）",)
        
        except Exception as e:
            return (f"错误：{str(e)}",)
    
    def _process_workflow_path(self, file_path, full_path, date_format, prefix, suffix, strip_extension):
        """处理工作流路径，应用路径控制、前缀、后缀、日期和扩展名功能"""
        try:
            # 标准化路径分隔符
            file_path = file_path.replace("\\", "/")
            
            if full_path:
                # 输出完整路径信息（相对于workflows目录）
                if "/" in file_path:
                    dir_part, file_name = file_path.rsplit("/", 1)
                else:
                    dir_part = ""
                    file_name = file_path
            else:
                # 只输出文件名
                file_name = os.path.basename(file_path)
                dir_part = ""
            
            # 处理文件名和扩展名
            if file_name.endswith('.json'):
                name_without_ext = file_name[:-5]  # 去除.json
                extension = '.json'
            else:
                name_without_ext = file_name
                extension = ''
            
            # 添加前缀和后缀
            processed_name = name_without_ext
            if prefix:
                processed_name = prefix + processed_name
            if suffix:
                processed_name = processed_name + suffix
            
            # 决定是否保留扩展名
            if strip_extension:
                final_name = processed_name
            else:
                final_name = processed_name + extension
            
            # 构建最终路径
            if full_path and dir_part:
                result = f"{dir_part}/{final_name}"
            else:
                result = final_name
            
            # 添加日期前缀
            if date_format != "none":
                now = datetime.now()
                if date_format == "yyyy-MM-dd":
                    date_str = now.strftime("%Y-%m-%d")
                elif date_format == "yyyy/MM/dd":
                    date_str = now.strftime("%Y/%m/%d")
                elif date_format == "yyyyMMdd":
                    date_str = now.strftime("%Y%m%d")
                elif date_format == "yyyy-MM-dd HH:mm":
                    date_str = now.strftime("%Y-%m-%d %H:%M")
                elif date_format == "yyyy/MM/dd HH:mm":
                    date_str = now.strftime("%Y/%m/%d %H:%M")
                elif date_format == "yyyy-MM-dd HH:mm:ss":
                    date_str = now.strftime("%Y-%m-%d %H:%M:%S")
                elif date_format == "MM-dd":
                    date_str = now.strftime("%m-%d")
                elif date_format == "MM/dd":
                    date_str = now.strftime("%m/%d")
                elif date_format == "MMdd":
                    date_str = now.strftime("%m%d")
                elif date_format == "dd":
                    date_str = now.strftime("%d")
                elif date_format == "HH:mm":
                    date_str = now.strftime("%H:%M")
                elif date_format == "HH:mm:ss":
                    date_str = now.strftime("%H:%M:%S")
                elif date_format == "yyyy年MM月dd日":
                    date_str = now.strftime("%Y年%m月%d日")
                elif date_format == "MM月dd日":
                    date_str = now.strftime("%m月%d日")
                elif date_format == "yyyyMMdd_HHmm":
                    date_str = now.strftime("%Y%m%d_%H%M")
                elif date_format == "yyyyMMdd_HHmmss":
                    date_str = now.strftime("%Y%m%d_%H%M%S")
                else:
                    date_str = now.strftime("%Y-%m-%d")
                
                result = f"{date_str}/{result}"
            
            return result
            
        except Exception as e:
            return f"路径处理错误：{str(e)}"


class RangeMapping:
    """范围映射
    滑动条的数值会根据min和max_value的修改实时变化
    rounding参数控制小数位数精度
    """
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "value": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001, "display": "slider"}),
                    "min": ("FLOAT", {"default": 0.0, "min": -0xffffffffffffffff, "max": 0xffffffffffffffff, "step": 0.001, "display": "number"}),
                    "max": ("FLOAT", {"default": 1.0, "min": -0xffffffffffffffff, "max": 0xffffffffffffffff, "step": 0.001, "display": "number"}),
                    "rounding": ("INT", {"default": 3, "min": 0, "max": 10, "step": 1, "display": "number"}),
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
 

# 在NODE_CLASS_MAPPINGS中更新节点映射
NODE_CLASS_MAPPINGS = {
    "1hew_WorkflowName": WorkflowName,
    "1hew_RangeMapping": RangeMapping,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "1hew_WorkflowName": "Workflow Name",
    "1hew_RangeMapping": "Range Mapping",
}