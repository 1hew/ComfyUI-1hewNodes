

class AnyEmptyBool:
    """
    通用空值检查节点（布尔输出版本）
    检查任意类型输入是否为空，返回布尔值
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "any": ("*", {"forceInput": True}),
            }
        }

    @classmethod
    def VALIDATE_INPUTS(cls, input_types):
        # 对于通配输入，跳过类型校验
        return True

    RETURN_TYPES = ("BOOLEAN",)
    RETURN_NAMES = ("bool",)
    FUNCTION = "check_empty"
    CATEGORY = "1hewNodes/logic"

    def check_empty(self, any):
        """
        检查输入是否为空，返回布尔值
        """
        try:
            return (self._is_empty(any),)
        except Exception as e:
            print(f"AnyEmptyBool error: {e}")
            # 发生错误时默认返回 True（空值）
            return (True,)

    def _is_empty(self, value):
        """
        判断各种类型的值是否为空
        """
        # None 检查
        if value is None:
            return True
        
        # 字符串类型检查
        if isinstance(value, str):
            return len(value.strip()) == 0
        
        # 布尔类型检查
        if isinstance(value, bool):
            return not value
        
        # 数值类型检查
        if isinstance(value, (int, float)):
            return value == 0
        
        # PyTorch tensor 检查 (image, mask 等)
        if isinstance(value, torch.Tensor):
            # 检查是否为空 tensor
            if value.numel() == 0:
                return True
            # 检查是否所有值都为 0
            return torch.all(value == 0).item()
        
        # NumPy array 检查
        if isinstance(value, np.ndarray):
            if value.size == 0:
                return True
            return np.all(value == 0)
        
        # 列表和元组检查
        if isinstance(value, (list, tuple)):
            if len(value) == 0:
                return True
            # 检查是否所有元素都为空
            return all(self._is_empty(item) for item in value)
        
        # 字典检查
        if isinstance(value, dict):
            return len(value) == 0
        
        # 其他可迭代对象检查
        if hasattr(value, '__len__'):
            try:
                return len(value) == 0
            except:
                pass
        
        # 如果无法判断，默认认为不为空
        return False


class AnyEmptyInt:
    """
    通用空值检查节点（整数输出版本）
    检查任意类型输入是否为空，返回自定义的整数值
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "any": ("*", {"forceInput": True}),
                "empty": ("INT", {"default": 0, "min": -999999, "max": 999999, "step": 1}),
                "not_empty": ("INT", {"default": 1, "min": -999999, "max": 999999, "step": 1}),
            }
        }

    @classmethod
    def VALIDATE_INPUTS(cls, input_types):
        # 对于通配输入，跳过类型校验
        return True

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("int",)
    FUNCTION = "check_empty"
    CATEGORY = "1hewNodes/logic"

    def check_empty(self, any, empty=0, not_empty=1):
        """
        检查输入是否为空，返回对应的整数值
        """
        try:
            is_empty = self._is_empty(any)
            return (empty if is_empty else not_empty,)
        except Exception as e:
            print(f"AnyEmptyInt error: {e}")
            # 发生错误时默认返回空值
            return (empty,)

    def _is_empty(self, value):
        """
        判断各种类型的值是否为空
        """
        # None 检查
        if value is None:
            return True
        
        # 字符串类型检查
        if isinstance(value, str):
            return len(value.strip()) == 0
        
        # 布尔类型检查
        if isinstance(value, bool):
            return not value
        
        # 数值类型检查
        if isinstance(value, (int, float)):
            return value == 0
        
        # PyTorch tensor 检查 (image, mask 等)
        if isinstance(value, torch.Tensor):
            # 检查是否为空 tensor
            if value.numel() == 0:
                return True
            # 检查是否所有值都为 0
            return torch.all(value == 0).item()
        
        # NumPy array 检查
        if isinstance(value, np.ndarray):
            if value.size == 0:
                return True
            return np.all(value == 0)
        
        # 列表和元组检查
        if isinstance(value, (list, tuple)):
            if len(value) == 0:
                return True
            # 检查是否所有元素都为空
            return all(self._is_empty(item) for item in value)
        
        # 字典检查
        if isinstance(value, dict):
            return len(value) == 0
        
        # 其他可迭代对象检查
        if hasattr(value, '__len__'):
            try:
                return len(value) == 0
            except:
                pass
        
        # 如果无法判断，默认认为不为空
        return False


class AnySwitchBool:
    """
    通用布尔切换节点，支持任意类型输入和惰性求值
    根据布尔值条件选择输出 on_true 或 on_false 的值
    """
    
    class AnyType(str):
        """用于表示任意类型的特殊类，在类型比较时总是返回相等"""
        def __eq__(self, _) -> bool:
            return True

        def __ne__(self, __value: object) -> bool:
            return False

    any = AnyType("*")
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "boolean": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "on_true": (cls.any, {"forceInput": True, "lazy": True}),
                "on_false": (cls.any, {"forceInput": True, "lazy": True}),
            }
        }

    @classmethod
    def VALIDATE_INPUTS(cls, input_types):
        # 对于通配输入，跳过类型校验
        return True
    
    RETURN_TYPES = (any,)
    RETURN_NAMES = ("output",)
    FUNCTION = "switch"
    CATEGORY = "1hewNodes/logic"
    
    def check_lazy_status(self, boolean, on_true=None, on_false=None):
        """
        惰性求值控制：只求值需要的分支
        """
        if boolean:
            return ["on_true"]
        else:
            return ["on_false"]
    
    def switch(self, boolean, on_true=None, on_false=None):
        """
        根据布尔值条件进行切换
        """
        try:
            if boolean:
                return (on_true,)
            else:
                return (on_false,)
        except Exception as e:
            print(f"AnySwitchBool error: {e}")
            return (None,)
    

class AnySwitchInt:
    """
    多路整数切换节点，支持多个输入选项的切换
    根据整数索引（1-5）选择对应的输入输出
    """
    
    class AnyType(str):
        """用于表示任意类型的特殊类，在类型比较时总是返回相等"""
        def __eq__(self, _) -> bool:
            return True

        def __ne__(self, __value: object) -> bool:
            return False

    any = AnyType("*")
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "select": ("INT", {"default": 1, "min": 1, "max": 5, "step": 1}),
            },
            "optional": {
                "input_1": (cls.any, {"forceInput": True, "lazy": True}),
                "input_2": (cls.any, {"forceInput": True, "lazy": True}),
                "input_3": (cls.any, {"forceInput": True, "lazy": True}),
                "input_4": (cls.any, {"forceInput": True, "lazy": True}),
                "input_5": (cls.any, {"forceInput": True, "lazy": True}),
            }
        }

    @classmethod
    def VALIDATE_INPUTS(cls, input_types):
        # 对于通配输入，跳过类型校验
        return True
    
    RETURN_TYPES = (any,)
    RETURN_NAMES = ("output",)
    FUNCTION = "switch_multi"
    CATEGORY = "1hewNodes/logic"
    
    def check_lazy_status(self, select, **kwargs):
        """
        惰性求值控制：只求值选中的输入
        """
        input_key = f"input_{select}"
        return [input_key]
    
    def switch_multi(self, select, **kwargs):
        """
        根据整数索引选择对应的输入
        """
        try:
            input_key = f"input_{select}"
            
            if input_key in kwargs:
                return (kwargs[input_key],)
            else:
                # 如果选中的输入不存在，返回 input_1 作为回退
                return (kwargs.get("input_1", None),)
                
        except Exception as e:
            print(f"AnySwitchInt error: {e}")
            return (None,)



NODE_CLASS_MAPPINGS = {
    "AnyEmptyBool": AnyEmptyBool,
    "AnyEmptyInt": AnyEmptyInt,
    "AnySwitchBool": AnySwitchBool,
    "AnySwitchInt": AnySwitchInt,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AnyEmptyBool": "Any Empty Bool",
    "AnyEmptyInt": "Any Empty Int",
    "AnySwitchBool": "Any Switch Bool",
    "AnySwitchInt": "Any Switch Int",
}
