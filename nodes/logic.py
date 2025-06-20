class ImageListAppend:
    """
    图片列表追加节点 - 将图片收集为列表格式
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_1": ("IMAGE",),
                "image_2": ("IMAGE",)
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image_list",)
    FUNCTION = "image_list_append"
    CATEGORY = "1hewNodes/logic"
    
    def image_list_append(self, image_1, image_2):
        """
        将两个图片输入追加为列表
        """
        try:
            # 处理None值
            if image_1 is None and image_2 is None:
                return ([],)
            elif image_1 is None:
                return ([image_2],)
            elif image_2 is None:
                return ([image_1],)
            
            return self._append_to_list(image_1, image_2)
                
        except Exception as e:
            print(f"图片列表追加错误: {str(e)}")
            return ([image_1],)
    
    def _append_to_list(self, image_1, image_2):
        """
        将输入追加为列表，保持批量结构
        """
        result = []
        
        # 处理第一个输入
        if isinstance(image_1, list):
            result.extend(image_1)
        else:
            result.append(image_1)
        
        # 处理第二个输入
        if isinstance(image_2, list):
            result.extend(image_2)
        else:
            result.append(image_2)
        
        print(f"图片列表追加完成: 收集了{len(result)}个图片项目")
        return (result,)


# 节点映射
NODE_CLASS_MAPPINGS = {
    "ImageListAppend": ImageListAppend,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageListAppend": "Image List Append",
}