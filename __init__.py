import os
import importlib
import importlib.util
import sys

# 初始化映射字典
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
# 添加自定义类型
CUSTOM_TYPES = {
    "CROP_BBOX": {"display_name": "crop_bbox", "color": (107, 176, 255)}  # 蓝色
}

# 获取当前目录
current_dir = os.path.dirname(os.path.realpath(__file__))

# 自动发现和导入 nodes 目录下的所有节点模块
nodes_dir = os.path.join(current_dir, "nodes")
if os.path.exists(nodes_dir) and os.path.isdir(nodes_dir):
    # 将 nodes 目录添加到 Python 路径
    if nodes_dir not in sys.path:
        sys.path.append(nodes_dir)
    
    # 遍历 nodes 目录下的所有 Python 文件
    for file in os.listdir(nodes_dir):
        if file.endswith(".py") and file != "__init__.py":
            module_name = file[:-3]  # 去掉 .py 后缀
            try:
                # 导入模块
                module = importlib.import_module(f".{module_name}", package=f"{__name__}.nodes")
                
                # 如果模块有节点映射，则添加到全局映射中
                if hasattr(module, "NODE_CLASS_MAPPINGS"):
                    NODE_CLASS_MAPPINGS.update(module.NODE_CLASS_MAPPINGS)
                if hasattr(module, "NODE_DISPLAY_NAME_MAPPINGS"):
                    NODE_DISPLAY_NAME_MAPPINGS.update(module.NODE_DISPLAY_NAME_MAPPINGS)
            except Exception as e:
                print(f"导入节点模块 {module_name} 时出错: {e}")

# 导出映射
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]