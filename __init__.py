import os
import importlib
import importlib.util
import sys
import logging
import atexit
import signal

# 初始化映射字典
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

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

# 添加这行来支持文档
WEB_DIRECTORY = os.path.join(current_dir, "web")

# 全局变量保存监控模块引用
_workflow_watcher = None

# 自动启动工作流监控功能
def _start_workflow_monitor():
    """启动工作流监控功能"""
    global _workflow_watcher
    try:
        # 导入工作流监控模块（从utils/workflow文件夹）
        from .utils.workflow import monitor
        _workflow_watcher = monitor
        
        # 启动监控
        success = monitor.start_workflow_monitor()
        if success:
            logging.info("[ComfyUI-1hewNodes] 工作流监控已自动启动")
            # 注册关闭钩子
            _register_shutdown_hooks()
        else:
            logging.warning("[ComfyUI-1hewNodes] 工作流监控启动失败")
    except Exception as e:
        logging.error(f"[ComfyUI-1hewNodes] 启动工作流监控时出错：{str(e)}")

def _stop_workflow_monitor():
    """停止工作流监控功能"""
    global _workflow_watcher
    try:
        if _workflow_watcher and _workflow_watcher.is_monitoring():
            _workflow_watcher.stop_workflow_monitor()
            logging.info("[ComfyUI-1hewNodes] 工作流监控已自动停止")
    except Exception as e:
        logging.error(f"[ComfyUI-1hewNodes] 停止工作流监控时出错：{str(e)}")

def _register_shutdown_hooks():
    """注册关闭钩子和信号处理"""
    # 注册atexit钩子
    atexit.register(_stop_workflow_monitor)
    
    # 注册信号处理（仅在主线程中有效）
    try:
        if hasattr(signal, 'SIGINT'):
            signal.signal(signal.SIGINT, _signal_handler)
        if hasattr(signal, 'SIGTERM'):
            signal.signal(signal.SIGTERM, _signal_handler)
    except ValueError:
        # 在非主线程中注册信号处理会失败，这是正常的
        pass

def _signal_handler(signum, frame):
    """信号处理函数"""
    logging.info(f"[ComfyUI-1hewNodes] 接收到信号 {signum}，正在停止工作流监控...")
    _stop_workflow_monitor()

# 自动启动工作流监控
_start_workflow_monitor()

# 导出映射
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]