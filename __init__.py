import atexit
import importlib
import inspect
import logging
import os
import pkgutil
import signal
import sys

from comfy_api.latest import ComfyExtension, io, ui


_CACHED_NODES = None


def _discover_nodes() -> list[type[io.ComfyNode]]:
    """
    递归扫描并导入 `nodes` 子包下的所有节点类，统一注册。

    - 仅在包级入口进行注册，节点文件不实现扩展入口。
    - 严格返回继承自 `io.ComfyNode` 的类列表。
    """
    global _CACHED_NODES
    if _CACHED_NODES is not None:
        return _CACHED_NODES

    current_dir = os.path.dirname(os.path.realpath(__file__))
    nodes_dir = os.path.join(current_dir, "nodes")

    discovered: list[type[io.ComfyNode]] = []

    if not os.path.isdir(nodes_dir):
        logging.info("[1hewNodesV3] 未检测到 nodes 目录，跳过扫描")
        _CACHED_NODES = discovered
        return discovered

    package_name = __name__

    # 导入顶层 nodes 包以获取遍历 __path__
    try:
        nodes_pkg = importlib.import_module(".nodes", package=package_name)
    except Exception as exc:  # pragma: no cover
        logging.error(f"[1hewNodesV3] 导入 nodes 包失败: {exc}")
        _CACHED_NODES = discovered
        return discovered

    prefix = f"{package_name}.nodes."
    for _, name, _ in pkgutil.walk_packages(nodes_pkg.__path__, prefix=prefix):
        # 跳过包级 __init__，按模块导入
        try:
            module = importlib.import_module(name)
        except Exception as exc:  # pragma: no cover
            logging.error(f"[1hewNodesV3] 导入模块失败: {name}, 错误: {exc}")
            continue

        for _, obj in inspect.getmembers(module, inspect.isclass):
            try:
                if issubclass(obj, io.ComfyNode) and obj is not io.ComfyNode:
                    discovered.append(obj)
            except Exception:
                continue

    _CACHED_NODES = discovered
    return discovered


class NodesV3Extension(ComfyExtension):
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return _discover_nodes()


async def comfy_entrypoint() -> ComfyExtension:
    return NodesV3Extension()

WEB_DIRECTORY = os.path.join(os.path.dirname(os.path.realpath(__file__)), "web")


# 前端脚本注册，确保浏览器加载扩展 JS
try:
    import nodes
    js_dir = os.path.join(WEB_DIRECTORY, "js")
    if os.path.isdir(js_dir):
        nodes.EXTENSION_WEB_DIRS["ComfyUI-1hewNodesV3"] = js_dir
except Exception:
    pass


# 全局变量保存监控模块引用
_workflow_watcher = None
_prev_sigint_handler = None
_prev_sigterm_handler = None

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
    except Exception as e:
        logging.error(f"[ComfyUI-1hewNodes] 停止工作流监控时出错：{str(e)}")

def _register_shutdown_hooks():
    """注册关闭钩子和信号处理"""
    # 注册atexit钩子
    atexit.register(_stop_workflow_monitor)
    
    # 注册信号处理（仅在主线程中有效）
    try:
        global _prev_sigint_handler, _prev_sigterm_handler
        if hasattr(signal, 'SIGINT'):
            _prev_sigint_handler = signal.getsignal(signal.SIGINT)
            signal.signal(signal.SIGINT, _signal_handler)
        if hasattr(signal, 'SIGTERM'):
            _prev_sigterm_handler = signal.getsignal(signal.SIGTERM)
            signal.signal(signal.SIGTERM, _signal_handler)
    except ValueError:
        # 在非主线程中注册信号处理会失败，这是正常的
        pass

def _signal_handler(signum, frame):
    """信号处理函数"""
    logging.info(f"[ComfyUI-1hewNodes] 接收到信号 {signum}，正在停止工作流监控...")
    _stop_workflow_monitor()
    # 保持默认终止行为或链式调用原处理器
    try:
        if hasattr(signal, 'SIGINT') and signum == signal.SIGINT:
            # 如果原处理器是默认或未设置，则触发默认 KeyboardInterrupt
            if _prev_sigint_handler in (signal.SIG_DFL, None, signal.default_int_handler):
                raise KeyboardInterrupt
            # 如果原处理器不是忽略，则链式调用原处理器
            if _prev_sigint_handler is not None and _prev_sigint_handler != signal.SIG_IGN:
                _prev_sigint_handler(signum, frame)
        elif hasattr(signal, 'SIGTERM') and signum == signal.SIGTERM:
            if _prev_sigterm_handler not in (None, signal.SIG_IGN):
                _prev_sigterm_handler(signum, frame)
    except KeyboardInterrupt:
        # 让上层按默认流程处理退出
        raise

# 自动启动工作流监控
_start_workflow_monitor()
