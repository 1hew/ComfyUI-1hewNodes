import os
import time
import logging
import threading
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# ==================== 自动配置区 ====================
# 自动检测ComfyUI根目录（从utils/workflow目录向上查找）
current_dir = os.path.dirname(os.path.abspath(__file__))
UTILS_ROOT = os.path.dirname(current_dir)  # utils目录
PLUGIN_ROOT = os.path.dirname(UTILS_ROOT)  # ComfyUI-1hewNodes目录
COMFYUI_ROOT = os.path.dirname(os.path.dirname(PLUGIN_ROOT))  # 从ComfyUI-1hewNodes向上两级到ComfyUI根目录

# 自动检测可能的工作流目录
POSSIBLE_WORKFLOW_DIRS = [
    os.path.join(COMFYUI_ROOT, "user", "default", "workflows"),
    os.path.join(COMFYUI_ROOT, "workflows"),
    os.path.join(COMFYUI_ROOT, "user", "workflows"),
    os.path.join(os.path.expanduser("~"), "ComfyUI", "workflows"),
]

# 临时文件路径（保存在utils/workflow文件夹中）
TEMP_FILE_PATH = os.path.join(current_dir, "current_workflow.tmp")

# 监控日志级别（DEBUG/INFO/WARNING）
LOG_LEVEL = logging.INFO
# ================================================

# 配置日志
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S"
)

class WorkflowFileHandler(FileSystemEventHandler):
    def __init__(self):
        self.last_updated = 0  # 用于过滤短时间内的重复触发
        self.min_interval = 0.5  # 减少最小触发间隔（秒）
        self.last_file = ""  # 记录上次处理的文件，避免重复处理
        self.workflow_root = None  # 缓存工作流根目录

    def _get_workflow_root(self):
        """获取当前使用的工作流根目录"""
        if self.workflow_root is None:
            for workflow_dir in POSSIBLE_WORKFLOW_DIRS:
                if os.path.exists(workflow_dir) and os.path.isdir(workflow_dir):
                    self.workflow_root = workflow_dir
                    break
        return self.workflow_root

    def _get_relative_path(self, file_path):
        """获取相对于workflows目录的路径"""
        workflow_root = self._get_workflow_root()
        if workflow_root is None:
            return os.path.basename(file_path)
        
        # 标准化路径分隔符
        file_path = os.path.abspath(file_path).replace("\\", "/")
        workflow_root = os.path.abspath(workflow_root).replace("\\", "/")
        
        # 检查文件是否在workflows目录下
        if file_path.startswith(workflow_root):
            # 获取相对路径
            relative_path = file_path[len(workflow_root):].lstrip("/")
            return relative_path
        else:
            # 如果不在workflows目录下，只返回文件名
            return os.path.basename(file_path)

    def _update_temp_file(self, file_path, event_type="unknown"):
        """更新临时文件的通用方法"""
        current_time = time.time()
        relative_path = self._get_relative_path(file_path)
        
        # 避免重复处理同一文件
        if (current_time - self.last_updated > self.min_interval 
            and relative_path != self.last_file):
            
            self.last_updated = current_time
            self.last_file = relative_path
            
            # 写入临时文件
            try:
                with open(TEMP_FILE_PATH, "w", encoding="utf-8") as f:
                    f.write(relative_path)
                logging.info(f"[{event_type}] 已更新工作流路径：{relative_path}")
            except Exception as e:
                logging.error(f"写入临时文件失败：{str(e)}")

    def on_modified(self, event):
        """处理文件修改事件（保存操作会触发）"""
        if (not event.is_directory 
            and event.src_path.endswith(".json") 
            and os.path.abspath(event.src_path) != os.path.abspath(TEMP_FILE_PATH)):
            self._update_temp_file(event.src_path, "修改")

    def on_moved(self, event):
        """处理文件移动/重命名事件"""
        if (not event.is_directory 
            and event.dest_path.endswith(".json") 
            and os.path.abspath(event.dest_path) != os.path.abspath(TEMP_FILE_PATH)):
            self._update_temp_file(event.dest_path, "移动")

    def on_created(self, event):
        """处理文件创建事件"""
        if (not event.is_directory 
            and event.src_path.endswith(".json") 
            and os.path.abspath(event.src_path) != os.path.abspath(TEMP_FILE_PATH)):
            self._update_temp_file(event.src_path, "创建")

    def on_opened(self, event):
        """处理文件打开事件（如果支持）"""
        if (hasattr(event, 'src_path') and not event.is_directory 
            and event.src_path.endswith(".json") 
            and os.path.abspath(event.src_path) != os.path.abspath(TEMP_FILE_PATH)):
            self._update_temp_file(event.src_path, "打开")

def find_workflow_directory():
    """自动查找工作流目录"""
    for workflow_dir in POSSIBLE_WORKFLOW_DIRS:
        if os.path.exists(workflow_dir) and os.path.isdir(workflow_dir):
            # logging.info(f"找到工作流目录：{workflow_dir}")
            return workflow_dir
    
    # 如果都不存在，创建默认目录
    default_dir = POSSIBLE_WORKFLOW_DIRS[0]
    try:
        os.makedirs(default_dir, exist_ok=True)
        logging.info(f"创建默认工作流目录：{default_dir}")
        return default_dir
    except Exception as e:
        logging.error(f"无法创建工作流目录：{str(e)}")
        return None

def validate_workflow_dir():
    """验证工作流目录是否存在"""
    workflow_root = find_workflow_directory()
    if workflow_root is None:
        logging.error("无法找到或创建工作流目录")
        logging.info("可能的目录位置：")
        for dir_path in POSSIBLE_WORKFLOW_DIRS:
            logging.info(f"  - {dir_path}")
        return False, None
    return True, workflow_root

# 全局变量管理监控状态
_observer = None
_monitor_thread = None
_is_monitoring = False

def _monitor_worker():
    """监控工作线程"""
    global _observer, _is_monitoring
    
    # 验证并获取工作流目录
    is_valid, workflow_root = validate_workflow_dir()
    if not is_valid:
        logging.error("工作流监控启动失败：无法找到工作流目录")
        return
    
    event_handler = WorkflowFileHandler()
    _observer = Observer()
    # 监控目标目录及所有子目录（recursive=True）
    _observer.schedule(event_handler, workflow_root, recursive=True)
    
    try:
        _observer.start()
        _is_monitoring = True
        
        # 保持监控运行
        while _is_monitoring:
            time.sleep(1)
            
    except Exception as e:
        logging.error(f"[ComfyUI-1hewNodes] 工作流监控启动失败：{str(e)}")
    finally:
        if _observer:
            _observer.stop()
            _observer.join()
        _is_monitoring = False

def start_workflow_monitor():
    """启动工作流监控（后台线程模式）"""
    global _monitor_thread, _is_monitoring
    
    if _is_monitoring:
        return True
    
    try:
        _monitor_thread = threading.Thread(target=_monitor_worker, daemon=True)
        _monitor_thread.start()
        return True
    except Exception as e:
        logging.error(f"[ComfyUI-1hewNodes] 启动工作流监控线程失败：{str(e)}")
        return False

def stop_workflow_monitor():
    """停止工作流监控"""
    global _is_monitoring, _observer, _monitor_thread
    
    if not _is_monitoring:
        return
    
    try:
        # 设置停止标志
        _is_monitoring = False
        
        # 停止Observer
        if _observer:
            _observer.stop()
            # 等待Observer线程结束
            try:
                _observer.join(timeout=5.0)  # 最多等待5秒
            except Exception as e:
                logging.warning(f"[ComfyUI-1hewNodes] Observer停止时出现异常：{str(e)}")
            _observer = None
        
        # 等待监控线程结束
        if _monitor_thread and _monitor_thread.is_alive():
            try:
                _monitor_thread.join(timeout=3.0)  # 最多等待3秒
            except Exception as e:
                logging.warning(f"[ComfyUI-1hewNodes] 监控线程停止时出现异常：{str(e)}")
            _monitor_thread = None
        
    except Exception as e:
        logging.error(f"[ComfyUI-1hewNodes] 停止工作流监控时出错：{str(e)}")
        # 强制重置状态
        _is_monitoring = False
        _observer = None
        _monitor_thread = None

def is_monitoring():
    """检查监控是否正在运行"""
    return _is_monitoring

# 兼容性：保留原有的start_observer函数用于独立运行
def start_observer():
    """启动文件监控器（独立运行模式）"""
    # 验证并获取工作流目录
    is_valid, workflow_root = validate_workflow_dir()
    if not is_valid:
        return False
    
    event_handler = WorkflowFileHandler()
    observer = Observer()
    # 监控目标目录及所有子目录（recursive=True）
    observer.schedule(event_handler, workflow_root, recursive=True)
    
    try:
        observer.start()
        logging.info(f"监控已启动 | 目标目录：{workflow_root}")
        logging.info("按 Ctrl+C 停止监控...")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
        logging.info("监控已停止")
    except Exception as e:
        logging.error(f"监控启动失败：{str(e)}")
        observer.stop()
    observer.join()
    return True

if __name__ == "__main__":
    if not start_observer():
        # 目录无效时暂停，让用户看清错误
        input("按 Enter 退出...")