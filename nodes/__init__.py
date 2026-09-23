"""1hewNodesV3 节点包

将每个节点实现放在独立的模块文件中，文件名使用下划线风格，
类名使用驼峰风格并继承 io.ComfyNode。无需在此处显式注册，扩展
入口将自动扫描并注册所有节点类。

命名规范（唯一真源就在各节点文件的 io.Schema 里）：

- 模型名与版本号一律用空格分词，禁止连写：
  "Image Resize Gemini 3.1 Flash Image"（不是 "Gemini31FlashImage"）、
  "String Ratio GPT Image 2.0"（不是 "Gpt20Image"）。
- node_id 是工作流兼容键，发布后不再改动；display_name 只影响界面显示，
  可以随时调整。
- 改动 display_name 后必须同步 README 节点列表与 web/docs/<node_id>，
  运行 scripts/check_node_consistency.py 可校验三处是否一致。
"""

# 保持包的轻量与可读性，不在此处做任何副作用导入。