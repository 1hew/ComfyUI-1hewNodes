<div align="center">
<a href="./README.md"><img src="https://img.shields.io/badge/🇬🇧English-e9e9e9"></a>
<a href="./README.ZH_CN.md"><img src="https://img.shields.io/badge/🇨🇳中文简体-0b8cf5"></a>
</div>

# ComfyUI-1hewNodes

这是 ComfyUI 的自定义节点集合，提供了一系列实用的图像处理、遮罩操作和工具节点。



## 📦 安装

将仓库克隆到 ComfyUI 的 `custom_nodes` 目录中：

```bash
git clone https://github.com/1hew/ComfyUI-1hewNodes
```



## 📜 更新日志

**v3.2.6**
- refactor(condition): 优化 `Text Encode QwenImageEdit` 节点

**v3.2.3**
- refactor(io): 优化 IO 组

**v3.2.1**
- feat(io): 添加视频加载条

**v3.2.0**
- refactor(io): 重构 IO 组

**v3.1.0**
- feat(io): 添加 `Save Video by Image` 节点，用于将图像序列编码保存为视频
- refactor(color): 优化 `Match Brightness Contrast` 节点一致性选项

**v3.0.8**
- refactor(image_crop): 优化 `Image Crop With BBox Mask` 节点
- refactor(color): 优化 `Match Brightness Contrast` 节点

**v3.0.7**
- feat(color): 添加 `Match Brightness Contrast` 节点

**v3.0.6**
- feat(mask): 添加 `Mask Repeat` 节点，支持遮罩批量重复与反转
- feat(io): 添加 `Get File Count`、`Load Image From Folder`、`Load Video From Folder` 节点，优化文件加载逻辑与稳定性

**v3.0.5**
- feat(image_tile): 合并 `Image Tile Split` 和 `Image Tile Split Preset` 节点为统一的 `Image Tile Split` 节点

**v3.0.2**
- feat(text): 添加 `String Filter` 节点
- feat(text): 添加 `String Join Multi` 节点
- feat(conversion): 添加 `Text List to String` 节点

**v3.0.1**
- feat(image): 添加 `Image PingPong` 节点，用于批量往返帧生成，支持预反转与拼接处去重、帧数截取
- feat(audio): 添加 `Audio Duration` 节点，用于获取音频时长（秒）

**v3.0.0**
- build: 版本升级至 3.0.0

<details>
<summary><b>2.x 版本更新日志</b></summary>

**v2.0.5**
- feat(multi): 新增 `Multi String Join`、`Multi Image Batch`、`Multi Mask Batch`、`Multi Image Stitch`
- feat(image): 新增 `Image Three Stitch` 节点
- feat(condition): 新增 `Text Encode QwenImageEdit Keep Size` 节点

**v2.0.3**
- feat(image_blend): 为 `Image Mask Blend` 节点新增 `output_mask_invert` 参数

**v2.0.0**
- 重要变更：重大版本更新，升级到 2.0.0 后，已有 1.x 工作流需重新设置节点与参数方可正常运行，请谨慎更新。

</details>

<details>
<summary><b>1.x 版本更新日志</b></summary>

**v1.2.46**
- feat(detection): 添加 `DetectGuideLine` 节点

**v1.2.45**
- feat(detection): 添加 `DetectYolo` 节点，支持YOLO模型目标检测

**v1.2.44**
- feat(util): 添加 `Workflow Name` 节点

**v1.2.43**
- feat(logic): 添加 `Video Cut Group` 节点

**v1.2.42**
- feat(logic): 添加 `Image Batch Extract` 节点，支持从批量图像中提取特定图像，提供多种模式包括自定义索引、步长间隔和均匀分布

**v1.2.40**
- feat(text): 添加 `IntWan` 节点，支持生成4n+1等差数列序列，具有可配置步长控制和范围验证功能
- refactor(logic): 增强 `Image Batch Split` 和 `Mask Batch Split` 节点，改进边界条件处理和全面错误恢复机制

<details>
<summary><b>v1.2.39</b></summary>

- feat(logic): 添加 `Image Batch Group` 节点，支持智能图像批次分组，具有重叠处理和灵活填充策略

​	</details>

<details>
<summary><b>v1.2.38</b></summary>

- refactor(image): 增强 `ImageResizeUniversal` 节点，完善 mask 处理逻辑

​	</details>

<details>
<summary><b>v1.2.37</b></summary>

- feat(image): 增强 `Image Solid` 节点，新增高级颜色参数
- feat(image): 添加 `ImageGridSplit` 节点，支持将图像分割为网格布局，提供灵活的输出选项

​	</details>

<details>
<summary><b>v1.2.36</b></summary>

- feat(conversion): 增强 `URL to Video` 节点

​	</details>

<details>
<summary><b>v1.2.35</b></summary>

- feat(image): 添加 `Image Resize Qwen Image` 节点，专为 Qwen 视觉模型优化的图像缩放器，提供 7 种预设分辨率和自动宽高比选择

​	</details>

<details>
<summary><b>v1.2.32</b></summary>

- feat(image): 添加 `Image Solid Flux Kontext` 节点，支持基于 Flux Kontext 尺寸预设生成纯色图像
- feat(image): 添加 `Image Solid Qwen Image` 节点，支持基于 QwenImage 尺寸预设生成纯色图像

​	</details>

<details>
<summary><b>v1.2.31</b></summary>

- fix: 修复相关bug，提升稳定性

​	</details>

<details>
<summary><b>v1.2.28</b></summary>

- feat(mask): 添加 `Mask Paste by BBox Mask` 节点，支持简化遮罩粘贴，具有自动基础遮罩创建和边界框检测功能
- feat(image_tile): 添加 `Image Tile Split Preset` 节点，提供预定义分辨率预设和智能瓦片尺寸选择
- feat(image): 添加 `Image Rotate with Mask` 节点，支持高级图像旋转，具有遮罩支持和多种填充模式
- feat(text): 增强 `Text Load Local` 节点，新增 `user_prompt` 参数，支持将JSON内容与额外用户提示词结合

​	</details>

<details>
<summary><b>v1.2.26</b></summary>

- feat(image_crop): 增强 `Image Crop with BBox Mask` 节点，添加精确尺寸控制功能，新增 `crop_to_side` 和 `crop_to_length` 参数

​	</details>

<details>
<summary><b>v1.2.25</b></summary>

- feat(image_crop): 为 `Image Paste by BBox Mask` 节点添加 `apply_paste_mask` 参数，用于控制智能缩放行为

​	</details>

<details>
<summary><b>v1.2.24</b></summary>

- feat(image_crop): 为 `Image Paste by BBox Mask` 节点添加 `opacity` 透明度参数，用于控制粘贴图像的透明度
- feat(image): 增强 `Image Stroke by Mask` 节点，添加批处理支持，可处理多个图像和遮罩

​	</details>

<details>
<summary><b>v1.2.23</b></summary>

- fix(image): 增强 `Image Stroke by Mask` 节点颜色解析逻辑，支持RGB字符串格式并改进默认兜底为白色
- fix(image): 增强 `Image Paste by BBox Mask` 节点旋转参数

​	</details>

<details>
<summary><b>v1.2.21</b></summary>

- feat(text): 添加 `Text Filter Comment` 节点，支持过滤单行注释（#开头）和多行注释（三引号包裹），保留非注释空行
- feat(text): 添加 `Text Join by Text List` 节点，支持将任意类型列表合并为字符串，支持前缀、后缀和自定义分隔符
- refactor(text): 重构 `Text Format` 节点为 `Text Prefix Suffix`，优化通配符输入处理和格式化功能

​	</details>

<details>
<summary><b>v1.2.18</b></summary>

- feat(sample): 添加 `Step Split` 节点，用于高低频采样步数分离，支持百分比和整数输入模式

​	</details>

<details>
<summary><b>v1.2.17</b></summary>

- feat(image_crop): 优化 `Image Crop with BBox Mask` 节点

​	</details>

<details>
<summary><b>v1.2.15</b></summary>

- feat(text): 添加 `Text Join Multi` 节点，支持多个文本输入的连接和动态变量引用
- feat(image_crop): 添加 `Image Edge Crop Pad` 节点，支持智能边缘裁剪和填充，提供遮罩输出功能
- feat(image_blend): 增强 `Image Luma Matte` 节点，新增羽化和透明度输出功能，支持多种颜色格式和边缘处理

​	</details>

<details>
<summary><b>v1.2.13</b></summary>

- feat(text): 添加 `Text Load Local` 节点，支持从prompt目录加载JSON格式提示词文件，提供中英文双语输出

​	</details>

<details>
<summary><b>v1.2.12</b></summary>

- feat(text): 添加 `Text Format` 节点，支持通配符输入的灵活文本格式化功能

​	</details>

<details>
<summary><b>v1.2.9</b></summary>

- feat(image_crop): 重构 Image Crop with BBox Mask 节点

​	</details>

<details>
<summary><b>v1.2.8</b></summary>

- feat(image): 添加 `Image Resize Flux Kontext` 节点，支持图像和遮罩的尺寸自动选择和手动选择
- feat(image): 优化 `Image Edit Stitch` 节点图像拼接算法和参数处理

​	</details>

<details>
<summary><b>v1.2.7</b></summary>

- feat(text): 添加 `List Custom Seed` 节点，支持生成唯一随机种子列表和control after generate功能

​	</details>

<details>
<summary><b>v1.2.6</b></summary>

- feat(image_hlfreq): 添加高低频分离节点组，包含 `Image HLFreq Separate`、`Image HLFreq Combine` 和 `Image HLFreq Transform` 三个节点，支持RGB、HSV、IGBI三种频率分离方法

​	</details>

<details>
<summary><b>v1.2.5</b></summary>

- feat(mask): 添加 `Mask Fill Hole` 节点，用于填充遮罩中的封闭区域孔洞，支持批量处理。

​	</details>

<details>
<summary><b>v1.2.3</b></summary>

- fix(image_blend): 修复 `Image Blend Modes by Alpha` 节点设备不一致的问题

​	</details>

<details>
<summary><b>v1.2.2</b></summary>

- feat(image): 添加 `Image BBox Overlay by Mask` 节点，基于遮罩的图像边界框叠加

​	</details>

<details>
<summary><b>v1.2.1</b></summary>

- refactor(image/crop): 重命名节点类并更新相关文档
- feat(image_crop): 增强 `ImageCropByMaskAlpha` 节点的功能和输出选项

​	</details>

<details>
<summary><b>v1.2.0</b></summary>

- feat: 新增 `conversion` ，重构图像混合与遮罩处理

​	</details>

<details>
<summary><b>v1.1.6</b></summary>

- feat(ImageEditStitch): 添加 spacing 参数控制拼接图像间的间距

​	</details>

<details>
<summary><b>v1.1.5</b></summary>

- feat: 新增文本处理和逻辑节点，优化现有节点功能
- refactor(util): 重构工具节点，重命名节点 `RangeMapping` 和 `PathBuild`
- feat(logic): 新增 `ImageListAppend` 节点用于图像列表合并
- feat(text): 新增 `TextCustomList` 和 `TextCustomExtract` 文本处理节点
- style: 清理节点参数标签，保持简洁统一

​	</details>

<details>
<summary><b>v1.1.2</b></summary>

- feat(image_tile): 改进 `Image Tile Merge`算法，使用权重蒙版和余弦渐变实现完美无缝拼接

​	</details>

<details>
<summary><b>v1.1.1</b></summary>

- feat(image_crop): 为 `Image BBox Paste` 添加智能批次处理功能

​	</details>

<details>
<summary><b>v1.1.0</b></summary>

- build: 添加 tile 新节点
- feat: 更新节点功能
- docs: 添加中英文文档，完善节点说明

​	</details>

<details>
<summary><b>v1.0.5</b></summary>

- 添加 `Path Select` 

​	</details>

<details>
<summary><b>v1.0.4</b></summary>

- 修复 `Image Cropped Paste` 错误，添加批处理功能。

​	</details>

</details>



## 📋 节点列表

### 🖼️ 图像处理节点
| 节点名称 | 功能描述 |
|---------|----------|
| Image Solid | 生成纯色图像，增强颜色参数支持多种输入格式和多种尺寸预设 |
| Image Resize FluxKontext | 图像尺寸调整为FluxKontext尺寸，支持图像和遮罩的尺寸自动选择和手动选择 |
| Image Resize Qwen Image | 专为 Qwen 视觉模型优化的图像尺寸调整，提供 7 种预设分辨率和自动宽高比选择 |
| Image Resize Universal | 通用图像尺寸调整，支持多种算法和约束 |
| Image Rotate with Mask | 高级图像旋转，支持遮罩集成、多种填充模式和遮罩中心旋转选项 |
| Image Edit Stitch | 图像拼接与缝合，支持多种拼接模式 |
| ImageMainStitch | 主画面拼接，支持动态 `image_2..image_N` 与方向/尺寸匹配/间距/填充 |
| Image Add Label | 为图像添加文本标签 |
| Image Plot | 图像绘制和可视化工具 |
| Image Stroke by Mask | 对遮罩区域应用描边效果，支持自定义宽度和颜色 |
| Image BBox Overlay by Mask | 基于遮罩的图像边界框叠加，支持独立和合并模式 |

### 🌈 颜色节点
| 节点名称 | 功能描述 |
|---------|----------|
| Match Brightness Contrast | 调整源图像的亮度和对比度以匹配参考图像 |

### 🎨 图像混合节点
| 节点名称 | 功能描述 |
|---------|----------|
| Image Mask Blend | 基于亮度的图像蒙版合成，支持羽化、透明度输出和多种颜色格式 |
| Image Blend Mode by Alpha | 基于透明度的图像混合，支持多种Photoshop风格混合模式 |
| Image Blend Mode by CSS | CSS标准混合模式，基于Pilgram库实现 |

### ✂️ 图像裁剪节点
| 节点名称 | 功能描述 |
|---------|----------|
| Image Mask Crop | 基于遮罩的批量裁剪，支持RGB/RGBA双输出模式和智能通道处理 |
| Image Crop Square | 方形裁剪，支持遮罩引导和缩放 |
| Image Crop with BBox Mask| 智能边界框裁剪，支持精确比例控制和缩放强度调节 |
| Image Paste by BBox Mask | 裁剪图像回贴，支持多种混合模式 |
| Image Edge Crop Pad | 智能边缘裁剪和填充，支持多种填充模式和遮罩输出 |
| Image Grid Split | 将图像分割为网格布局，支持灵活的行列配置和选择性输出选项 |

### 🧩 图像分块节点
| 节点名称 | 功能描述 |
|---------|----------|
| Image Tile Split | 图像分块分割，支持自动/网格/预设模式、重叠处理及参考图尺寸分割 |
| Image Tile Merge | 图像分块合并，智能拼接处理 |

### 🌊 高低频分离节点
| 节点名称 | 功能描述 |
|---------|----------|
| Image HLFreq Separate | 高级频率分离节点，支持RGB、HSV、IGBI三种分离方法，提供精确的高低频图像分离和自动重组功能 |
| Image HLFreq Combine | 高级频率重组节点，支持RGB、HSV、IGBI三种重组模式，提供强度调整和批处理智能匹配 |
| Image HLFreq Transform | 高级细节迁移节点，支持IGBI、RGB、HSV三种迁移方法，实现从细节图像向生成图像的精确纹理细节迁移 |

### 🎭 遮罩操作节点
| 节点名称 | 功能描述 |
|---------|----------|
| Mask Fill Hole | 填充遮罩中的封闭区域孔洞，支持批量处理 |
| Mask Crop by BBox Mask | 基于蒙版区域的遮罩边界框裁剪 |
| Mask Paste by BBox Mask | 简化遮罩粘贴，支持自动基础遮罩创建和边界框检测 |
| Mask Repeat | 批量重复遮罩，支持反转功能 |

### 🔍 检测节点
| 节点名称 | 功能描述 |
|---------|----------|
| Detect Guide Line | 引导线检测，融合 Canny、HoughLinesP 与 DBSCAN 消失点聚类 |
| Detect Yolo | YOLO模型目标检测，支持子文件夹模型、可自定义置信度阈值和可选标签显示控制 |

### 🔧 工具节点
| 节点名称 | 功能描述 |
|---------|----------|
| Workflow Name | 自动获取当前工作流文件名，支持路径控制、自定义前缀后缀和日期格式化 |
| Range Mapping | 数值范围映射工具，支持滑块值的线性变换和精度控制 |

### 🔄 转换节点
| 节点名称 | 功能描述 |
|---------|----------|
| URL to Video | 将视频URL转换为ComfyUI VIDEO对象，改进错误处理、超时控制，支持同步和异步下载方法 |
| Image Batch to List | 将批量图像转换为图像列表，用于单独处理 |
| Image List to Batch | 将图像列表转换为批量图像，自动进行尺寸标准化 |
| Mask Batch to List | 将批量遮罩转换为遮罩列表，用于单独处理 |
| Mask List to Batch | 将遮罩列表转换为批量遮罩，自动进行尺寸标准化 |
| String Coordinate to BBoxes | 将字符串格式坐标转换为BBOXES格式，增强格式支持并改进SAM2兼容性 |
| String Coordinate to BBox Mask | 将字符串格式坐标转换为BBoxMask格式，支持图像尺寸获取和灵活的输出模式 |
| Text List to String | 文本列表合并，逐项应用前后缀并按分隔符拼接，支持转义与复合分隔符 |

### 🧠 逻辑节点
| 节点名称 | 功能描述 |
|---------|----------|
| Any Empty Bool | 通用空值检查节点（布尔输出版本），检查任意类型输入是否为空，返回布尔值 |
| Any Empty Int | 通用空值检查节点（整数输出版本），检查任意类型输入是否为空，返回自定义的整数值 |
| Any Switch Bool | 通用布尔切换节点，支持任意类型输入和惰性求值，根据布尔值条件选择输出 |
| Any Switch Int | 多路整数切换节点，支持多个输入选项的切换，根据整数索引（1-5）选择对应的输入输出 |

### 🔢 整数节点
| 节点名称 | 功能描述 |
|---------|----------|
| Int Image Side Length | 基于图像尺寸输出选定边长（最长/最短/宽/高） |
| Int Image Size | 输出图像宽度与高度两个整数 |
| Int Mask Side Length | 基于遮罩尺寸输出选定边长（最长/最短/宽/高） |
| Int Split | 将总数值分割为两部分，支持百分比与整数分割点 |
| Int Wan | 生成 4n+1 等差数列，支持步长与范围校验 |

### 📦 批处理节点
| 节点名称 | 功能描述 |
|---------|----------|
| Image Batch Extract | 智能图像批次提取器，支持多种提取模式包括自定义索引、步长间隔和均匀分布 |
| Image Batch Split | 智能图像批次拆分器，支持正向/反向拆分模式和增强的边界条件处理 |
| Image Batch Group | 智能图像批次分组器，支持可配置的批次大小、重叠处理和灵活的填充策略 |
| Image Batch Range | 从图像批次选择连续范围，支持起始索引与数量，越界安全 |
| Image PingPong | 批量往返帧生成，支持预反转、拼接处去重与帧数截取 |
| Image List Append | 图像列表追加器，智能合并图像到列表中 |
| Mask Batch Math Ops | 批量遮罩数学运算 |
| Mask Batch Range | 从遮罩批次选择连续范围，支持起始索引与数量，越界安全 |
| Mask Batch Split | 智能遮罩批次拆分器，支持正向/反向拆分模式和增强的边界条件处理 |
| Video Cut Group | 视频硬切检测器，通过分析相邻帧相似度识别场景切换点，支持快速和精确模式 |

### 📝 文本处理节点
| 节点名称 | 功能描述 |
|---------|----------|
| Text Prefix Suffix | 文本前缀后缀器，支持通配符输入的灵活数据格式化，可自定义前缀和后缀 |
| Text Custom Extract | 文本自定义提取器，从JSON中提取指定键值 |
| String Filter | 文本过滤器，支持 `{input}` 替换、注释过滤（# 与三引号）、可选空行移除 |
| String Join Multi | 多段文本拼接，支持 `{input}` 占位符、注释/空行过滤与复合分隔符 |
| List Custom Int | 自定义整数列表生成器，支持连字符分割和多种分隔符 |
| List Custom Float | 自定义浮点数列表生成器，支持连字符分割和多种分隔符 |
| List Custom String | 自定义字符串列表生成器，支持连字符分割和多种分隔符 |
| List Custom Seed | 自定义种子列表生成器，支持生成唯一随机种子列表和control after generate功能 |

### 🔗 多输入节点
| 节点名称 | 功能描述 |
|---------|----------|
| Multi String Join | 动态多输入字符串连接，支持 `{input}` 变量与注释/三引号过滤，可自定义分隔符 |
| Multi Image Batch | 从动态 `image_X` 构建批次，支持 crop/pad/stretch 尺寸统一与边缘/颜色填充 |
| Multi Image Stitch | 动态多图像拼接，支持方向、尺寸匹配、间距与填充颜色 |
| Multi Mask Batch | 从动态 `mask_X` 构建批次，支持 crop/pad/stretch 尺寸统一与灰度填充 |
| Multi Mask Math Ops | 动态多遮罩运算（交/并/差/异或），支持批次广播与统一尺寸 |


### 📁 IO 节点
| 节点名称 | 功能描述 |
|---------|----------|
| Get File Count | 统计目录中图片或视频文件数量，支持递归扫描 |
| Load Image | 从文件/目录加载图片，支持批量加载、尺寸统一与遮罩输出 |
| Load Video | 从文件/目录选择视频并输出 VIDEO 对象，解码阶段应用裁切与 FPS 设置 |
| Load Video to Image | 将视频解码为图像帧批次、音频、fps 与帧数信息 |
| Save Image | 保存图像批次到输出/临时目录，并输出保存后的绝对路径 |
| Save Video by Image | 将图像批次编码为视频，支持可选音频混流与 Alpha 输出策略 |
| Save Video | 保存 VIDEO 对象并返回路径，沿用容器扩展名并生成 Alpha 预览 |


### 🎛️ 条件编码节点
| 节点名称 | 功能描述 |
|---------|----------|
| Text Encode QwenImageEdit | Qwen 图文编辑条件编码，支持多图视觉编码、尺寸保持策略与参考潜空间 |


### 🔊 音频节点
| 节点名称 | 功能描述 |
|---------|----------|
| Audio Duration | 获取音频时长（秒）|



## 🙆 致谢

[ComfyUI](https://github.com/comfyanonymous/ComfyUI)

[ComfyUI-KJNodes](https://github.com/kijai/ComfyUI-KJNodes)

[ComfUI-Easy-Use](https://github.com/yolain/ComfyUI-Easy-Use)

[ComfyUI_LayerStyle](https://github.com/chflame163/ComfyUI_LayerStyle)

[Comfyui_TTP_Toolset](https://github.com/TTPlanetPig/Comfyui_TTP_Toolset)

[comfyUI_FrequencySeparation_RGB-HSV](https://github.com/risunobushi/comfyUI_FrequencySeparation_RGB-HSV)

[comfyui_extractstoryboards](https://github.com/gitadmini/comfyui_extractstoryboards)

[ComfyUI-VideoHelperSuite](https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite)



## 🌟 星星

感谢慷慨点亮星星的人

[<img src="imgs/Stargazers.png" alt="Stargazers" style="zoom:80%;" />](https://github.com/1hew/ComfyUI-1hewNodes/stargazers)
