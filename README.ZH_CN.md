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



## 📜 更新

**v1.2.38**
- refactor(image): 增强 `ImageResizeUniversal` 节点，完善 mask 处理逻辑

**v1.2.37**
- feat(image): 增强 `Image Solid` 节点，新增高级颜色参数
- feat(image): 添加 `ImageGridSplit` 节点，支持将图像分割为网格布局，提供灵活的输出选项

**v1.2.36**
- feat(conversion): 增强 `URL to Video` 节点

**v1.2.35**
- feat(image): 添加 `Image Resize Qwen Image` 节点，专为 Qwen 视觉模型优化的图像缩放器，提供 7 种预设分辨率和自动宽高比选择

**v1.2.32**
- feat(image): 添加 `Image Solid Flux Kontext` 节点，支持基于 Flux Kontext 尺寸预设生成纯色图像
- feat(image): 添加 `Image Solid Qwen Image` 节点，支持基于 QwenImage 尺寸预设生成纯色图像

**v1.2.31**
- fix: 修复相关bug，提升稳定性

**v1.2.28**
- feat(mask): 添加 `Mask Paste by BBox Mask` 节点，支持简化遮罩粘贴，具有自动基础遮罩创建和边界框检测功能
- feat(image_tile): 添加 `Image Tile Split Preset` 节点，提供预定义分辨率预设和智能瓦片尺寸选择
- feat(image): 添加 `Image Rotate with Mask` 节点，支持高级图像旋转，具有遮罩支持和多种填充模式
- feat(text): 增强 `Text Load Local` 节点，新增 `user_prompt` 参数，支持将JSON内容与额外用户提示词结合

**v1.2.26**
- feat(image_crop): 增强 `Image Crop with BBox Mask` 节点，添加精确尺寸控制功能，新增 `crop_to_side` 和 `crop_to_length` 参数

**v1.2.25**
- feat(image_crop): 为 `Image Paste by BBox Mask` 节点添加 `apply_paste_mask` 参数，用于控制智能缩放行为

**v1.2.24**
- feat(image_crop): 为 `Image Paste by BBox Mask` 节点添加 `opacity` 透明度参数，用于控制粘贴图像的透明度
- feat(image): 增强 `Image Stroke by Mask` 节点，添加批处理支持，可处理多个图像和遮罩

**v1.2.23**
- fix(image): 增强 `Image Stroke by Mask` 节点颜色解析逻辑，支持RGB字符串格式并改进默认兜底为白色
- fix(image): 增强 `Image Paste by BBox Mask` 节点旋转参数

**v1.2.21**
- feat(text): 添加 `Text Filter Comment` 节点，支持过滤单行注释（#开头）和多行注释（三引号包裹），保留非注释空行
- feat(text): 添加 `Text Join by Text List` 节点，支持将任意类型列表合并为字符串，支持前缀、后缀和自定义分隔符
- refactor(text): 重构 `Text Format` 节点为 `Text Prefix Suffix`，优化通配符输入处理和格式化功能

**v1.2.18**
- feat(sample): 添加 `Step Split` 节点，用于高低频采样步数分离，支持百分比和整数输入模式

**v1.2.17**
- feat(image_crop): 优化 `Image Crop with BBox Mask` 节点

**v1.2.15**
- feat(text): 添加 `Text Join Multi` 节点，支持多个文本输入的连接和动态变量引用
- feat(image_crop): 添加 `Image Edge Crop Pad` 节点，支持智能边缘裁剪和填充，提供遮罩输出功能
- feat(image_blend): 增强 `Image Luma Matte` 节点，新增羽化和透明度输出功能，支持多种颜色格式和边缘处理

**v1.2.13**
- feat(text): 添加 `Text Load Local` 节点，支持从prompt目录加载JSON格式提示词文件，提供中英文双语输出

**v1.2.12**
- feat(text): 添加 `Text Format` 节点，支持通配符输入的灵活文本格式化功能

**v1.2.9**
 - feat(image_crop): 重构 Image Crop with BBox Mask 节点

**v1.2.8**
- feat(image): 添加 `Image Resize Flux Kontext` 节点，支持图像和遮罩的尺寸自动选择和手动选择
- feat(image): 优化 `Image Edit Stitch` 节点图像拼接算法和参数处理

**v1.2.7**
- feat(text): 添加 `List Custom Seed` 节点，支持生成唯一随机种子列表和control after generate功能

**v1.2.6**
- feat(image_hlfreq): 添加高低频分离节点组，包含 `Image HLFreq Separate`、`Image HLFreq Combine` 和 `Image HLFreq Transform` 三个节点，支持RGB、HSV、IGBI三种频率分离方法

**v1.2.5**
- feat(mask): 添加 `Mask Fill Hole` 节点，用于填充遮罩中的封闭区域孔洞，支持批量处理。

**v1.2.3**
- fix(image_blend): 修复 `Image Blend Modes by Alpha` 节点设备不一致的问题

**v1.2.2**
- feat(image): 添加 `Image BBox Overlay by Mask` 节点，基于遮罩的图像边界框叠加

**v1.2.1**

- refactor(image/crop): 重命名节点类并更新相关文档
- feat(image_crop): 增强 `ImageCropByMaskAlpha` 节点的功能和输出选项

**v1.2.0**

- feat: 新增 `conversion` ，重构图像混合与遮罩处理

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



## 📋 节点列表

### 🖼️ 图像处理节点
| 节点名称 | 功能描述 |
|---------|----------|
| Image Solid | 生成纯色图像，增强颜色参数支持多种输入格式（灰度值、RGB元组、十六进制颜色、颜色名称、单字母快捷方式）和多种尺寸预设 |
| Image Solid FluxKontext | 基于 Flux Kontext 尺寸预设生成纯色图像，支持灵活的颜色输入格式 |
| Image Solid QwenImage | 基于 QwenImage 尺寸预设生成纯色图像，支持灵活的颜色输入格式 |
| Image Resize Universal | 通用图像尺寸调整，支持多种算法和约束 |
| Image Resize FluxKontext | 图像尺寸调整为FluxKontext尺寸，支持图像和遮罩的尺寸自动选择和手动选择 |
| Image Resize Qwen Image | 专为 Qwen 视觉模型优化的图像尺寸调整，提供 7 种预设分辨率和自动宽高比选择 |
| Image Edit Stitch | 图像拼接与缝合，支持多种拼接模式 |
| Image Add Label | 为图像添加文本标签 |
| Image Plot | 图像绘制和可视化工具 |
| Image Stroke by Mask | 对遮罩区域应用描边效果，支持自定义宽度和颜色 |
| Image BBox Overlay by Mask | 基于遮罩的图像边界框叠加，支持独立和合并模式 |
| Image Rotate with Mask | 高级图像旋转，支持遮罩集成、多种填充模式和遮罩中心旋转选项 |
| Image Grid Split | 将图像分割为网格布局，支持灵活的行列配置和选择性输出选项 |

### 🎨 图像混合节点
| 节点名称 | 功能描述 |
|---------|----------|
| Image Luma Matte | 基于亮度的图像蒙版合成，支持羽化、透明度输出和多种颜色格式 |
| Image Blend Modes by Alpha | 基于透明度的图像混合，支持多种Photoshop风格混合模式 |
| Image Blend Modes by CSS | CSS标准混合模式，基于Pilgram库实现 |

### ✂️ 图像裁剪节点
| 节点名称 | 功能描述 |
|---------|----------|
| Image Crop Square | 方形裁剪，支持遮罩引导和缩放 |
| Image Crop with BBox Mask| 智能边界框裁剪，支持精确比例控制和缩放强度调节 |
| Image Crop by Mask Alpha | 基于遮罩的批量裁剪，支持RGB/RGBA双输出模式和智能通道处理 |
| Image Paste by BBox Mask | 裁剪图像回贴，支持多种混合模式 |
| Image Edge Crop Pad | 智能边缘裁剪和填充，支持多种填充模式和遮罩输出 |

### 🧩 图像分块节点
| 节点名称 | 功能描述 |
|---------|----------|
| Image Tile Split | 图像分块分割，支持重叠和自定义网格 |
| Image Tile Split Preset | 图像分块分割，提供预定义分辨率预设和智能瓦片尺寸选择 |
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
| Mask Math Ops | 遮罩数学运算（交集、并集、差集、异或） |
| Mask Batch Math Ops | 批量遮罩数学运算 |
| Mask Crop by BBox Mask | 基于蒙版区域的遮罩边界框裁剪 |
| Mask Paste by BBox Mask | 简化遮罩粘贴，支持自动基础遮罩创建和边界框检测 |

### 🔧 工具节点
| 节点名称 | 功能描述 |
|---------|----------|
| Image Get Size | 图像尺寸获取器，从输入图像中提取宽度和高度信息，支持自动批处理 |
| Step Split | 高低频采样步数分离器，支持百分比(0.0-1.0)和整数输入模式，用于精确采样控制 |
| Range Mapping | 数值范围映射工具，支持滑块值的线性变换和精度控制 |
| Path Build | 路径构建器，支持预设路径和自定义扩展 |

### 🔄 转换节点
| 节点名称 | 功能描述 |
|---------|----------|
| Image Batch to List | 将批量图像转换为图像列表，用于单独处理 |
| Image List to Batch | 将图像列表转换为批量图像，自动进行尺寸标准化 |
| Mask Batch to List | 将批量遮罩转换为遮罩列表，用于单独处理 |
| Mask List to Batch | 将遮罩列表转换为批量遮罩，自动进行尺寸标准化 |
| String Coordinate to BBoxes | 将字符串格式坐标转换为BBOXES格式，增强格式支持并改进SAM2兼容性 |
| String Coordinate to BBox Mask | 将字符串格式坐标转换为BBoxMask格式，支持图像尺寸获取和灵活的输出模式 |
| URL to Video | 将视频URL转换为ComfyUI VIDEO对象，改进错误处理、超时控制，支持同步和异步下载方法 |

### 🧠 逻辑节点
| 节点名称 | 功能描述 |
|---------|----------|
| Image List Append | 图像列表追加器，智能合并图像到列表中 |

### 📝 文本处理节点
| 节点名称 | 功能描述 |
|---------|----------|
| Text Filter Comment | 文本注释过滤器，支持过滤单行注释（#开头）和多行注释（三引号包裹），保留非注释空行 |
| Text Join Multi | 文本连接多输入器，支持多个文本输入的连接和动态变量引用，可自定义分隔符 |
| Text Join by Text List | 文本列表连接器，支持将任意类型列表合并为字符串，支持前缀、后缀和自定义分隔符 |
| Text Prefix Suffix | 文本前缀后缀器，支持通配符输入的灵活数据格式化，可自定义前缀、后缀和分隔符 |
| Text Custom Extract | 文本自定义提取器，从JSON中提取指定键值 |
| List Custom Int | 自定义整数列表生成器，支持连字符分割和多种分隔符 |
| List Custom Float | 自定义浮点数列表生成器，支持连字符分割和多种分隔符 |
| List Custom String | 自定义字符串列表生成器，支持连字符分割和多种分隔符 |
| List Custom Seed | 自定义种子列表生成器，支持生成唯一随机种子列表和control after generate功能 |



## 🙆 致谢

[ComfyUI](https://github.com/comfyanonymous/ComfyUI)

[ComfyUI-KJNodes](https://github.com/kijai/ComfyUI-KJNodes)

[ComfUI-Easy-Use](https://github.com/yolain/ComfyUI-Easy-Use)

[ComfyUI_LayerStyle](https://github.com/chflame163/ComfyUI_LayerStyle)

[Comfyui_TTP_Toolset](https://github.com/TTPlanetPig/Comfyui_TTP_Toolset)

[comfyUI_FrequencySeparation_RGB-HSV](https://github.com/risunobushi/comfyUI_FrequencySeparation_RGB-HSV)



## 🌟 星星

感谢慷慨点亮星星的人

[<img src="imgs/Stargazers.png" alt="Stargazers" style="zoom:80%;" />](https://github.com/1hew/ComfyUI-1hewNodes/stargazers)
