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

**v1.1.0**

- build: 添加 tile 新节点

- feat: 更新节点功能
- docs: 添加中英文文档，完善节点说明

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
| Image Solid | 生成纯色图像，支持多种尺寸和颜色格式 |
| Image Resize Universal | 通用图像尺寸调整，支持多种算法和约束 |
| Image Edit Stitch | 图像拼接与缝合，支持多种拼接模式 |
| Image Detail HL Freq Separation | 高低频分离处理 |
| Image Add Label | 为图像添加文本标签 |
| Image Plot | 图像绘制和可视化工具 |

### 🎨 图像混合节点
| 节点名称 | 功能描述 |
|---------|----------|
| Image Luma Matte | 基于亮度的图像蒙版合成 |
| Image Blend Modes By Alpha | 基于透明度的图像混合，支持多种Photoshop风格混合模式 |
| Image Blend Modes By CSS | CSS标准混合模式，基于Pilgram库实现 |

### ✂️ 图像裁剪节点
| 节点名称 | 功能描述 |
|---------|----------|
| Image Crop Square | 方形裁剪，支持遮罩引导和缩放 |
| Image Crop Edge | 边缘裁剪，支持四边独立设置 |
| Image Crop With BBox | 基于边界框的智能裁剪 |
| Image BBox Crop | 批量边界框裁剪 |
| Image BBox Paste | 裁剪图像回贴，支持多种混合模式 |

### 🧩 图像分块节点
| 节点名称 | 功能描述 |
|---------|----------|
| Image Tile Split | 图像分块分割，支持重叠和自定义网格 |
| Image Tile Merge | 图像分块合并，智能拼接处理 |

### 🎭 遮罩操作节点
| 节点名称 | 功能描述 |
|---------|----------|
| Mask Math Ops | 遮罩数学运算（交集、并集、差集、异或） |
| Mask Batch Math Ops | 批量遮罩数学运算 |
| Mask BBox Crop | 遮罩边界框裁剪 |

### 🔧 工具节点
| 节点名称 | 功能描述 |
|---------|----------|
| Coordinate Extract | JSON坐标数据提取器 |
| Slider Value Range Mapping | 数值范围映射工具 |
| Path Select | 路径选择器，支持文件和目录选择 |
| Prompt Extract | 提示词提取和处理工具 |



## 🙆 致谢

[ComfyUI](https://github.com/comfyanonymous/ComfyUI)

[ComfyUI-KJNodes](https://github.com/kijai/ComfyUI-KJNodes)

[ComfUI-Easy-Use](https://github.com/yolain/ComfyUI-Easy-Use)

[ComfyUI_LayerStyle](https://github.com/chflame163/ComfyUI_LayerStyle)



## 🌟 星星

感谢慷慨点亮星星的人

[<img src="imgs/Stargazers.png" alt="Stargazers" style="zoom:80%;" />](https://github.com/1hew/ComfyUI-1hewNodes/stargazers)
