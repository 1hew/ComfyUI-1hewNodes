<div align="center">
<a href="./README.md"><img src="https://img.shields.io/badge/🇬🇧English-0b8cf5"></a>
<a href="./README.ZH_CN.md"><img src="https://img.shields.io/badge/🇨🇳中文简体-e9e9e9"></a>
</div>

# ComfyUI-1hewNodes

This is a custom node collection for ComfyUI that provides a series of practical image processing, mask operations, and utility nodes.



## 📦 Installation

Clone the repository into ComfyUI's `custom_nodes` directory:

```bash
git clone https://github.com/1hew/ComfyUI-1hewNodes
```



## 📜 Changelog

**v1.2.2**
- feat(image): Added the `Image BBox Overlay by Mask` node, which overlays the image bounding box based on a mask.

**v1.2.1**

- refactor(image/crop): Renamed node classes and updated related documentation
- feat(image_crop): Enhanced the functionality and output options of the `ImageCropByMaskAlpha` node

**v1.2.0**

- feat: Added `conversion`, and restructured image mixing and masking processing

<details>
<summary><b>v1.1.6</b></summary>

- feat(ImageEditStitch): Add a “spacing” parameter to control the distance between stitched images

​	</details>

<details>
<summary><b>v1.1.5</b></summary>

- feat: Added text processing and logic nodes, optimized existing node functions 
- refactor(util): Refactored utility nodes, renamed nodes `RangeMapping` and `PathBuild` 
- feat(logic): Added `ImageListAppend` node for image list merging
- feat(text): Added `TextCustomList` and `TextCustomExtract` text processing nodes
- style: Cleaned up node parameter labels to maintain simplicity and consistency

​	</details>

<details>
<summary><b>v1.1.2</b></summary>

- feat(image_tile): Improved the `Image Tile Merge` algorithm, using weight masks and cosine gradients to achieve perfect seamless stitching

​	</details>

<details>
<summary><b>v1.1.1</b></summary>

- feat (image_crop): Added intelligent batch processing for `Image BBox Paste`

​	</details>

<details>
<summary><b>v1.1.0</b></summary>

- build: Add new tile nodes
- feat: Update node functionality
- docs: Add bilingual documentation, improve node descriptions

​	</details>

<details>
<summary><b>v1.0.5</b></summary>

- Add Path Select

​	</details>

<details>
<summary><b>v1.0.4</b></summary>

- Fix Image Cropped Paste error, add batch processing feature.

​	</details>



## 📋 Node List

### 🖼️ Image Processing Nodes
| Node Name | Description |
|-----------|--------------|
| Image Solid | Generate solid color images with multiple size and color format support |
| Image Resize Universal | Universal image resizing with multiple algorithms and constraints |
| Image Edit Stitch | Image stitching and merging with multiple stitching modes |
| Image Detail HL Freq Separation | High-low frequency separation processing |
| Image Add Label | Add text labels to images |
| Image Plot | Image plotting and visualization tools |
| Image BBox Overlay by Mask | Mask-based image bounding box overlay with independent and merge modes |

### 🎨 Image Blending Nodes
| Node Name | Description |
|-----------|-------------|
| Image Luma Matte | Luminance-based image mask compositing |
| Image Blend Modes by Alpha | Alpha-based image blending with multiple Photoshop-style blend modes |
| Image Blend Modes by CSS | CSS standard blend modes based on Pilgram library |

### ✂️ Image Cropping Nodes
| Node Name | Description |
|-----------|-------------|
| Image Crop Square | Square cropping with mask guidance and scaling support |
| Image Crop Edge | Edge cropping with independent settings for four sides |
| Image Crop with BBox Mask | Smart cropping based on bounding boxes with aspect ratio control |
| Image Crop by Mask Alpha | Batch mask-based cropping with RGB/RGBA dual output modes and smart channel processing |
| Image Paste by BBox Mask | Paste cropped images back with multiple blend modes |

### 🧩 Image Tiling Nodes
| Node Name | Description |
|-----------|-------------|
| Image Tile Split | Image tile splitting with overlap and custom grid support |
| Image Tile Merge | Image tile merging with intelligent stitching |

### 🎭 Mask Operation Nodes
| Node Name | Description |
|-----------|-------------|
| Mask Math Ops | Mask mathematical operations (intersection, union, difference, XOR) |
| Mask Batch Math Ops | Batch mask mathematical operations |
| Mask Crop by BBox Mask | Mask bounding box cropping based on mask regions |

### 🔧 Utility Nodes
| Node Name | Description |
|-----------|-------------|
| Range Mapping | Value range mapping tool supporting linear transformation and precision control for slider values |
| Path Build | Path builder supporting preset paths and custom extensions |

### 🔄 Conversion Nodes
| Node Name | Description |
|-----------|-------------|
| Image Batch to List | Convert batch images to image lists for individual processing |
| Image List to Batch | Convert image lists to batch images with automatic size normalization |
| Mask Batch to List | Convert batch masks to mask lists for individual processing |
| Mask List to Batch | Convert mask lists to batch masks with automatic size normalization |
| String Coordinate to BBoxes | Convert string format coordinates to BBOXES format with multiple input format support |
| String Coordinate to BBox Mask | Convert string format coordinates to BBoxMask format with image dimension support and flexible output modes |

### 🧠 Logic Nodes
| Node Name | Description |
|-----------|-------------|
| Image List Append | Image list appender for intelligently merging images into lists |

### 📝 Text Processing Nodes
| Node Name | Description |
|-----------|-------------|
| Text Custom Extract | Text custom extractor for extracting specified key values from JSON |
| List Custom Int | Custom integer list generator with dash separator and multiple delimiter support |
| List Custom Float | Custom float list generator with dash separator and multiple delimiter support |
| List Custom String | Custom string list generator with dash separator and multiple delimiter support |



## 🙆 Acknowledgments

[ComfyUI](https://github.com/comfyanonymous/ComfyUI)

[ComfyUI-KJNodes](https://github.com/kijai/ComfyUI-KJNodes)

[ComfUI-Easy-Use](https://github.com/yolain/ComfyUI-Easy-Use)

[ComfyUI_LayerStyle](https://github.com/chflame163/ComfyUI_LayerStyle)

[Comfyui_TTP_Toolset](https://github.com/TTPlanetPig/Comfyui_TTP_Toolset)



## 🌟 Star

My gratitude extends to the generous souls who bestow a star.

[<img src="imgs/Stargazers.png" alt="Stargazers" style="zoom:80%;" />](https://github.com/1hew/ComfyUI-1hewNodes/stargazers)