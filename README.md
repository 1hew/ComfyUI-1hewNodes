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

**v1.1.2**

- feat(image_tile): Improved the `Image Tile Merge` algorithm, using weight masks and cosine gradients to achieve perfect seamless stitching.

**v1.1.1**

- feat (image_crop): Added intelligent batch processing for `Image BBox Paste`

**v1.1.0**

- build: Add new tile nodes
- feat: Update node functionality
- docs: Add bilingual documentation, improve node descriptions

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
|-----------|-------------|
| Image Solid | Generate solid color images with multiple size and color format support |
| Image Resize Universal | Universal image resizing with multiple algorithms and constraints |
| Image Edit Stitch | Image stitching and merging with multiple stitching modes |
| Image Detail HL Freq Separation | High-low frequency separation processing |
| Image Add Label | Add text labels to images |
| Image Plot | Image plotting and visualization tools |

### 🎨 Image Blending Nodes

| Node Name | Description |
|-----------|-------------|
| Image Luma Matte | Luminance-based image mask compositing |
| Image Blend Modes By Alpha | Alpha-based image blending with multiple Photoshop-style blend modes |
| Image Blend Modes By CSS | CSS standard blend modes based on Pilgram library |

### ✂️ Image Cropping Nodes

| Node Name | Description |
|-----------|-------------|
| Image Crop Square | Square cropping with mask guidance and scaling support |
| Image Crop Edge | Edge cropping with independent settings for four sides |
| Image Crop With BBox | Smart cropping based on bounding boxes |
| Image BBox Crop | Batch bounding box cropping |
| Image BBox Paste | Paste cropped images back with multiple blend modes |

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
| Mask BBox Crop | Mask bounding box cropping |

### 🔧 Utility Nodes

| Node Name | Description |
|-----------|-------------|
| Coordinate Extract | JSON coordinate data extractor |
| Slider Value Range Mapping | Value range mapping tool |
| Path Select | Path selector supporting files and directories |
| Prompt Extract | Prompt extraction and processing tool |



## 🙆 Acknowledgments

[ComfyUI](https://github.com/comfyanonymous/ComfyUI)

[ComfyUI-KJNodes](https://github.com/kijai/ComfyUI-KJNodes)

[ComfUI-Easy-Use](https://github.com/yolain/ComfyUI-Easy-Use)

[ComfyUI_LayerStyle](https://github.com/chflame163/ComfyUI_LayerStyle)



## 🌟 Star

My gratitude extends to the generous souls who bestow a star.

[<img src="imgs/Stargazers.png" alt="Stargazers" style="zoom:80%;" />](https://github.com/1hew/ComfyUI-1hewNodes/stargazers)
