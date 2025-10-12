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

**v1.2.38**
- refactor(image): Enhanced `ImageResizeUniversal` node with comprehensive mask processing logic

**v1.2.37**
- feat(image): Enhanced `Image Solid` node with advanced color parameter
- feat(image): Added the `ImageGridSplit` node for splitting images into grid layouts with flexible output options

**v1.2.36**
- feat(conversion): Enhanced `URL to Video` node

**v1.2.35**
- feat(image): Added `Image Resize Qwen Image` node for Qwen vision model optimized image resizing with 7 preset resolutions and automatic aspect ratio selection

**v1.2.32**
- feat(image): Added `Image Solid Flux Kontext` node for generating solid color images with Flux Kontext dimension presets
- feat(image): Added `Image Solid Qwen Image` node for generating solid color images with QwenImage dimension presets

**v1.2.31**
- fix: Fixed various bugs and improved stability

**v1.2.28**
- feat(mask): Added `Mask Paste by BBox Mask` node for simplified mask pasting with automatic base mask creation and bounding box detection
- feat(image_tile): Added `Image Tile Split Preset` node with predefined resolution presets and intelligent tile size selection
- feat(image): Added `Image Rotate with Mask` node for advanced image rotation with mask support and multiple fill modes
- feat(text): Enhanced `Text Load Local` node with `user_prompt` parameter for combining JSON content with additional user prompts

**v1.2.26**
- feat(image_crop): Enhanced `Image Crop with BBox Mask` node with precise dimension control, added `crop_to_side` and `crop_to_length` parameters

**v1.2.25**
- feat(image_crop): Added `apply_paste_mask` parameter to `Image Paste by BBox Mask` node for controlling smart scaling behavior

**v1.2.24**
- feat(image_crop): Added `opacity` parameter to `Image Paste by BBox Mask` node for controlling paste image transparency
- feat(image): Enhanced `Image Stroke by Mask` node with batch processing support for handling multiple images and masks

**v1.2.23**
- fix(image): Enhanced `Image Stroke by Mask` node color parsing logic, supporting RGB string formats and improved default fallback to white color
- fix(image): Enhanced `Image Paste by BBox Mask` node rotation parameter

**v1.2.21**
- feat(text): Added `Text Filter Comment` node for filtering single-line comments (starting with #) and multi-line comments (wrapped in triple quotes), preserving non-comment blank lines
- feat(text): Added `Text Join by Text List` node for merging any type of list into a string with support for prefix, suffix, and custom separators
- refactor(text): Refactored `Text Format` node to `Text Prefix Suffix`, optimizing wildcard input processing and formatting functionality

**v1.2.18**
- feat(sample): Added `Step Split` node for high-low frequency sampling step separation with support for percentage and integer input modes

**v1.2.17**
- feat(image_crop): Optimized `Image Crop with BBox Mask` node

**v1.2.15**
- feat(text): Added `Text Join Multi` node for concatenating multiple text inputs with dynamic variable referencing
- feat(image_crop): Added `Image Edge Crop Pad` node with smart edge cropping and padding capabilities, featuring mask output functionality
- feat(image_blend): Enhanced `Image Luma Matte` node with feathering and alpha output features, supporting multiple color formats and edge processing

**v1.2.13**
- feat(text): Added `Text Load Local` node for loading JSON format prompt files from prompt directory with bilingual Chinese and English output

**v1.2.12**
- feat(text): Added `Text Format` node for flexible text formatting with wildcard input support

**v1.2.9**
 - feat(image_crop): Refactored `Image Crop with BBox Mask` node
 
**v1.2.8**
- feat(image): Added `Image Resize Flux Kontext` node with support for automatic and manual size selection for images and masks
- feat(image): Enhanced `Image Edit Stitch` node with improved stitching algorithms and parameter handling

**v1.2.7**
- feat(text): Added `List Custom Seed` node for creating unique random seed lists with control after generate functionality

**v1.2.6**
- feat(image_hlfreq): Added high-low frequency separation node group, including `Image HLFreq Separate`, `Image HLFreq Combine`, and `Image HLFreq Transform` nodes with support for RGB, HSV, and IGBI frequency separation methods

**v1.2.5**
- feat(mask): Added the `Mask Fill Hole` node, which fills holes in enclosed areas of masks with support for batch processing.

**v1.2.3**
- fix(image_blend): Fixed the issue of inconsistency between devices for the `Image Blend Modes by Alpha` node.

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
| Image Solid | Generate solid color images with enhanced color parameter supporting multiple input formats (grayscale, RGB tuples, hex colors, color names, single-letter shortcuts) and multiple size presets |
| Image Solid FluxKontext | Generate solid color images with Flux Kontext dimension presets and flexible color input formats |
| Image Solid QwenImage | Generate solid color images with QwenImage dimension presets and flexible color input formats |
| Image Resize FluxKontext | Resize images to FluxKontext dimensions with support for automatic and manual size selection for images and masks |
| Image Resize Qwen Image | Resize images to Qwen vision model optimized dimensions with 7 preset resolutions and automatic aspect ratio selection |
| Image Resize Universal | Universal image resizing with multiple algorithms and constraints |
| Image Edit Stitch | Image stitching and merging with multiple stitching modes |
| Image Add Label | Add text labels to images |
| Image Plot | Image plotting and visualization tools |
| Image Stroke by Mask | Apply stroke effects to mask regions with customizable width and color |
| Image BBox Overlay by Mask | Mask-based image bounding box overlay with independent and merge modes |
| Image Rotate with Mask | Advanced image rotation with mask support, multiple fill modes, and mask center rotation options |
| Image Grid Split | Split images into grid layouts with flexible row/column configuration and selective output options |

### 🎨 Image Blending Nodes
| Node Name | Description |
|-----------|-------------|
| Image Luma Matte | Luminance-based image mask compositing with feathering, alpha output, and multiple color format support |
| Image Blend Modes by Alpha | Alpha-based image blending with multiple Photoshop-style blend modes |
| Image Blend Modes by CSS | CSS standard blend modes based on Pilgram library |

### ✂️ Image Cropping Nodes
| Node Name | Description |
|-----------|-------------|
| Image Crop Square | Square cropping with mask guidance and scaling support |
| Image Crop with BBox Mask | Smart bounding box cropping with precise aspect ratio control and scale strength adjustment |
| Image Crop by Mask Alpha | Batch mask-based cropping with RGB/RGBA dual output modes and smart channel processing |
| Image Paste by BBox Mask | Paste cropped images back with multiple blend modes |
| Image Edge Crop Pad | Smart edge cropping and padding with multiple padding modes and mask output |

### 🧩 Image Tiling Nodes
| Node Name | Description |
|-----------|-------------|
| Image Tile Split | Image tile splitting with overlap and custom grid support |
| Image Tile Split Preset | Image tile splitting with predefined resolution presets and intelligent tile size selection |
| Image Tile Merge | Image tile merging with intelligent stitching |

### 🌊 High-Low Frequency Separation Nodes
| Node Name | Description |
|-----------|-------------|
| Image HLFreq Separate | Advanced frequency separation node supporting RGB, HSV, and IGBI separation methods with precise high-low frequency image separation and automatic recombination |
| Image HLFreq Combine | Advanced frequency recombination node supporting RGB, HSV, and IGBI recombination modes with intensity adjustment and intelligent batch matching |
| Image HLFreq Transform | Advanced detail transfer node supporting IGBI, RGB, and HSV transfer methods for precise texture detail migration from detail images to generated images |

### 🎭 Mask Operation Nodes
| Node Name | Description |
|-----------|-------------|
| Mask Fill Hole | Fill holes in enclosed areas of masks with batch processing support |
| Mask Math Ops | Mask mathematical operations (intersection, union, difference, XOR) |
| Mask Batch Math Ops | Batch mask mathematical operations |
| Mask Crop by BBox Mask | Mask bounding box cropping based on mask regions |
| Mask Paste by BBox Mask | Simplified mask pasting with automatic base mask creation and bounding box detection |

### 🔧 Utility Nodes
| Node Name | Description |
|-----------|-------------|
| Image Get Size | Extract image dimensions (width and height) from input images with automatic batch processing support |
| Step Split | High-low frequency sampling step separator supporting percentage (0.0-1.0) and integer input modes for precise sampling control |
| Range Mapping | Value range mapping tool supporting linear transformation and precision control for slider values |
| Path Build | Path builder supporting preset paths and custom extensions |

### 🔄 Conversion Nodes
| Node Name | Description |
|-----------|-------------|
| Image Batch to List | Convert batch images to image lists for individual processing |
| Image List to Batch | Convert image lists to batch images with automatic size normalization |
| Mask Batch to List | Convert batch masks to mask lists for individual processing |
| Mask List to Batch | Convert mask lists to batch masks with automatic size normalization |
| String Coordinate to BBoxes | Convert string format coordinates to BBOXES format with enhanced format support and improved SAM2 compatibility |
| String Coordinate to BBox Mask | Convert string format coordinates to BBoxMask format with image dimension support and flexible output modes |
| URL to Video | Convert video URLs to ComfyUI VIDEO objects with improved error handling, timeout control, and support for both synchronous and asynchronous download methods |

### 🧠 Logic Nodes
| Node Name | Description |
|-----------|-------------|
| Image List Append | Image list appender for intelligently merging images into lists |

### 📝 Text Processing Nodes
| Node Name | Description |
|-----------|-------------|
| Text Filter Comment | Text comment filter for filtering single-line comments (starting with #) and multi-line comments (wrapped in triple quotes), preserving non-comment blank lines |
| Text Join Multi | Multi-input text concatenator supporting multiple text inputs with dynamic variable referencing and custom separators |
| Text Join by Text List | Text list joiner for merging any type of list into a string with support for prefix, suffix, and custom separators |
| Text Prefix Suffix | Text prefix suffix formatter with wildcard input support for flexible data formatting with custom prefix, suffix, and separator |
| Text Custom Extract | Text custom extractor for extracting specified key values from JSON |
| List Custom Int | Custom integer list generator with dash separator and multiple delimiter support |
| List Custom Float | Custom float list generator with dash separator and multiple delimiter support |
| List Custom String | Custom string list generator with dash separator and multiple delimiter support |
| List Custom Seed | Custom seed list generator for creating unique random seed lists with control after generate functionality |



## 🙆 Acknowledgments

[ComfyUI](https://github.com/comfyanonymous/ComfyUI)

[ComfyUI-KJNodes](https://github.com/kijai/ComfyUI-KJNodes)

[ComfUI-Easy-Use](https://github.com/yolain/ComfyUI-Easy-Use)

[ComfyUI_LayerStyle](https://github.com/chflame163/ComfyUI_LayerStyle)

[Comfyui_TTP_Toolset](https://github.com/TTPlanetPig/Comfyui_TTP_Toolset)

[comfyUI_FrequencySeparation_RGB-HSV](https://github.com/risunobushi/comfyUI_FrequencySeparation_RGB-HSV)



## 🌟 Star

My gratitude extends to the generous souls who bestow a star.

[<img src="imgs/Stargazers.png" alt="Stargazers" style="zoom:80%;" />](https://github.com/1hew/ComfyUI-1hewNodes/stargazers)
