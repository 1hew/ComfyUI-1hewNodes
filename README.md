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

**v3.2.6**
- refactor(condition): Optimize `Text Encode QwenImageEdit` node

**v3.2.3**
- refactor(io): Optimize IO group

**v3.2.1**
- feat(io): Add video loading progress bar

**v3.2.0**
- refactor(io): Refactor IO group

**v3.1.0**
- feat(io): Add `Save Video by Image` node for encoding IMAGE batches into video
- refactor(color): Enhance `Match Brightness Contrast` sequence consistency options

**v3.0.8**
- refactor(image_crop): Optimize `Image Crop With BBox Mask` node
- refactor(color): Optimize `Match Brightness Contrast` node

**v3.0.7**
- feat(color): Add `Match Brightness Contrast` node

**v3.0.6**
- feat(mask): Add `Mask Repeat` node, supporting batch mask repetition and inversion
- feat(io): Add `Get File Count`, `Load Image From Folder`, and `Load Video From Folder` nodes to optimize file loading logic and stability

**v3.0.5**
- feat(image_tile): Merged `Image Tile Split` and `Image Tile Split Preset` nodes into a unified `Image Tile Split` node

**v3.0.2**
- feat(text): Add `String Filter` node
- feat(text): Add `String Join Multi` node
- feat(conversion): Add `Text List to String` node

**v3.0.1**
- feat(image): Add `Image PingPong` node for bidirectional batch frame generation with pre-reverse, link-frame removal, and frame truncation
- feat(audio): Add `Audio Duration` node to get audio length (seconds)

**v3.0.0**
- build: bump version to 3.0.0

<details>
<summary><b> 2.x releases</b></summary>

**v2.0.5**
- feat(multi): Added `Multi String Join`, `Multi Image Batch`, `Multi Mask Batch`, `Multi Image Stitch`
- feat(image): Added `Image Three Stitch` node
- feat(condition): Added `Text Encode QwenImageEdit Keep Size` node

**v2.0.3**
- feat(image_blend): Added `output_mask_invert` parameter to `Image Mask Blend` for output-only mask inversion

**v2.0.0**
- breaking: Major update. Existing workflows built with previous nodes require reconfiguration after upgrading to function correctly. Please review nodes and parameters carefully.

</details>

<details>
<summary><b> 1.x releases</b></summary>

**v1.2.46**
- feat(detection): Added `DetectGuideLine` node 

**v1.2.45**
- feat(detection): Added `DetectYolo` node for YOLO model object detection

**v1.2.44**
- feat(util): Added `Workflow Name` node

**v1.2.43**
- feat(logic): Added `Video Cut Group` node 

**v1.2.42**
- feat(logic): Added `Image Batch Extract` node for extracting specific images from batches with multiple modes including custom indices, step intervals, and uniform distribution

**v1.2.40**
- feat(text): Added `IntWan` node for generating 4n+1 arithmetic progression sequences with configurable step control and range validation
- refactor(logic): Enhanced `Image Batch Split` and `Mask Batch Split` nodes with improved boundary condition handling and comprehensive error recovery

<details>
<summary><b>v1.2.39</b></summary>

- feat(logic): Added `Image Batch Group` node for intelligent image batch splitting with overlap support and flexible padding strategies

​	</details>

<details>
<summary><b>v1.2.38</b></summary>

- refactor(image): Enhanced `ImageResizeUniversal` node with comprehensive mask processing logic

​	</details>

<details>
<summary><b>v1.2.37</b></summary>

- feat(image): Enhanced `Image Solid` node with advanced color parameter
- feat(image): Added the `ImageGridSplit` node for splitting images into grid layouts with flexible output options

​	</details>

<details>
<summary><b>v1.2.36</b></summary>

- feat(conversion): Enhanced `URL to Video` node

​	</details>

<details>
<summary><b>v1.2.35</b></summary>

- feat(image): Added `Image Resize Qwen Image` node for Qwen vision model optimized image resizing with 7 preset resolutions and automatic aspect ratio selection

​	</details>

<details>
<summary><b>v1.2.32</b></summary>

- feat(image): Added `Image Solid Flux Kontext` node for generating solid color images with Flux Kontext dimension presets
- feat(image): Added `Image Solid Qwen Image` node for generating solid color images with QwenImage dimension presets

​	</details>

<details>
<summary><b>v1.2.31</b></summary>

- fix: Fixed various bugs and improved stability

​	</details>

<details>
<summary><b>v1.2.28</b></summary>

- feat(mask): Added `Mask Paste by BBox Mask` node for simplified mask pasting with automatic base mask creation and bounding box detection
- feat(image_tile): Added `Image Tile Split Preset` node with predefined resolution presets and intelligent tile size selection
- feat(image): Added `Image Rotate with Mask` node for advanced image rotation with mask support and multiple fill modes
- feat(text): Enhanced `Text Load Local` node with `user_prompt` parameter for combining JSON content with additional user prompts

​	</details>

<details>
<summary><b>v1.2.26</b></summary>

- feat(image_crop): Enhanced `Image Crop with BBox Mask` node with precise dimension control, added `crop_to_side` and `crop_to_length` parameters

​	</details>

<details>
<summary><b>v1.2.25</b></summary>

- feat(image_crop): Added `apply_paste_mask` parameter to `Image Paste by BBox Mask` node for controlling smart scaling behavior

​	</details>

<details>
<summary><b>v1.2.24</b></summary>

- feat(image_crop): Added `opacity` parameter to `Image Paste by BBox Mask` node for controlling paste image transparency
- feat(image): Enhanced `Image Stroke by Mask` node with batch processing support for handling multiple images and masks

​	</details>

<details>
<summary><b>v1.2.23</b></summary>

- fix(image): Enhanced `Image Stroke by Mask` node color parsing logic, supporting RGB string formats and improved default fallback to white color
- fix(image): Enhanced `Image Paste by BBox Mask` node rotation parameter

​	</details>

<details>
<summary><b>v1.2.21</b></summary>

- feat(text): Added `Text Filter Comment` node for filtering single-line comments (starting with #) and multi-line comments (wrapped in triple quotes), preserving non-comment blank lines
- feat(text): Added `Text Join by Text List` node for merging any type of list into a string with support for prefix, suffix, and custom separators
- refactor(text): Refactored `Text Format` node to `Text Prefix Suffix`, optimizing wildcard input processing and formatting functionality

​	</details>

<details>
<summary><b>v1.2.18</b></summary>

- feat(sample): Added `Step Split` node for high-low frequency sampling step separation with support for percentage and integer input modes

​	</details>

<details>
<summary><b>v1.2.17</b></summary>

- feat(image_crop): Optimized `Image Crop with BBox Mask` node

​	</details>

<details>
<summary><b>v1.2.15</b></summary>

- feat(text): Added `Text Join Multi` node for concatenating multiple text inputs with dynamic variable referencing
- feat(image_crop): Added `Image Edge Crop Pad` node with smart edge cropping and padding capabilities, featuring mask output functionality
- feat(image_blend): Enhanced `Image Luma Matte` node with feathering and alpha output features, supporting multiple color formats and edge processing

​	</details>

<details>
<summary><b>v1.2.13</b></summary>

- feat(text): Added `Text Load Local` node for loading JSON format prompt files from prompt directory with bilingual Chinese and English output

​	</details>

<details>
<summary><b>v1.2.12</b></summary>

- feat(text): Added `Text Format` node for flexible text formatting with wildcard input support

​	</details>

<details>
<summary><b>v1.2.9</b></summary>

- feat(image_crop): Refactored `Image Crop with BBox Mask` node

​	</details>

<details>
<summary><b>v1.2.8</b></summary>

- feat(image): Added `Image Resize Flux Kontext` node with support for automatic and manual size selection for images and masks
- feat(image): Enhanced `Image Edit Stitch` node with improved stitching algorithms and parameter handling

​	</details>

<details>
<summary><b>v1.2.7</b></summary>

- feat(text): Added `List Custom Seed` node for creating unique random seed lists with control after generate functionality

​	</details>

<details>
<summary><b>v1.2.6</b></summary>

- feat(image_hlfreq): Added high-low frequency separation node group, including `Image HLFreq Separate`, `Image HLFreq Combine`, and `Image HLFreq Transform` nodes with support for RGB, HSV, and IGBI frequency separation methods

​	</details>

<details>
<summary><b>v1.2.5</b></summary>

- feat(mask): Added the `Mask Fill Hole` node, which fills holes in enclosed areas of masks with support for batch processing.

​	</details>

<details>
<summary><b>v1.2.3</b></summary>

- fix(image_blend): Fixed the issue of inconsistency between devices for the `Image Blend Modes by Alpha` node.

​	</details>

<details>
<summary><b>v1.2.2</b></summary>

- feat(image): Added the `Image BBox Overlay by Mask` node, which overlays the image bounding box based on a mask.

​	</details>

<details>
<summary><b>v1.2.1</b></summary>

- refactor(image/crop): Renamed node classes and updated related documentation
- feat(image_crop): Enhanced the functionality and output options of the `ImageCropByMaskAlpha` node

​	</details>

<details>
<summary><b>v1.2.0</b></summary>

- feat: Added `conversion`, and restructured image mixing and masking processing

​	</details>

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

</details>



## 📋 Node List

### 🖼️ Image Processing Nodes
| Node Name | Description |
|-----------|--------------|
| Image Solid FluxKontext | Generate solid color images based on Flux Kontext dimension presets with flexible color input format support |
| Image Resize FluxKontext | Resize images to FluxKontext dimensions with support for automatic and manual size selection for images and masks |
| Image Resize Qwen Image | Image resizing optimized for Qwen vision models, providing 7 preset resolutions and automatic aspect ratio selection |
| Image Resize Universal | Universal image resizing with multiple algorithms and constraints |
| Image Rotate with Mask | Advanced image rotation with mask integration, multiple fill modes, and mask center rotation options |
| Image Edit Stitch | Image stitching and merging with multiple stitching modes |
| ImageMainStitch | Main-image stitcher supporting dynamic `image_2..image_N`, direction, size matching, spacing, and padding |
| Image Add Label | Add text labels to images |
| Image Plot | Image plotting and visualization tools |
| Image Stroke by Mask | Apply stroke effects to mask regions with customizable width and color |
| Image BBox Overlay by Mask | Mask-based image bounding box overlay with independent and merge modes |

### 🌈 Color Nodes
| Node Name | Description |
|-----------|-------------|
| Match Brightness Contrast | Adjusts the brightness and contrast of the source image to match the reference image |

### 🎨 Image Blending Nodes
| Node Name | Description |
|-----------|-------------|
| Image Mask Blend | Luminance-based image mask compositing with feathering, alpha output, and multiple color format support |
| Image Blend Mode by Alpha | Alpha-based image blending with multiple Photoshop-style blend modes |
| Image Blend Mode by CSS | CSS standard blend modes based on Pilgram library |

### ✂️ Image Cropping Nodes
| Node Name | Description |
|-----------|-------------|
| Image Mask Crop | Batch mask-based cropping with RGB/RGBA dual output modes and smart channel processing |
| Image Crop Square | Square cropping with mask guidance and scaling support |
| Image Crop with BBox Mask | Smart bounding box cropping with precise aspect ratio control and scale strength adjustment |
| Image Paste by BBox Mask | Paste cropped images back with multiple blend modes |
| Image Edge Crop Pad | Smart edge cropping and padding with multiple padding modes and mask output |
| Image Grid Split | Split images into grid layouts with flexible row/column configuration and selective output options |

### 🧩 Image Tiling Nodes
| Node Name | Description |
|-----------|-------------|
| Image Tile Split | Image tile splitting with auto/grid/preset modes, overlap support, and reference image sizing |
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
| Mask Crop by BBox Mask | Mask bounding box cropping based on mask regions |
| Mask Paste by BBox Mask | Simplified mask pasting with automatic base mask creation and bounding box detection |
| Mask Repeat | Batch repeat masks with optional inversion support |

### 🔍 Detection Nodes
| Node Name | Description |
|-----------|-------------|
| Detect Yolo | YOLO model object detection with subfolder model support, customizable confidence thresholds, and optional label display control |
| Detect Guide Line | Guide line detection combining Canny, HoughLinesP, and DBSCAN vanishing-point clustering |

### 🔧 Utility Nodes
| Node Name | Description |
|-----------|-------------|
| Workflow Name | Automatically retrieves current workflow filename with path control, custom prefixes/suffixes, and date formatting |
| Range Mapping | Value range mapping tool supporting linear transformation and precision control for slider values |

### 🔄 Conversion Nodes
| Node Name | Description |
|-----------|-------------|
| URL to Video | Convert video URLs to ComfyUI VIDEO objects with improved error handling, timeout control, and support for both synchronous and asynchronous download methods |
| Image Batch to List | Convert batch images to image lists for individual processing |
| Image List to Batch | Convert image lists to batch images with automatic size normalization |
| Mask Batch to List | Convert batch masks to mask lists for individual processing |
| Mask List to Batch | Convert mask lists to batch masks with automatic size normalization |
| String Coordinate to BBoxes | Convert string format coordinates to BBOXES format with enhanced format support and improved SAM2 compatibility |
| String Coordinate to BBox Mask | Convert string format coordinates to BBoxMask format with image dimension support and flexible output modes |
| Text List to String | Merge a text list by applying per-item prefix/suffix and joining with a separator; supports escape sequences and composite separators |

### 🧠 Logic Nodes
| Node Name | Description |
|-----------|-------------|
| Any Empty Bool | Universal empty value checker (boolean output version) that checks if any type of input is empty and returns a boolean value |
| Any Empty Int | Universal empty value checker (integer output version) that checks if any type of input is empty and returns custom integer values |
| Any Switch Bool | Universal boolean switch node supporting any type input with lazy evaluation, selecting output based on boolean condition |
| Any Switch Int | Multi-way integer switch node supporting multiple input options, selecting corresponding input/output based on integer index (1-5) |

### 🔢 Integer Nodes
| Node Name | Description |
|-----------|-------------|
| Int Image Side Length | Output selected side length (longest/shortest/width/height) from image dimensions |
| Int Image Size | Output width and height integers from image dimensions |
| Int Mask Side Length | Output selected side length (longest/shortest/width/height) from mask dimensions |
| Int Split | Split a total value into two parts, supporting percentage and integer split point |
| Int Wan | Generate 4n+1 arithmetic sequences with step and range validation |

### 📦 Batch Processing Nodes
| Node Name | Description |
|-----------|-------------|
| Image Batch Extract | Intelligent image batch extractor supporting multiple extraction modes including custom indices, step intervals, and uniform distribution |
| Image Batch Split | Intelligent image batch splitter with forward/backward splitting modes and enhanced boundary condition handling |
| Image Batch Group | Intelligent image batch grouper with configurable batch sizes, overlap handling, and flexible padding strategies |
| Image Batch Range | Select a contiguous range from an image batch using start index and count; out-of-bounds safe |
| Image PingPong | Bidirectional frame repeat over batch, supports pre-reverse, link-frame removal, and frame truncation |
| Image List Append | Image list appender for intelligently merging images into lists |
| Mask Batch Math Ops | Batch mask mathematical operations |
| Mask Batch Range | Select a contiguous range from a mask batch using start index and count; out-of-bounds safe |
| Mask Batch Split | Intelligent mask batch splitter with forward/backward splitting modes and enhanced boundary condition handling |
| Video Cut Group | Video scene cut detector that identifies scene transitions by analyzing frame similarity, supporting both fast and precise modes |

### 📝 Text Processing Nodes
| Node Name | Description |
|-----------|-------------|
| Text Prefix Suffix | Text prefix suffix formatter with wildcard input support for flexible data formatting with custom prefix and suffix |
| Text Custom Extract | Text custom extractor for extracting specified key values from JSON |
| String Filter | Text cleaner supporting `{input}` substitution, comment filtering (# and triple quotes), and optional empty-line removal |
| String Join Multi | Join up to 5 text blocks with `{input}` substitution, comment/empty-line filtering, and composite separators|
| List Custom Int | Custom integer list generator with dash separator and multiple delimiter support |
| List Custom Float | Custom float list generator with dash separator and multiple delimiter support |
| List Custom String | Custom string list generator with dash separator and multiple delimiter support |
| List Custom Seed | Custom seed list generator for creating unique random seed lists with control after generate functionality |

### 🔗 Multi Nodes
| Node Name | Description |
|-----------|-------------|
| Multi String Join | Concatenate dynamic `string_X` inputs with `{input}` variable support and comment/triple-quote filtering; customizable separator |
| Multi Image Batch | Build image batch from dynamic `image_X` with crop/pad/stretch size unification and edge/color padding |
| Multi Image Stitch | Dynamic multi-image stitcher with direction, size matching, spacing, and padding color control |
| Multi Mask Batch | Build mask batch from dynamic `mask_X` with crop/pad/stretch size unification and configurable gray padding |
| Multi Mask Math Ops | Dynamic multi-mask operations (union/intersection/difference/XOR) with batch broadcasting and size alignment |


### 📁 IO Nodes
| Node Name | Description |
|-----------|-------------|
| Get File Count | Count image/video files in a folder with optional recursive scanning |
| Load Image | Load images from a file or folder with batch mode, size unification, and derived masks |
| Load Video | Select a video from a file or folder and apply trimming/FPS settings during decoding |
| Load Video to Image | Decode a video into an image batch, audio, fps, and frame count |
| Save Image | Save image batches to output/temp and return absolute saved file paths |
| Save Video by Image | Encode an image batch into a video with optional audio muxing and alpha support |
| Save Video | Save a VIDEO object to disk with container extension preservation and alpha preview |


### 🎛️ Conditioning Nodes
| Node Name | Description |
|-----------|-------------|
| Text Encode QwenImageEdit | Qwen image-edit conditioning encoder combining vision inputs and text, supports size preservation modes and reference latents |


### 🔊 Audio Nodes
| Node Name | Description |
|-----------|-------------|
| Audio Duration | Get audio length (seconds) |



## 🙆 Acknowledgments

[ComfyUI](https://github.com/comfyanonymous/ComfyUI)

[ComfyUI-KJNodes](https://github.com/kijai/ComfyUI-KJNodes)

[ComfUI-Easy-Use](https://github.com/yolain/ComfyUI-Easy-Use)

[ComfyUI_LayerStyle](https://github.com/chflame163/ComfyUI_LayerStyle)

[Comfyui_TTP_Toolset](https://github.com/TTPlanetPig/Comfyui_TTP_Toolset)

[comfyUI_FrequencySeparation_RGB-HSV](https://github.com/risunobushi/comfyUI_FrequencySeparation_RGB-HSV)

[comfyui_extractstoryboards](https://github.com/gitadmini/comfyui_extractstoryboards)

[ComfyUI-VideoHelperSuite](https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite)



## 🌟 Star

My gratitude extends to the generous souls who bestow a star.

[<img src="imgs/Stargazers.png" alt="Stargazers" style="zoom:80%;" />](https://github.com/1hew/ComfyUI-1hewNodes/stargazers)
