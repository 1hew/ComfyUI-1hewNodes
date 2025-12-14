# Load Image From Folder

**Node Purpose:** Loads image files from a specified folder. Supports batch loading, subfolder traversal, and automatic resizing based on a reference image.

## Inputs

| Name | Port | Type | Default | Range | Description |
| ---- | ---- | ---- | ------- | ----- | ----------- |
| `get_image_size` | optional | IMAGE | - | - | Optional reference image. If provided, loaded images will be resized to match this image's dimensions. |
| `folder` | - | STRING | "" | - | The absolute path to the folder containing image files. |
| `index` | - | INT | 0 | -8192~8192 | The index of the image to load (ignored if `all` is True). |
| `include_subfolder` | - | BOOLEAN | True | - | Whether to include image files in subfolders. |
| `all` | - | BOOLEAN | False | - | If True, loads all images in the folder as a batch. If False, loads a single image specified by `index`. |

## Outputs

| Name | Type | Description |
|------|------|-------------|
| `image` | IMAGE | The loaded image(s). Returns a batch of images if `all` is True. |
| `mask` | MASK | The mask(s) corresponding to the image(s). Uses alpha channel if available, otherwise full white. |

## Features

- **Batch Loading**: Can load all images in a folder at once (`all=True`) or a single image by index.
- **Auto Resizing**: If `get_image_size` is connected, all loaded images are cropped and resized (Center Crop + Resize) to match the reference dimensions.
- **Batch Resizing**: In batch mode (`all=True`), if no reference image is provided, the first image's size is used as the target size for all subsequent images to ensure tensor shape consistency.
- **Supported Formats**: `.png`, `.jpg`, `.jpeg`, `.bmp`, `.webp`, `.tiff`, `.gif`.
