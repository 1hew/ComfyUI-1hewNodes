# Load Image - Load images from file or folder

**Node Purpose:** `Load Image` loads images from a file path or a folder and outputs an image batch plus a matching mask batch. It supports recursive folder scan, indexed selection, batch loading, and size unification via center-crop + resize.

## Inputs

| Name | Port | Type | Default | Range | Description |
| ---- | ---- | ---- | ------- | ----- | ----------- |
| `get_image_size` | optional | IMAGE | - | - | Reference image used to unify output size (first frame size). |
| `path` | - | STRING | `""` | - | File path or folder path. |
| `index` | - | INT | `0` | -8192-8192 | Image index when `all=false`; supports negative indices via modulo selection. |
| `include_subdir` | - | BOOLEAN | `true` | - | Include subfolders when `path` is a folder. |
| `all` | - | BOOLEAN | `false` | - | Load all matched images as a batch. |

## Outputs

| Name | Type | Description |
|------|------|-------------|
| `image` | IMAGE | Loaded image (single) or image batch (when `all=true`). |
| `mask` | MASK | Mask aligned to `image`; derived from alpha/transparency and optional sidecar masks. |

## Features

- File or folder input: accepts a single image file or a directory.
- Batch mode: `all=true` loads all matched files as a batch.
- Size unification: uses `get_image_size` or the first loaded image as the target size and applies center-crop + resize.
- Mask extraction: builds masks from image alpha/transparency and merges with sidecar mask files when present.
- Stable ordering: paths are sorted case-insensitively for consistent indexing across platforms.

## Typical Usage

- Load a directory of images as a batch, unify size with a reference, then feed into video encoding.
- Load a single image by index from a folder and use its derived mask for downstream compositing.

## Notes & Tips

- Sidecar masks follow common naming patterns such as `name_mask.png` or `name.mask.webp`.

