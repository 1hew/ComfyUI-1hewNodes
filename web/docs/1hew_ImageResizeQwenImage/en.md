# Image Resize Qwen Image

**Node Function:** The `Image Resize Qwen Image` node is a Qwen image preset scaler that supports Qwen vision model optimized preset resolutions and automatic optimal selection. It can resize images and masks to predefined Qwen-optimized resolutions or automatically select the best resolution based on input aspect ratio.

## Inputs

| Parameter | Required | Data Type | Default | Range | Description |
|--|--|--|--|--|--|
| `preset_size` | Required | COMBO[STRING] | auto | Preset resolution list + auto | Target resolution selection. "auto" automatically selects the best resolution based on input aspect ratio, or choose from 7 predefined Qwen-optimized resolutions |
| `image` | Optional | IMAGE | - | - | Input image to resize. When provided with "auto" mode, the node selects the best preset resolution based on image aspect ratio |
| `mask` | Optional | MASK | - | - | Input mask to resize. When provided with "auto" mode (without image), the node selects the best preset resolution based on mask aspect ratio |

## Outputs

| Output Name | Data Type | Description |
|-------------|-----------|-------------|
| `image` | IMAGE | Resized image. If no input image provided, outputs a black solid image |
| `mask` | MASK | Resized mask. If no input mask provided, outputs a white solid mask |