# Image Resize Flux Kontext

**Node Function:** The `Image Resize Flux Kontext` node is a Flux preset image scaler that supports preset resolutions and automatic optimal selection. It can resize images and masks to predefined Flux-optimized resolutions or automatically select the best resolution based on input aspect ratio.

## Inputs

| Parameter | Required | Data Type | Default | Range | Description |
|--|--|--|--|--|--|
| `preset_size` | Required | COMBO[STRING] | auto | Preset resolution list + auto | Target resolution selection. "auto" automatically selects the best resolution based on input aspect ratio, or choose from 17 predefined Flux-optimized resolutions |
| `image` | Optional | IMAGE | - | - | Input image to resize. When provided with "auto" mode, the node selects the best preset resolution based on image aspect ratio |
| `mask` | Optional | MASK | - | - | Input mask to resize. When provided with "auto" mode (without image), the node selects the best preset resolution based on mask aspect ratio |

## Outputs

| Output Name | Data Type | Description |
|-------------|-----------|-------------|
| `image` | IMAGE | Resized image. If no input image provided, outputs a black solid image |
| `mask` | MASK | Resized mask. If no input mask provided, outputs a white solid mask |