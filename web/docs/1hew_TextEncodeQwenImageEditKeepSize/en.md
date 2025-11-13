# Text Encode QwenImageEdit Keep Size

**Node Function:** The `Text Encode QwenImageEdit Keep Size` node produces conditioning for Qwen/VL image-edit workflows. It combines the user's text with one or more vision inputs, standardizes vision features, and, when a `VAE` is provided, encodes and attaches `reference_latents`, preserving original image sizes according to `keep_size` or scaling to `base_size` aligned to multiples of 8.

## Inputs

| Parameter | Required | Data Type | Default | Range | Description |
|--|--|--|--|--|--|
| `clip` | Required | CLIP | - | - | Encoder (Qwen/CLIP) used to tokenize text and vision features |
| `prompt` | Required | STRING | "" | Multi-line text | Edit instruction text, supports multi-line |
| `keep_size` | Required | COMBO[STRING] | first | none, first, all | Size retention policy for reference latents when `VAE` is present |
| `base_size` | Required | COMBO[STRING] | 1024 | 1024, 1536, 2048 | Base side length used for area-conserving scaling and 8x alignment |
| `vae` | Optional | VAE | - | - | VAE to encode `reference_latents`; omit to skip latents attachment |
| `image_1` | Optional | IMAGE | - | - | Primary input image; supports additional dynamic ports `image_2`, `image_3`, ... |

## Outputs

| Output Name | Data Type | Description |
|-------------|-----------|-------------|
| `conditioning` | CONDITIONING | Conditioning with tokenized text and vision features, optionally including `reference_latents` |
