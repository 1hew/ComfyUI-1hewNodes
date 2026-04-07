# Image to URL - Convert image to URL

**Node Purpose:** `Image to URL` converts an `IMAGE` tensor into a URL string that can be passed to downstream nodes or external APIs.

## Inputs

| Name | Port | Type | Default | Range | Description |
| ---- | ---- | ---- | ------- | ----- | ----------- |
| `image` | - | IMAGE | - | - | Input image, supports single image or batch |
| `mode` | dropdown | COMBO | `auto` | `auto` / `kefan` / `data` | Conversion mode: automatic fallback, upload-only via `kefan.cn`, or direct data URL output |
| `timeout` | - | INT | 30 | 5-300 | Upload timeout in seconds |

## Outputs

| Name | Type | Description |
|------|------|-------------|
| `url` | STRING | URL output; returns a single URL for one image, or a multi-line string with one URL per line for batch input |

## Features

- `auto`: tries uploading to `kefan.cn` first and falls back to a data URL if upload fails.
- `kefan`: uploads only to `kefan.cn`; returns a public URL on success and raises an error on failure.
- `data`: directly emits a `data:image/...;base64,...` string and does not depend on any external upload service.
- Batch support: when the input is an image batch, each image is converted and the final output joins URLs as a multi-line string, one URL per line.
- Reuses upload cache: repeated uploads of the same image reuse the cached URL mapping when available.

## Typical Usage

- Convert generated ComfyUI images with `data` mode for LLM or HTTP request nodes that accept image URLs / data URLs.
- Switch to `kefan` mode for downstream services that only accept public image links.
- Use `auto` by default when you prefer a public URL but still want a data URL fallback.
- For batch images, pass the multi-line result to text-processing nodes that can work line by line.

## Notes & Tips

- `kefan` depends on external network access and third-party service availability; prefer `data` if stability matters more.
- If a downstream node expects exactly one URL, make sure the input is a single image or split the multi-line output first.
