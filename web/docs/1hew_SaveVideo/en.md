# Save Video - Save to Output with Preview

**Node Purpose:** `Save Video` saves the input video to the output folder using a selected container and codec. When the input is empty, the node completes without side effects. It provides a preview entry in the UI.

## Inputs

| Name | Port | Type | Default | Range | Description |
| ---- | ---- | ---- | ------- | ----- | ----------- |
| `video` | optional | VIDEO | - | - | Video to save; when absent, the node passes through. |
| `filename_prefix` | - | STRING | `video/ComfyUI` | - | Save path prefix; supports formatting placeholders handled by the framework. |
| `format` | - | COMBO | `auto` | container options | Container format selection; `auto` chooses a suitable format. |
| `codec` | - | COMBO | `auto` | codec options | Codec selection; `auto` chooses based on container and content. |

## Outputs

| Name | Type | Description |
|------|------|-------------|
| - | - | Output node; returns a UI preview element for the saved video. |

## Features

- Safe no-input behavior: completes immediately when `video` is not provided.
- Path computation: uses `get_save_image_path` to compute folder, filename, counter, and subfolder.
- Metadata handling: includes prompt and extra info unless metadata is disabled.
- Extension resolution: derives extension from selected container.
- Async saving: saves via `video.save_to` on a background thread.
- UI preview: returns `PreviewVideo` with saved result info.
- Output node: marked as an output node and hides prompt/extra PNG info in the interface.

## Typical Usage

- Choose container and codec: set `format` and `codec` explicitly for target platforms.
- Prefix organization: use `filename_prefix` to route outputs into subfolders like `video/ComfyUI`.
- Batch-friendly naming: rely on the auto-increment `counter` in filename to avoid collisions.

## Notes & Tips

- The filename pattern is `{filename}_{counter:05}_.{extension}`; customize `filename_prefix` to guide folder and base name.
- When metadata is enabled, prompt and extra info are saved alongside the video.