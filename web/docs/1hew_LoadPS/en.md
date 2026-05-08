# Load PS - Load PSD/PSB layers

**Node Purpose:** `Load PS` loads one PSD/PSB file as a selected layer, an all-layer image batch, or the merged document image. It also outputs matching alpha masks, the source filename, and layer names. The node supports drag-and-drop/file-picker upload and shows a preview for the current mode.

## Inputs

| Name | Port | Type | Default | Range | Description |
| ---- | ---- | ---- | ------- | ----- | ----------- |
| `file` | - | STRING | `""` | - | PSD/PSB file path. Supports paths relative to the ComfyUI input directory or absolute paths. |
| `index` | - | INT | `0` | -8192-8192 | Layer/group index used when `output_mode=single_layer`; supports negative indices via modulo selection. |
| `include_hidden` | - | BOOLEAN | `false` | - | Include hidden Photoshop layers or hidden groups. |
| `group_mode` | - | COMBO | `layer` | `layer`, `merged` | `layer` expands group children; `merged` treats each group as one composited image. |
| `output_mode` | - | COMBO | `all_layers` | `single_layer`, `all_layers`, `merged` | Output one indexed layer, all valid layers as a batch, or the merged PSD image. |
| `preview` | - | BOOLEAN | `false` | - | Show a node preview. When disabled, drag-and-drop upload only fills `file` and does not trigger PSD compositing for preview. |

## Outputs

| Name | Type | Description |
|------|------|-------------|
| `image` | IMAGE | Loaded RGBA image. `all_layers` returns a batch; other modes return one image. |
| `mask` | MASK | Alpha mask aligned to `image`; visible pixels are 1 and transparent pixels are 0. |
| `filename` | STRING | PSD/PSB filename stem. |
| `layer_name` | STRING | Current layer/group name. Batch mode returns newline-separated names matching the batch order. |

## Features

- Single-file input: loads one explicit PSD/PSB file and does not scan folders.
- Drag-and-drop upload: drop `.psd` / `.psb` files onto the node, or use the `choose psd to upload` button.
- Node preview: when `preview` is enabled, `merged` and `single_layer` show one preview image, while `all_layers` shows a layer grid preview.
- Layer modes: `single_layer` selects one layer/group by `index`; `all_layers` outputs all valid layers/groups as a batch.
- Merged mode: `merged` outputs the full PSD composite and ignores `index` and `group_mode`.
- Group handling: `group_mode=layer` expands group children; `group_mode=merged` outputs each group as one image.
- Empty filtering: empty layers and empty group composites are skipped.

## Typical Usage

- Load rasterized PSD layers as an IMAGE batch for per-layer processing in ComfyUI.
- Use `single_layer` plus `index` to load one specific layer or group.
- Use `merged` to quickly get the full PSD preview image.

## Notes & Tips

- This node requires `psd-tools`; make sure dependencies are installed.
- Rasterized PSD layers are recommended. Complex layer styles, smart objects, and adjustment layers depend on what the PSD file and parser can provide.
- For large or layer-heavy PSD files, keep `preview=false` and enable it only when needed.
