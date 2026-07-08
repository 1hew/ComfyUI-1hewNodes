# Memory Cleanup - Memory Cleanup

**Node Purpose:** `Memory Cleanup` requests ComfyUI to release execution cache and model memory after the current task completes, ideal for cleaning up RAM/VRAM between queued executions.

## Inputs

| Name | Port | Type | Default | Range | Description |
| ---- | ---- | ---- | ------- | ----- | ----------- |
| `anything` | optional | `*` | - | - | Arbitrary input, passed through to output. |
| `unload_model` | - | BOOLEAN | `False` | `True` / `False` | Whether to request unloading loaded models after the task. Saves VRAM/RAM but may require model reloading on the next run. |

## Outputs

| Name | Type | Description |
|------|------|-------------|
| `output` | `*` | Passes through the `anything` input value. |

## Features

- Calls `gc.collect()` to trigger Python garbage collection on execution.
- Invokes ComfyUI's `soft_empty_cache` to free PyTorch cache.
- Sets the `free_memory` flag via prompt_queue to request execution cache release.
- When `unload_model=True`, additionally sets `unload_models` flag to unload loaded models.

## Typical Usage

- Insert this node between queued tasks to prevent VRAM from accumulating.
- Enable `unload_model=True` when switching models or workflows to release old models.
- Place at the end of a workflow to auto-cleanup after each execution.

## Notes & Tips

- This is an output node (`is_output_node=True`), so ComfyUI ensures it executes at the end of the workflow.
- Enabling `unload_model` causes model reload on the next run, increasing single-run time; use as needed.
- The `anything` input accepts any type, allowing you to chain preceding nodes to ensure cleanup triggers at the correct execution order.
