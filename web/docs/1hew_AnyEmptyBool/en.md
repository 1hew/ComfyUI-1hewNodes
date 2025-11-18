# Any Empty Bool - Unified Emptiness Detection

**Node Purpose:** `Any Empty Bool` evaluates whether an input is considered “empty” and outputs a boolean. It supports common Python types, tensors, arrays, and nested containers with consistent rules.

## Inputs

| Name | Port | Type | Default | Range | Description |
| ---- | ---- | ---- | ------- | ----- | ----------- |
| `any` | - | ANY (`*`) | - | - | Value to test for emptiness. |

## Outputs

| Name | Type | Description |
|------|------|-------------|
| `bool` | BOOLEAN | `True` when input is empty, otherwise `False`.

## Features

- Type-aware rules:
- `None` → empty.
- `str` → empty when `len(value.strip()) == 0`.
- `bool` → empty when `False`.
- `int/float` → empty when `== 0`.
- `torch.Tensor` → empty when `numel()==0` or all elements `==0`.
- `numpy.ndarray` → empty when `size==0` or all elements `==0`.
- `list/tuple` → empty when length `==0` or all items empty (recursive).
- `dict` → empty when length `==0`.
- Any object with `__len__` → empty when `len(obj)==0`.
- Robust fallback: on exceptions, returns `True` to indicate empty.

## Typical Usage

- Flow gating: connect to conditional nodes to enable/disable subgraphs based on presence of data.
- Parameter defaults: detect empty inputs and branch to defaults via `Any Switch Bool`.
- Batch safety: verify computed arrays/tensors are non-zero before downstream processing.

## Notes & Tips

- Tensors/arrays must be entirely zero to count as empty; any non-zero element is considered non-empty.
- Nested containers are checked recursively, enabling deep validation of compound inputs.