# Any to Any - Wildcard Adapter

**Node Purpose:** `Any to Any` forwards any connected value unchanged and exposes both ports as the wildcard type (`*`), so nodes with different declared types can be wired together even when ComfyUI would otherwise reject the link.

## Inputs

| Name | Port | Type | Default | Range | Description |
| ---- | ---- | ---- | ------- | ----- | ----------- |
| `any` | optional | ANY (`*`) | - | - | Value to forward; accepts any upstream output type |

## Outputs

| Name | Type | Description |
|------|------|-------------|
| `any` | ANY (`*`) | The same value, re-exposed as a wildcard |

## Features

- Type-erasing pass-through: the value is forwarded untouched, with no cast or data conversion.
- Wildcard on both sides: a typed output can connect to `any`, and `any` can connect to a typed input.
- Optional input: returns `None` when nothing is connected.
- Type agnostic: works with tensors, models, conditioning, audio/video objects, strings, numbers, lists, and custom payloads.

## Typical Usage

- Bridge a node whose output type differs from the next node's declared input type.
- Feed a typed source into a widget or `COMBO` input that ComfyUI would not link directly.
- Prototype a graph quickly without inserting a real converter on every edge.

## Notes & Tips

- This node only removes the connection-type restriction; it does not convert the data itself. Make sure the downstream node can actually consume the upstream value.
- When an actual transform is required (for example image to mask), use the matching conversion node instead.
- Each output consumer receives the original object, so a wildcard link cannot repair shape, channel, or format mismatches.
