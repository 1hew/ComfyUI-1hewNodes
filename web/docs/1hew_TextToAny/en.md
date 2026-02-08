# Text to Any - Text to wildcard

**Node Purpose:** `Text to Any` outputs a user-typed text as a wildcard (`*`) payload, enabling more flexible connections (for example, some COMBO inputs can accept a text payload when wired).

## Inputs

| Name | Port | Type | Default | Range | Description |
| ---- | ---- | ---- | ------- | ----- | ----------- |
| `text` | - | STRING | `` | multiline | Input text |

## Outputs

| Name | Type | Description |
|------|------|-------------|
| `any` | * | Wildcard payload containing the input text |

## Features

- Wildcard output: uses `*` to maximize connection compatibility.
- List input support: when `text` is provided as a list, the node uses the first item.

## Typical Usage

- Provide file names, identifiers, or prompt text to downstream nodes via a flexible wildcard connection.

## Notes & Tips

- This node only forwards the text; it does not validate formatting. Keep the value consistent with downstream expectations.
