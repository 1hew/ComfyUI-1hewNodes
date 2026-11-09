# Text System Prompt - Load a builtin system prompt

**Node Purpose:** `Text System Prompt` returns one stored system prompt as a plain string, so any LLM or prompt-consuming node can be fed a curated system prompt without copy-pasting it into a widget. Prompts live as `.txt` files under `utils/system_prompt/`; the dropdown mirrors that folder tree, and the selected body is returned verbatim.

## Inputs

| Name | Type | Default | Range | Description |
| ---- | ---- | ------- | ----- | ----------- |
| `file` | COMBO | first entry | every `.txt` file under `utils/system_prompt/` | Dropdown entry of the form `<model>/[<lang>] <name>`, which is the file path relative to `utils/system_prompt/` without the `.txt` suffix |

## Outputs

| Name | Type | Description |
|------|------|-------------|
| `string` | STRING | The selected prompt body, returned verbatim: no trimming, no placeholder substitution, only a UTF-8 BOM is dropped |

## Folder Layout

```
utils/system_prompt/
└── <model folder>/
    └── [<language>] <prompt name>.txt
```

| File | Dropdown entry |
| ---- | -------------- |
| `qwen_image_2.1/[en] edit.txt` | `qwen_image_2.1/[en] edit` |
| `qwen_image_2.1/[en] t2i.txt` | `qwen_image_2.1/[en] t2i` |

## Features

- **Model-oriented tree:** one folder per target model keeps the official spelling, dots included (`qwen_image_2.1`); these are plain directories, not Python packages, so the dot is harmless.
- **Language in the file name:** `[en]` / `[cn]` is part of the file name, so one file holds exactly one language and an English/Chinese pair is two files.
- **Startup registry:** the key list is scanned once when the extension loads and cached, so opening the dropdown costs nothing.
- **mtime-aware reads:** editing an existing body takes effect on the next run without restarting ComfyUI; adding or removing a file needs a reload because the dropdown is fixed at startup.
- **Skip rules:** files and folders whose name starts with `_` or `.` are ignored, so notes can sit next to the prompts without polluting the dropdown.
- **Never blocks a graph:** an empty folder shows one placeholder entry and returns an empty string; an unknown key logs a warning and returns `""`.

## Typical Usage

- Feed a curated system prompt straight into an LLM or prompt-enhancer node.
- Keep one prompt per model/language pair and pick the variant per workflow instead of editing a big text widget.
- Pair with `String Join Multi` or `Text Prefix Suffix` when the system prompt has to be merged with a user prompt.

## Notes & Tips

- The dropdown is a flat list; the `/` in each entry is the folder separator, not a sub-widget.
- Bodies are read as UTF-8; a BOM is stripped, everything else is preserved byte for byte.
- Do not rename an entry casually: the dropdown value is stored in workflows, so renaming a file makes older workflows select the first entry instead.
- Only self-contained prompts belong in this folder: no caption injected by another node, no extraction wrapper, no JSON envelope keyed to one sampler.
- The bundled Qwen-Image-2.1 bodies are adapted from [benjiyaya/ComfyUI-Qwen-Image-2.1-Prompt-Enhancer](https://github.com/benjiyaya/ComfyUI-Qwen-Image-2.1-Prompt-Enhancer) (MIT). Their upstream versions return JSON plus a ratio field; the local copies answer in plain text so the output can go straight into an encoder.
- A body is not trimmed: if it ends with a lead-in line such as "The user's edit instruction to rewrite is:", that line is part of the output by design and expects the user request on the following turn.
