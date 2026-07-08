# Text Compare - Text Comparison

**Node Purpose:** `Text Compare` performs comparison operations on two input strings and outputs a boolean result.

## Inputs

| Name | Port | Type | Default | Range | Description |
| ---- | ---- | ---- | ------- | ----- | ----------- |
| `a` | - | STRING | `""` | - | Left-side input string. |
| `operator` | - | COMBO | `==` | `==` / `!=` / `⊂` / `⊃` / `⊄` / `⊅` / `startswith` / `endswith` / `regex` | Comparison operator. |
| `b` | - | STRING | `""` | - | Right-side input string. |

## Outputs

| Name | Type | Description |
|------|------|-------------|
| `bool` | BOOLEAN | Boolean result of the comparison. |

## Operator Reference

| Operator | Meaning | Description |
|----------|---------|-------------|
| `==` | Equal | `a` equals `b` |
| `!=` | Not equal | `a` does not equal `b` |
| `⊂` | Subset of | `a` is a substring of `b` |
| `⊃` | Superset of | `b` is a substring of `a` |
| `⊄` | Not subset of | `a` is not a substring of `b` |
| `⊅` | Not superset of | `b` is not a substring of `a` |
| `startswith` | Starts with | `b` starts with `a` |
| `endswith` | Ends with | `b` ends with `a` |
| `regex` | Regex match | Search `a` with regex pattern `b` |

## Typical Usage

- Check if two strings match: `operator="=="`, compare the output of two prefix/suffix nodes.
- Detect keywords in prompts: `operator="⊂"`, `a` as the keyword, `b` as the prompt text.
- Regex-based conditional logic: `operator="regex"`, `b` as the regex pattern, `a` as the target text.

## Regex Quick Reference

| Scenario | `b` Pattern | `a` Example | Note |
|----------|-------------|-------------|------|
| Is numeric | `^\d+$` | `12345` | At least one digit |
| Is alphabetic | `^[A-Za-z]+$` | `Hello` | Letters only |
| Contains Chinese | `[\u4e00-\u9fff]` | `你好世界` | Any CJK character |
| Is image file | `(?i)\.(png\|jpe?g\|gif\|webp)$` | `photo.jpg` | Case-insensitive image extension |
| Is video file | `(?i)\.(mp4\|webm\|mov\|avi)$` | `movie.mp4` | Common video extensions |
| Strip version suffix | `^(.+?)(?:_v\d+)?$` | `image_v3.jpg` | Capture text before optional `_vN` |
| Is seed value | `^\d{1,10}$` | `4294967295` | 1 to 10 digits |
| NSFW keyword check | `NSFW\|nude\|naked` | `photo of people` | Multi-keyword with `\|` |
| Is aspect ratio | `^\d+:\d+$` | `16:9` | `n:n` format |
| Has lora trigger | `:\d+(\.\d+)?>` | `<lora:name:0.8>` | Detect lora weight syntax |

## Regex Quick Reference

| Scenario | `b` Pattern | `a` Example | Note |
|----------|-------------|-------------|------|
| Is numeric | `^\d+$` | `12345` | At least one digit |
| Is alphabetic | `^[A-Za-z]+$` | `Hello` | Letters only |
| Contains Chinese | `[\u4e00-\u9fff]` | `你好世界` | Any CJK character |
| Is image file | `(?i)\.(png\|jpe?g\|gif\|webp)$` | `photo.jpg` | Case-insensitive image extension |
| Is video file | `(?i)\.(mp4\|webm\|mov\|avi)$` | `movie.mp4` | Common video extensions |
| Strip version suffix | `^(.+?)(?:_v\d+)?$` | `image_v3.jpg` | Capture text before optional `_vN` |
| Is seed value | `^\d{1,10}$` | `4294967295` | 1 to 10 digits |
| NSFW keyword check | `NSFW\|nude\|naked` | `photo of people` | Multi-keyword with `\|` |
| Is aspect ratio | `^\d+:\d+$` | `16:9` | `n:n` format |
| Has lora trigger | `:\d+(\.\d+)?>` | `<lora:name:0.8>` | Detect lora weight syntax |

## Notes & Tips

- For `startswith` / `endswith`, `a` is the match fragment and `b` is the string being checked.
- In `regex` mode, an invalid pattern returns `False`.
- All inputs are converted to strings before comparison.
