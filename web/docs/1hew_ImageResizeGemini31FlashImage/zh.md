# Image Resize Gemini31FlashImage - Gemini 3.1 Flash 图像尺寸适配

**节点功能：** `Image Resize Gemini31FlashImage` 继承 Gemini 3.0 Pro 版本的缩放逻辑，并扩展了更宽比例范围（包含 `0.5k` / `1k` / `2k` / `4k` 档位）。适用于 Gemini 3.1 Flash 任务前的图像与遮罩尺寸标准化。

## 输入

| 参数名称 | 入端选择 | 数据类型 | 默认值 | 取值范围 | 描述 |
| -------- | -------- | -------- | ------ | -------- | ---- |
| `preset_size` | - | COMBO | `auto (2k \| 4k)` | `auto` / `auto (0.5k)` / `auto (1k \| 2k)` / `auto (2k \| 4k)` / 各预设分辨率项 | 目标尺寸选择；`auto (0.5k)` 会优先在 `[512]` 预设中自动匹配 |
| `fit` | - | COMBO | `crop` | `crop` / `pad` / `stretch` | 适应模式：裁剪、填充、拉伸 |
| `pad_color` | - | STRING | `1.0` | 灰度/HEX/RGB/颜色名/`edge`/`average`/`extend`/`mirror` | `pad` 模式背景填充策略 |
| `image` | 可选 | IMAGE | - | - | 输入图像批次 |
| `mask` | 可选 | MASK | - | - | 输入遮罩批次 |

## 输出

| 输出名称 | 数据类型 | 描述 |
|---------|----------|------|
| `image` | IMAGE | 调整后的图像批次 |
| `mask` | MASK | 与输出尺寸匹配的遮罩批次 |

## 功能说明

- 扩展预设集：比 Gemini30 版本新增更多超宽/超高比例组合与 `0.5k` 自动档位。
- 自动匹配逻辑：`auto*` 模式下按输入宽高比和面积匹配最近预设。
- 适配策略一致：`crop` / `pad` / `stretch` 与 Gemini30 版本行为一致，遮罩同步输出。
- 输入弹性：支持 image-only、mask-only、无输入三种场景。

## 典型用法

- 低分辨率快速预处理：`preset_size=auto (0.5k)`。
- 覆盖更大尺寸范围：`preset_size=auto (2k | 4k)`，结合 `fit=crop`。
- 保留完整主体：`fit=pad`，并设置合适的 `pad_color`。

## 注意与建议

- 若追求速度可优先使用 `auto (0.5k)` 或 `auto (1k | 2k)`。
- 若追求细节保留可使用更高档位并避免过强拉伸（`stretch`）。
