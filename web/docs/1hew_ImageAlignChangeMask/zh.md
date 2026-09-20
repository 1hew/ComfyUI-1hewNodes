# Image Align Change Mask（对齐差异遮罩）

比较**原图**与 AI 编辑/局部重绘后的**编辑图**，自动校正轻微位移、旋转、缩放与全局色漂，提取模型真正修改的区域，输出对齐图像、原始遮罩、清理遮罩、变化热图和叠加预览。

## 适用场景

- Flux Kontext、Qwen Image、GPT Image 等编辑模型在局部修改之外，令全图产生轻微颜色、纹理或 1～数像素位移/缩放；
- 想把编辑图中真正变化的内容回贴到原图；
- 想根据模型**实际修改的位置**自动生成下一次 Inpaint 的遮罩。

> 两张图必须表示相同画布、相同主体，宽高必须一致。它无法可靠地区分“整图重绘/整图换风格”和“局部修改”。

## 输入

| 输入 | 说明 |
|---|---|
| `edit_image` | AI 编辑/局部重绘的输出。宽高必须与 `org_image` 完全一致。 |
| `org_image` | 编辑前的原图。 |
| `mode` | `auto`（默认）自动考评 none / translation / similarity；`none` 不对齐；`translation` 仅平移；`similarity` 支持受限平移、旋转和等比缩放。手动模式在未改善背景残差时也会安全回退到 `none`。 |
| `max_offset` | 最大允许平移范围。通常 4～8 px；超出会拒绝候选，防止把真实几何编辑错误地扭回去。 |
| `max_rotation` | Similarity 最大旋转角。建议保持 1～2°。 |
| `max_scale` | 最大等比缩放变化，`0.03` 表示允许 0.97～1.03。 |
| `min_align_score` | 对齐候选最低置信度，不足时安全回退。 |
| `color_compensation` | 默认开启。仅用可信未变化像素拟合每通道 gain/bias 以减轻全局轻微色偏；若全图调色本身也算修改，请关闭。 |
| `sensitivity` | 修改灵敏度。**越低越敏感**（遮罩越大），越高越干净。默认 `8.0`。 |
| `min_component_area` | Clean Mask 保留的最小连通区域像素面积。小文字、睫毛等小修改被删掉时调低。 |
| `expand` | Clean Mask 向外扩张的像素量，为二次重绘提供上下文。 |
| `feather` | Clean Mask 的羽化像素量，便于自然合成；设为 `0` 得到二值遮罩。 |

## 输出

| 输出 | 说明 |
|---|---|
| `align_image` | 已对齐到原图坐标系的编辑图。后续合成请使用它，而不是原始的 `edit_image`。 |
| `clean_mask` | 经过去噪、小区域过滤、扩张和羽化的软遮罩，适合 Inpaint、回贴和融合。 |
| `raw_mask` | 高召回的二值变化遮罩，适合检查模型实际修改了什么。 |
| `change_score` | 0～1 连续变化热图，用于诊断和自行设定阈值。 |
| `overlay_image` | 原图上以红色标出 Clean Mask 的预览。 |

## 推荐工作流

```text
原图 ──────────────┐
                  ├─ Image Align Change Mask ─ clean_mask ─ Inpaint / Composite
编辑图 ────────────┘                            ├─ align_image ─┐
                                                └─ overlay_image ─ Preview
```

### 将修改内容合成回原图

```text
org_image ──────────────┐
                        ├─ Image Align Change Mask
edit_image ─────────────┘
                           ├─ align_image
                           └─ clean_mask

org_image ────────────────┐
align_image ───────────────┼─ Composite Masked → 最终图
clean_mask ────────────────┘
```

## 参数预设

### 预设 A：一般局部编辑

```text
mode: auto
max_offset: 8
max_rotation: 2
max_scale: 0.03
min_align_score: 0.80
color_compensation: true
sensitivity: 8.0
min_component_area: 32
expand: 16
feather: 8
```

### 预设 B：检测小文字/五官/饰品

```text
sensitivity: 5.0
min_component_area: 4
expand: 4
feather: 1
```

### 预设 C：只保留明显的大物体修改

```text
sensitivity: 12.0
min_component_area: 200
expand: 12
feather: 6
```

## 排查顺序

1. 先查看 `align_image` 与 `overlay_image`，确认不是真实编辑被错误配准；
2. 若对象边缘一圈都被选中，检查是否存在缩放/旋转，可改用 `auto` 或 `similarity`；
3. 遮罩包含过多微弱模型噪声时，提高 `sensitivity`，并提高 `min_component_area`；
4. 遮罩漏掉细小但真实的修改时，降低 `sensitivity`、降低 `min_component_area`；
5. 如果“全图调色”也应算作修改，关闭 `color_compensation`。

## 算法与边界

`auto` 模式建立 identity、translation 和 similarity 候选。Translation 使用 Hann window phase correlation 初始化与 ECC 精修；Similarity 使用 AKAZE（不可用时 SIFT/ORB）特征、RANSAC 与 ECC，并将变换投影为无剪切的等比缩放+旋转。候选必须满足位移/旋转/缩放/置信度限制，更复杂模式还必须在排除高残差局部编辑后的稳健梯度考评上显著优于简单模式。随后执行可信背景颜色补偿、Lab/梯度残差融合、双阈值连通和保守清理。

节点**不使用默认光流**：稠密光流可能把新增、删除或重绘的真实内容扭回原图。对于大幅重构、视角变化、遮挡、无纹理画面或整图风格迁移，没有算法能仅由两张图精确推断“作者意图”；请使用 `change_score`、`overlay_image` 和 `align_image` 人工复核。
