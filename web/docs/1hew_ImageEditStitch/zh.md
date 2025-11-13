# Image Edit Stitch（图像编辑拼接）

将参考图与编辑图进行拼接（可选编辑掩码），支持位置、间隔条与颜色解析配置。输出合成后的图像，以及与输出对齐的两张掩码。

## 输入

| 名称 | 类型 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- | --- |
| reference_image | IMAGE | 是 | - | 参考图像，作为拼接基准，支持批量。 |
| edit_image | IMAGE | 是 | - | 编辑图像，拼接到参考图的一侧，支持批量。 |
| edit_image_position | STRING(enum) | 是 | right | 编辑图的拼接位置：right、left、top、bottom。 |
| match_edit_size | BOOLEAN | 是 | false | 若为 true，则参考图按编辑图尺寸进行带填充的等比缩放，最终尺寸与编辑图一致；若为 false，仅沿拼接轴保持等比对齐。 |
| spacing | INT | 是 | 0 | 两图之间的间隔条宽/高，单位像素。0 表示不插入间隔条。 |
| spacing_color | STRING | 是 | "1.0" | 间隔条颜色，支持高级颜色字符串（见下）。 |
| pad_color | STRING | 是 | "1.0" | 当 match_edit_size 为 true 时，用于缩放填充的颜色，支持高级颜色字符串。 |
| edit_mask | MASK | 否 | - | 编辑图对应的掩码（与 edit_image 对齐）。缺省时，编辑侧视为全 1。 |

## 输出

| 名称 | 类型 | 说明 |
| --- | --- | --- |
| image | IMAGE | 合成后的拼接图像（含可选间隔条），支持批量。 |
| mask | MASK | 与输出图像对齐的掩码，标记编辑侧区域为 1，其余（含间隔条）为 0，支持批量。 |
| split_mask | MASK | 与输出图像对齐的分区掩码：参考侧为 0、编辑侧为 1；间隔条为 0，支持批量。 |
