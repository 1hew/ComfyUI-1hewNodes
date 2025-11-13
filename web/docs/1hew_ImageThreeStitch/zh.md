# Image Three Stitch（三图拼接）

将 image_2 与 image_3 先合并为一对，再按指定方向将该对拼接到 image_1。支持间隔条与颜色解析，并提供尺寸匹配或填充两种策略。

## 输入

| 名称 | 类型 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- | --- |
| image_1 | IMAGE | 是 | - | 主图像，最终组合的附着对象。 |
| image_2 | IMAGE | 是 | - | 二者中的第一个，先与 image_3 合并为一对。 |
| image_3 | IMAGE | 是 | - | 二者中的第二个，与 image_2 先合并为一对。 |
| direction | STRING(enum) | 是 | left | 该对相对 image_1 的拼接方向：top、bottom、left、right。 |
| match_image_size | BOOLEAN | 是 | true | 若为 true，则沿拼接轴对齐尺寸并保持长宽比；若为 false，则不缩放，通过 pad_color 进行填充对齐。 |
| spacing_width | INT | 是 | 10 | 间隔条宽度，既用于二者之间，也用于该对与 image_1 之间。 |
| spacing_color | STRING | 是 | "1.0" | 间隔条颜色，支持颜色字符串（见下）。 |
| pad_color | STRING | 是 | "1.0" | 当 match_image_size 为 false 时，用于填充对齐的颜色。 |

## 输出

| 名称 | 类型 | 说明 |
| --- | --- | --- |
| image | IMAGE | 三图合成的拼接结果，可包含间隔条。 |
