# Detect Remove SolidBG UnMult - 纯色背景 UnMult 抠图

**节点功能：** `Detect Remove SolidBG UnMult` 面向纯色背景素材：自动从图像边缘估计背景色，做通道级 UnMult 反解得到 alpha，并可用 `threshold` 清理低 alpha、用 `background_color` 把结果合成到指定底色；同时输出前景图与 alpha 遮罩。自动背景估计与抖动开关按固定策略写死（自动背景开、抖动关），无需额外参数。

## 输入

| 参数名称 | 入端选择 | 数据类型 | 默认值 | 取值范围 | 描述 |
| -------- | -------- | -------- | ------ | -------- | ---- |
| `image` | 必填 | IMAGE | - | - | 输入图像批次，支持 1 / 2 / 3 / 4 通道 |
| `threshold` | - | INT | `0` | `0-255` | 8bit alpha 黑场：`0` 不做清理；数值越大越压制低 alpha，并把剩余 alpha 重映射到 `[0,1]`；`255` 输出全透明 |
| `background_color` | - | STRING | 空 | 颜色名 / HEX / RGB / `sm` / 留空 | 后处理底色。留空保留透明 RGBA；`sm` 按每张图智能挑选黑 / 白 / `#808080`；其它值走 `Image Alpha Split` 的颜色语法 |

## 输出

| 输出名称 | 数据类型 | 描述 |
|---------|----------|------|
| `image` | IMAGE | 反解后的前景图：留空时为 RGBA，指定底色时为合成后的 RGB |
| `mask` | MASK | 0~1 浮点 alpha 遮罩（已乘输入 alpha 并应用 `threshold`） |

## 功能说明

- 背景估计：截取图像四周的边缘条带，按量化后的联合 RGB 聚类投票，选出空间支持度最高的颜色作为该图背景色，逐图独立估计。
- UnMult 反解：按“背景偏向白/黑”选择单边或双边通道反解，估计 alpha 后做色域安全的反预乘，避免出现负值或过曝。
- 白色背景增益：当估计背景接近纯白时，对 alpha 施加一个随中位数自适应、上限 2.5 的增益，改善半透明边缘。
- `threshold` 黑场：对 matte alpha 做 `(a - t) / (1 - t)` 重映射，低于阈值的像素 alpha 归零，同时保留输入自带的 alpha。
- 后处理合成：`background_color` 只影响最终输出图，不参与背景估计；`sm` 会调用 `Image Alpha Split` 的智能底色选择。

## 典型用法

- 纯色/摄影棚背景去背：直接连入 `image`，`threshold` 从 `0` 逐步提高以压掉边缘灰雾。
- 需要透明通道继续合成：`background_color` 留空，取 `image`（RGBA）和 `mask`。
- 需要立即得到可用底色：`background_color` 填 `sm` 或 `white` 等，直接输出 RGB。

## 注意与建议

- 该节点不做模型推理，不下载任何权重，全部为张量运算。
- 背景假设是“单一纯色”，复杂渐变或多色背景的边缘与透明区域可能出现误差，可用 `threshold` 或 `background_color` 缓解。
- 输入本身已带 alpha 时同样适用：输入 alpha 会参与最终 mask 计算。
- `background_color` / `sm` 的语法与 `Image Alpha Split` 完全一致，两个节点可互换使用。
