# Image Tile Merge - 兼容接缝的切片合成

**节点功能：** `Image Tile Merge` 使用给定的 `tile_meta` 将由切片节点产出的 tile 重建为完整图像。节点会在重叠区域上应用基于余弦权重的平滑混合，`blend_strength` 用于控制接缝融合强度。

## 输入

| 参数名称 | 入端选择 | 数据类型 | 默认值 | 取值范围 | 描述 |
| -------- | -------- | -------- | ------ | -------- | ---- |
| `tile` | - | IMAGE | - | - | 切片批次或单张切片；支持 `B×H×W×3` 或 `H×W×3` |
| `tile_meta` | - | DICT | - | - | 来自切片节点的元数据字典，用于定位 tile 与还原布局 |
| `blend_strength` | - | FLOAT | 1.0 | 0.0-1.0 | 控制重叠区域权重混合强度；`0` 表示关闭平滑混合 |

## 输出

| 输出名称 | 数据类型 | 描述 |
|---------|----------|------|
| `image` | IMAGE | 还原后的完整图像 |

## 功能说明

- 切片数量处理：当 tile 数量少于或多于预期时，会自动补齐或裁掉，使其满足 `rows*cols` 的重建要求。
- 余弦权重掩模：根据 tile 所在位置与网格关系，在重叠边缘生成平滑的渐变权重。
- 加权合成：对所有 tile 做加权累积，再按总权重归一化，降低拼接缝可见度。
- 重叠感知：读取 `tile_meta` 中的 `overlap_width/height` 作为融合带宽度，并受 `blend_strength` 调节。

## 典型用法

- 切后合并：将 `Image Tile Split` 输出的 `tile` 与 `tile_meta` 直接传入本节点，并按需要调节 `blend_strength`（例如 `0.4-0.8`）以减轻接缝。
- 容错恢复：当上游 tile 数量略有偏差时，自动 trim/pad 可帮助保持网格结构完整。

## 注意与建议

- `tile_meta['tile_metas'][i]` 中通常包含 `crop_region`、`position (col,row)` 与 `actual_crop_size`，这些信息决定 tile 的放置位置与裁剪对齐方式。
- 重叠较大时，若希望边界更清晰可适当降低 `blend_strength`；需要更柔和时提高该值。
