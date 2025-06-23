# Image Blend Modes By CSS - CSS图层混合模式

**节点功能：** `Image Blend Modes By CSS`节点实现CSS标准的混合模式进行图像合成，提供与Web兼容的混合效果和精确的颜色管理。

## 输入

| 参数名称 | 入端选择 | 数据类型 | 默认值 | 取值范围 | 描述 |
| -------- | -------- | -------- | ------ | -------- | ---- |
| `base_image` | 必选 | IMAGE | - | - | 基础图层图像 |
| `overlay_image` | 必选 | IMAGE | - | - | 叠加图层图像 |
| `blend_mode` | - | COMBO[STRING] | normal | CSS混合模式 | CSS标准混合模式 |
| `opacity` | - | FLOAT | 1.0 | 0.0-1.0 | 叠加图层的不透明度 |
| `overlay_mask` | 可选 | MASK | - | - | 用于选择性混合的可选遮罩 |
| `invert_mask` | - | BOOLEAN | False | True/False | 是否反转叠加遮罩 |

## 输出

| 输出名称 | 数据类型 | 描述 |
|---------|----------|------|
| `image` | IMAGE | CSS混合结果图像 |