# String Coordinate to BBox Mask - 字符串坐标转边界框遮罩

**节点功能：** `String Coordinate to BBox Mask`节点将字符串格式的坐标列表转换为BBoxMask格式，支持多种输入格式，需要图像输入来获取宽高信息以生成准确的遮罩。

## 输入

| 参数名称 | 入端选择 | 数据类型 | 默认值 | 取值范围 | 描述 |
| -------- | -------- | -------- | ------ | -------- | ---- |
| `image` | 必选 | IMAGE | - | - | 用于获取尺寸信息的图像 |
| `coordinates_string` | 必选 | STRING | "" | 多行文本 | 坐标字符串，格式为"x1,y1,x2,y2"或"[x1,y1,x2,y2]"，支持多行坐标 |
| `output_mode` | - | COMBO[STRING] | merge | separate, merge | 输出模式：separate（每个坐标行单独输出遮罩），merge（所有坐标合并为一个遮罩） |

## 输出

| 输出名称 | 数据类型 | 描述 |
|---------|----------|------|
| `bbox_mask` | MASK | 基于坐标生成的边界框遮罩 |