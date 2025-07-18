# Image List Append - 图片列表追加

**节点功能：** `Image List Append` 节点用于将两个图片输入追加为列表格式，支持图片批次的智能合并，常用于图片收集和批量处理工作流。

## 输入

| 参数名称 | 入端选择 | 数据类型 | 默认值 | 取值范围 | 描述 |
| -------- | -------- | -------- | ------ | -------- | ---- |
| `image_1` | 必选 | IMAGE | - | - | 第一个图片输入，可以是单张图片或图片批次 |
| `image_2` | 必选 | IMAGE | - | - | 第二个图片输入，可以是单张图片或图片批次 |

## 输出

| 输出名称 | 数据类型 | 描述 |
|---------|----------|------|
| `image_list` | IMAGE | 合并后的图片列表 |

## 功能说明

### 应用场景
- **图片收集**：将多个来源的图片合并为一个批次
- **批量处理**：为批量处理节点准备图片列表
- **工作流整合**：连接不同的图片处理分支