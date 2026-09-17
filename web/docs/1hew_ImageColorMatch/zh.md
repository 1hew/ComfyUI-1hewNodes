# Image Color Match - 参考颜色迁移

**节点功能：** `Image Color Match` 将 `reference_image` 的颜色特征迁移到 `source_image` 上，支持多种颜色迁移算法。`wavelet` 与 `adain` 为原生实现；`mkl`、`hm`、`reinhard`、`mvgd` 及混合方法需要 `color-matcher` 包。支持批次输入，参考图帧数不一致时按 `i % ref_count` 循环取用。

## 输入

| 参数名称 | 入端选择 | 数据类型 | 默认值 | 取值范围 | 描述 |
| -------- | -------- | -------- | ------ | -------- | ---- |
| `source_image` | - | IMAGE | - | - | 需要重新着色的源图像 |
| `reference_image` | - | IMAGE | - | - | 提供目标颜色的参考图像 |
| `method` | - | COMBO | `mkl` | `wavelet` / `adain` / `mkl` / `hm` / `reinhard` / `mvgd` / `hm-mvgd-hm` / `hm-mkl-hm` | 颜色迁移算法。`wavelet`/`adain` 为原生实现，其余依赖 `color-matcher` |

## 输出

| 输出名称 | 数据类型 | 描述 |
|---------|----------|------|
| `image` | IMAGE | 重新着色后的图像批次；源图含 alpha 时保留该通道 |

## 功能说明

- 多种算法：`wavelet` 与 `adain` 为纯 PyTorch 实现（移植自 Easy-Use）；`mkl`、`hm`、`reinhard`、`mvgd`、`hm-mvgd-hm`、`hm-mkl-hm` 调用 `color-matcher` 包。
- 批次循环：单张参考图应用到全部帧；批次相等时逐帧对应；参考图较少时循环复用，较多时按源批次截断。
- alpha 保留：颜色迁移只在 RGB 上进行，源图的 alpha 通道保持不变。

## 典型用法

- 让生成图像的颜色对齐某张参考照片的色调。
- 在合成中让主体色调与背景保持一致。

## 注意与建议

- 除 `wavelet`/`adain` 外的方法需要安装 `color-matcher`（`pip install color-matcher`），否则节点会报错并提示安装。
- 参考帧按索引循环取用，与项目整体的广播/循环约定一致。
