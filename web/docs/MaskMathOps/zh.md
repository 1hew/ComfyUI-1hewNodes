# Mask Math Ops - 蒙版数学运算

**节点功能：** `Mask Math Ops`节点支持蒙版之间的相交、相加、相减、异或等数学运算操作，并支持批处理功能。

## 输入

| 参数名称 | 入端选择 | 数据类型 | 默认值 | 取值范围 | 描述 |
| -------- | -------- | -------- | ------ | -------- | ---- |
| `mask_1` | 必选 | MASK | - | - | 第一个输入蒙版 |
| `mask_2` | 必选 | MASK | - | - | 第二个输入蒙版 |
| `operation` | - | COMBO[STRING] | or | or, and, subtract (a-b), subtract (b-a), xor | 数学运算操作类型 |

## 输出

| 输出名称 | 数据类型 | 描述 |
|---------|----------|------|
| `mask` | MASK | 运算后的蒙版结果 |

## 功能说明

### 运算类型
#### 逻辑运算
- **or**：逻辑或运算，取两个蒙版的最大值
- **and**：逻辑与运算，取两个蒙版的最小值
- **xor**：异或运算，计算两个蒙版的绝对差值

#### 减法运算
- **subtract (a-b)**：从蒙版A中减去蒙版B
- **subtract (b-a)**：从蒙版B中减去蒙版A