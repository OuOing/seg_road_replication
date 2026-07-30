# SegRoadV2 模块地图

这页用于先建立模块级地图，避免一上来陷入公式。

## 1. 模块总览

| 模块 | 放在哪里 | 解决什么问题 | 对应 v1 问题 |
| --- | --- | --- | --- |
| DSA | Encoder Transformer block | 全局注意力采样位置可变 | SRA 固定采样不够贴合弯曲道路 |
| GroupDCN | Encoder CNN/local branch | 局部卷积采样位置可变 | 普通卷积固定 3x3 网格 |
| Strip Conv | Decoder | 更适合细长道路形状 | 普通方形卷积会看进无关背景 |
| Re-parameterization | Inference stage | 保持训练表达力，加快推理 | 多分支结构推理成本高 |
| PCS | Output/auxiliary branch | 强化道路连通性 | 道路断裂、拓扑不稳定 |

## 2. DSA

DSA 是 Deformable Self-Attention。

v1 的 SRA：

```text
Q 固定
K/V 做空间缩减
Q 和 SR(K/V) 做 attention
```

v2 的 DSA：

```text
Q -> 预测 offset
根据 offset 重新采样 Q，得到 Q_new
Q_new 和 SR(K/V) 做 attention
```

直觉：

```text
SRA：站在原地看
DSA：先移动到更有用的位置再看
```

道路是弯曲、斜向、断续的，固定网格不一定正好落在道路结构上，所以可变形采样有价值。

## 3. GroupDCN

普通卷积：

```text
固定 3x3 采样点
```

Deformable Convolution：

```text
每个采样点都有可学习 offset
```

GroupDCN：

```text
按 group 做可变形采样
减少参数量和计算量
保留对道路形状的自适应能力
```

论文中 DCNv3 指标可能更强，但计算复杂度高；GroupDCN 是精度和效率之间的折中。

## 4. Strip Convolution

道路是细长目标，普通 `3x3` 或方形卷积不够贴合。

Strip convolution 使用长条卷积：

```text
1x13
13x1
斜向 strip conv
```

作用：

```text
沿道路方向聚合信息
减少邻近背景干扰
更容易捕捉细长连续结构
```

## 5. Re-parameterization

训练时：

```text
多分支 strip conv
表达力更强
```

推理时：

```text
把 Conv + BN 融合
把多个卷积分支合并
减少实际推理计算
```

一句话：

```text
训练时复杂一点，推理时合并变快
```

## 6. PCS

SegRoadV2 继续保留 PCS，这说明作者仍然认为道路提取不能只做像素分类，还要监督连通关系。

PCS 在 v2 中的角色：

```text
不是替代 DSA/GroupDCN
而是和更强 encoder/decoder 配合
继续约束道路连通性
```

## 7. 学习顺序建议

```text
先学 DSA，因为它从 v1 的 SRA 自然升级
再学 GroupDCN，因为它和 DSA 都是 deformable 思想
再学 Strip Conv，因为它对应道路细长形状
最后学 re-parameterization，因为它偏工程推理优化
```
