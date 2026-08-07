# SegRoadV2 笔记索引

本目录只记录第二篇论文 SegRoadV2，不和 Seg-Road v1 笔记混放。

论文：`SegRoadv2: a hybrid deformable self-attention and convolutional network for road extraction with connectivity structure`

## 学习顺序

| 顺序 | 笔记 | 内容 | 状态 |
| :--- | :--- | :--- | :--- |
| 0 | `00_paper_overview.md` | 论文动机、整体结构、与 v1 的关系 | 已完成 |
| 1 | `01_module_map.md` | DSA、GroupDCN、条带卷积、PCS 的模块地图 | 已完成 |
| 2 | `02_dsa.md` | DSA 从 SRA 升级而来的直觉和公式 | 已完成 |
| 3 | `03_groupdcn.md` | 普通卷积、DCN、DCNv3 与 GroupDCN | 已完成 |
| 4 | `04_strip_convolution.md` | 条带卷积、Conv-BN 融合与重参数化 | 已完成 |
| 5 | `05_pcs_loss_and_training.md` | PCS 推理、联合损失和三阶段训练 | 已完成 |
| 6 | `06_review_summary.md` | SegRoadV2 复习与对比速查总览 | 已归纳收口 |

## 学习主线

SegRoadV2 不是推翻 Seg-Road v1，而是针对 v1 暴露出的道路提取痛点做结构升级：

```text
SRA 固定采样        -> DSA 可变形注意力
普通卷积固定网格    -> GroupDCN 可变形局部采样
普通 decoder 卷积   -> 可重参数化条带卷积
PCS                -> 继续保留，强化连通性
```

学习时优先问：

```text
这个模块解决 v1 的哪个问题？
它改变了输入输出形状，还是只改变了采样方式？
它提升精度、速度，还是连通性？
如果复现，最小可实现版本是什么？
```
