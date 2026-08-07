# DiffRoad 笔记索引

本目录记录第三篇论文：`DiffRoad: A Conditional Diffusion-Based Network for Accurate Road Extraction` (IEEE TGRS 2026)。

## 学习顺序

| 顺序 | 笔记 | 内容 | 状态 |
| :--- | :--- | :--- | :--- |
| 0 | `README.md` | 第三篇笔记索引与学习路线 | 已建立 |
| 1 | `00_paper_overview.md` | 论文动机、整体结构、与 Seg-Road v1 / SegRoadV2 的演进关系 | 核心原理已完成 |

## 核心演进主线

```text
Seg-Road v1  (直接像素分割 + PCS)
  -> SegRoadV2 (可变形算子 DSA/GroupDCN + 条带卷积 + PCS)
  -> DiffRoad   (条件去噪扩散模型 + DSA-FPN + PAGDecoder)
```

## 学习重点问答

1. **范式转变**：为什么要把二值 mask 分割任务变成“噪声掩码（Noisy Mask）的条件去噪”过程？
2. **架构变化**：`DSA-FPN` 如何提取多尺度空间特征？`PAGDecoder` 如何通过扩散粗特征指导解码？
3. **拓扑评估**：除了常规像素指标 (IoU/F1)，如何引入 `APLS` 和 `TOPO-F1` 拓扑连通性指标？
