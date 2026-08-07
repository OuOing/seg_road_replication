# DiffRoad 笔记索引

本目录记录第三篇论文：`DiffRoad: A Conditional Diffusion-Based Network for Accurate Road Extraction` (IEEE TGRS 2026)。

## 学习路线与进度

为了不一上来就陷入复杂的数学推导，我们按照“直觉概念 -> 编码器条件 -> 解码器生成 -> 拓扑评估 -> 三篇大一统”的递进顺序学习：

| 步骤 | 笔记 | 核心学习内容 | 状态 |
| :--- | :--- | :--- | :--- |
| 0 | `README.md` | 第三篇笔记索引与学习路线 | 已建立 |
| 1 | `00_paper_overview.md` | 论文动机、整体结构、与前两篇的范式演进 | 已完成 |
| 2 | `01_diffusion_basics.md` | 扩散模型基础、前向加噪、反向去噪与条件引导 (直觉大白话) | 已完成 |
| 3 | `02_dsa_fpn_encoder.md` | DSA-FPN 条件编码器：复用 v2 的 DSA 并融合特征金字塔 | 已完成 |
| 4 | `03_pag_decoder.md` | PAGDecoder 解码器：渐进式注意力引导与掩码生成 | 准备开启 🚀 |
| 5 | `04_topology_metrics.md` | 拓扑评估指标：突破传统 IoU，学习 APLS 与 TOPO-F1 | 待开始 |
| 6 | `05_review_and_comparison.md` | 三篇论文大一统复习总结与演进脉络收口 | 待开始 |

## 核心演进主线

```text
Seg-Road v1  (静态 SRA + 普通卷积 + 像素分类 + PCS 连通性)
  -> SegRoadV2 (可变形 DSA/GroupDCN + 重参数化条带卷积 + 像素分类 + PCS 连通性)
  -> DiffRoad   (条件去噪扩散范式 + DSA-FPN + PAGDecoder + 拓扑指标 APLS/TOPO)
```

## 学习重点问答

1. **范式转变**：为什么要把二值 mask 分割任务变成“噪声掩码（Noisy Mask）的条件去噪”过程？
2. **架构变化**：`DSA-FPN` 如何提取多尺度空间特征？`PAGDecoder` 如何通过扩散粗特征指导解码？
3. **拓扑评估**：除了常规像素指标 (IoU/F1)，如何引入 `APLS` 和 `TOPO-F1` 拓扑连通性指标？
