# SegRoadV2 复习与归纳速查

这份文档用于快速复习 SegRoadV2 的核心算法机制、模块分工及与 Seg-Road v1 的对比。

## 1. Seg-Road v1 与 SegRoadV2 全方位对比

| 维度 | Seg-Road v1 | SegRoadV2 | 升级动机 / 解决什么问题 |
| :--- | :--- | :--- | :--- |
| **Encoder 全局注意力** | SRA (稀疏常规注意力) | DSA (可变形自注意力) | SRA 采样点固定在矩形网格，DSA 的采样点由偏移量 `offset` 动态移动，贴合弯曲道路 |
| **Encoder 局部卷积** | 普通固定卷积 (`3x3`) | GroupDCN (分组可变形卷积) | 普通卷积无法自适应道路宽度变化，GroupDCN 分组独立预测采样点偏置，适应复杂局部几何 |
| **Decoder 细节恢复** | 普通 CNN Decoder (`3x3`) | 可重参数化条带卷积 (Strip Conv) | 普通卷积引入过多背景噪声，`1x13` / `13x1` 多方向长条卷积核能顺着细长道路收集长程上下文 |
| **推理部署加速** | 无特殊重参数化 | 双层重参数化 (Conv-BN + 1D转2D) | 训练时使用 4 方向多分支提高特征丰富度，推理时融合成 2 个 `13x13` 算子，降低延迟与显存开销 |
| **连通性监督** | PCS (8方向邻域二值指示牌) | 保留 PCS 机制 | 保持多任务辅助监督，推理时与分割图取交集 (`Seg ∩ PCS`) 过滤孤立误检噪点 |
| **训练策略** | 固定类别权重与静态训练 | 前期 `3:1` 后期 `1:1` + 三阶段训练 | 前期加大道路损失权重防止背景塌缩，后期恢复 `1:1` 防止道路预测偏粗；三阶段逐级减小 Batch 与 LR 进行微雕 |

---

## 2. 核心模块一句话速记

```text
DSA (Deformable Self-Attention)：
  - 给全局注意力发了“会走路的脚”
  - 核心逻辑：给 Query 动态算 offset，只在马路骨架上采点计算上下文

GroupDCN (Groupable Deformable Convolution)：
  - 给局部卷积按通道分组定制“橡皮筋”
  - 核心逻辑：按 channel 组预测不同偏移，宽路、细路、斜路分通道自适应提取

Strip Convolution (条带卷积与重参数化)：
  - 细长马路用长条尺子量（1x13 & 13x1）
  - 训练用 4 角度旋转分支，部署时融合成 2 个 13x13 十字算子

PCS (Pixel Connectivity Structure)：
  - 8 方向邻域连通标志
  - 训练：Loss = L_seg + 0.2 * L_pcs
  - 推理：Seg Map ∩ PCS Map (保守交集过滤)
```

---

## 3. 规范化术语与算法描述

1. **重参数化算子合并公式**：
   * 训练态：`Out = Conv_1x13(x) + Conv_13x1(x) + ...`
   * 部署态：`W_fused = pad(W_h, (0,0,6,6)) + pad(W_v, (6,6,0,0))`
   * 结果：`Out = Conv_13x13_fused(x)`

2. **动态类别权重规则**：
   * `Epoch 0 ~ 30`: `pos_weight_seg = 3.0` (抑止背景塌缩，强迫找前景)
   * `Epoch 30 ~ 100`: `pos_weight_seg = 1.0` (精修边缘，提升 Precision)

3. **三阶段训练路线**：
   * `Phase 1 (Epoch 0-30)`: `Batch Size = 32`, `LR = 0.001` (大步走，快看大局)
   * `Phase 2 (Epoch 30-90)`: `Batch Size = 16`, `LR = 0.0001` (中步走，局部调整)
   * `Phase 3 (Epoch 90-100)`: `Batch Size = 4`, `LR = 固定低值` (微步走，拓扑微雕)

---

## 4. 论文 1 与 论文 2 学习收口标记

* **Seg-Road v1**: 基线建立完成，消融与后处理 Sweep 总结见 `notes/segroad_v1/10_segroad_v1_closure.md`。
* **SegRoadV2**: 原理精读、模块拆解、重参数化推导与训练协议总结见 `notes/segroad_v2/00_paper_overview.md` 至 `notes/segroad_v2/06_review_summary.md`。

---
