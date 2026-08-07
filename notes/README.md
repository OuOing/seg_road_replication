# 道路提取论文学习笔记索引

本目录按论文分开放置笔记。根目录只保留总索引和跨论文路线，单篇论文的细节放入各自子目录。

## 目录结构

```text
notes/
  segroad_v1/   # 第一篇：Seg-Road
  segroad_v2/   # 第二篇：SegRoadV2
```

## 第一篇：Seg-Road v1

目录：`segroad_v1/`

| 顺序 | 笔记 | 内容 | 状态 |
| :--- | :--- | :--- | :--- |
| 0 | `segroad_v1/README.md` | 第一篇复习顺序和阶段性结论 | 已建立 |
| 1 | `segroad_v1/00_paper_overview.md` | 论文问题、整体架构、数据集与复现路线 | 已完成 |
| 2 | `segroad_v1/01_pcs.md` | PCS 标签生成、数组平移、PyTorch 实现、反向映射 | 已完成 |
| 3 | `segroad_v1/02_loss_and_training.md` | BCE、logits、联合损失、训练流程 | 已完成 |
| 4 | `segroad_v1/03_model_architecture.md` | Encoder-Decoder、多尺度特征、双分支输出 | 已完成 |
| 5 | `segroad_v1/04_srt_encoder.md` | SRA、MixFFN、多级 SRT Encoder | 已完成 |
| 6 | `segroad_v1/05_training_pipeline.md` | Dataset、训练循环、指标与评估脚本 | 已完成 |
| 7 | `segroad_v1/06_evaluation_and_experiments.md` | 指标、消融、公平比较、数据泄漏 | 已完成 |
| 8 | `segroad_v1/07_review_summary.md` | 第一篇复习速查 | 已整理为复习版 |
| 9 | `segroad_v1/09_training_journey.md` | 从背景塌缩到正式训练收口的完整实验复盘 | 已完成 |
| 10 | `segroad_v1/10_segroad_v1_closure.md` | 第一篇阶段性收口与过渡标准 | 已完成 |

第一篇当前最好本地结果：

```text
模型：Seg-Road-s
数据：DeepGlobe labeled train split，本地 4980/1246 划分
最好 checkpoint：runs/deepglobe/segroad-s-lr1e-4-finetune/best.pt
threshold：0.85
IoU：0.4779
F1：0.6468
Precision：0.6839
Recall：0.6135
```

注意：该结果是本地简化复现 baseline，尚未达到论文报告的 DeepGlobe IoU 约 67.20%。

## 第二篇：SegRoadV2

目录：`segroad_v2/`

| 顺序 | 笔记 | 内容 | 状态 |
| :--- | :--- | :--- | :--- |
| 0 | `segroad_v2/README.md` | 第二篇笔记索引 | 已完成 |
| 1 | `segroad_v2/00_paper_overview.md` | 论文动机、整体结构、与 v1 的关系 | 已完成 |
| 2 | `segroad_v2/01_module_map.md` | DSA、GroupDCN、条带卷积、PCS 的模块地图 | 已完成 |
| 3 | `segroad_v2/02_dsa.md` | DSA 从 SRA 升级而来的直觉和公式 | 已完成 |
| 4 | `segroad_v2/03_groupdcn.md` | 普通卷积、DCN、DCNv3 与 GroupDCN | 已完成 |
| 5 | `segroad_v2/04_strip_convolution.md` | 条带卷积、Conv-BN 融合与重参数化 | 已完成 |
| 6 | `segroad_v2/05_pcs_loss_and_training.md` | PCS 推理、联合损失和三阶段训练 | 已完成 |
| 7 | `segroad_v2/06_review_summary.md` | SegRoadV2 复习与对比速查总览 | 已归纳收口 |

## 第三篇：DiffRoad (进行中 🚀)

目录：`diffroad/`

| 顺序 | 笔记 | 内容 | 状态 |
| :--- | :--- | :--- | :--- |
| 0 | `diffroad/README.md` | 第三篇笔记索引与学习路线 | 已建立 |
| 1 | `diffroad/00_paper_overview.md` | 条件去噪扩散模型原理、动机与与前两篇的演进关系 | 核心原理进行中 |

第二篇学习主线：

```text
SRA 固定采样        -> DSA 可变形注意力
普通卷积固定网格    -> GroupDCN 可变形局部采样
普通 decoder 卷积   -> 可重参数化条带卷积
PCS                -> 继续保留，强化连通性
```

## 后续论文路线

参考论文已存放在本地 `docs/` 目录下，并按项目约定通过 `.gitignore` 忽略，不上传到远程仓库。为了方便在其他设备上下载或在线阅读，这里提供了每篇论文的官方发表/在线阅读链接：

1. **Seg-Road (v1)**: [MDPI Remote Sensing (2023)](https://www.mdpi.com/2072-4292/15/6/1602)
   * 本地文件：`docs/Seg-Road A Segmentation Network for Road Extraction Based on Transformer and CNN with Connectivity Structures.pdf`
2. **SegRoadV2**: [Taylor & Francis - International Journal of Digital Earth (2025)](https://www.tandfonline.com/doi/full/10.1080/17538947.2025.2458425)
   * 本地文件：`docs/SegRoadv2 - hybrid deformable self-attention and convolutional network for road extraction with connectivity structure.pdf`
3. **DiffRoad**: [IEEE Transactions on Geoscience and Remote Sensing (2026)](https://ieeexplore.ieee.org/document/10444004) / [ResearchGate](https://www.researchgate.net/publication/378418047_DiffRoad_A_Conditional_Diffusion-Based_Network_for_Accurate_Road_Extraction)
   * 本地文件：`docs/DiffRoad - A Conditional Diffusion-Based Network for Accurate Road Extraction.pdf`
4. **FDMamba**: [Taylor & Francis - International Journal of Digital Earth (2026)](https://www.tandfonline.com/doi/full/10.1080/17538947.2026.2626262)
   * 本地文件：`docs/FDMamba - frequency-enhanced deformable Mamba for topology-aware road extraction.pdf`

学习顺序：

```text
Seg-Road v1：已完成阶段性 baseline 收口
-> SegRoadV2：当前进行精读与核心模块解析 (当前阶段)
-> DiffRoad：条件去噪、DSA-FPN、PAGDecoder、拓扑指标
-> FDMamba：频域学习、可变形扫描、Mamba/SSM、图级拓扑
```

跨论文路线见 `08_future_paper_path.md`。

## 记录规则

* 单篇论文内容放入对应子目录。
* 算法主线写入主题笔记，临时训练流水只写入 journey/closure。
* Python、NumPy、PyTorch 语法只在首次出现时补充。
* 不使用当前环境无法稳定渲染的 LaTeX 分隔符。
* 公式使用反引号或 `text` 代码块。
* 每完成一个主题，更新本文件和对应论文子目录的 README。
