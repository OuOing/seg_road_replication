# Seg-Road 学习笔记索引

本目录按论文模块组织学习笔记，不再按对话日期拆分。新知识优先合并到对应主题；只有无法归入现有主题时才新增文件。

## 学习顺序与状态

| 顺序 | 笔记 | 内容 | 状态 |
| :--- | :--- | :--- | :--- |
| 0 | `00_paper_overview.md` | 论文问题、整体架构、数据集与复现路线 | 已建立总览 |
| 1 | `01_pcs.md` | PCS 标签生成、数组平移、PyTorch 实现、反向映射 | 已完成核心学习 |
| 2 | `02_loss_and_training.md` | BCE、logits、联合损失、`SegRoadLoss` 执行流程 | 已完成基础学习 |
| 3 | `03_model_architecture.md` | Encoder-Decoder、多尺度特征、双分支输出、模型配置 | 已完成整体结构 |
| 4 | `04_srt_encoder.md` | SRA、MixFFN、多级 SRT Encoder | 已完成基础结构，待深入代码 |
| 5 | `05_training_pipeline.md` | Dataset、训练循环、指标与评估脚本 | 最小版本与 Smoke Test 已通过 |
| 6 | `06_evaluation_and_experiments.md` | 指标总结、消融实验、公平比较与数据泄漏 | 已完成基础学习 |
| 7 | `07_review_summary.md` | 全流程复习速查：模型、损失、训练、推理与实验 | 已完成整理 |
| 8 | `08_future_paper_path.md` | SegRoadV2、DiffRoad、FDMamba 的进阶学习与实验路线 | 已建立路线 |
| 9 | `09_training_journey.md` | 从背景塌缩、诊断修正到全量 8 轮结束的完整实验复盘 | 已完成记录 |

## 后续论文路线

参考论文均已放在 `docs/`，并按项目约定被 `.gitignore` 忽略，不上传到远程仓库：

```text
docs/SegRoadv2 - hybrid deformable self-attention and convolutional network for road extraction with connectivity structure.pdf
docs/DiffRoad - A Conditional Diffusion-Based Network for Accurate Road Extraction.pdf
docs/FDMamba - frequency-enhanced deformable Mamba for topology-aware road extraction.pdf
```

学习顺序暂定为：

```text
完成 Seg-Road v1 的数据、训练、指标和复现闭环
-> SegRoadV2：可变形注意力、GroupDCN、条带卷积
-> DiffRoad：条件扩散、DSA-FPN、PAGDecoder、拓扑指标
-> FDMamba：频域学习、可变形扫描、SSM/Mamba、图级拓扑
```

详细路线见 `08_future_paper_path.md`。三篇论文当前状态：**已归档并完成摘要级定位，待当前正式基线闭环后依次精读**。

## 当前知识闭环

```text
真实 road_mask
  -> code/pcs.py
  -> pcs_target

输入 image
  -> code/model.py
  -> seg_pred, pcs_pred

预测与标签
  -> code/loss.py
  -> total_loss
```

## 下一步

下一阶段继续完成第一篇论文的正式实验闭环：

```text
全量 threshold sweep 并保存 CSV
低学习率短程续训
PCS 消融实验
统一生成 summary.md 结果表
补充推理速度统计
开始 SegRoadV2 精读
```

DeepGlobe 已完成固定的 4980/1246 train/validation 划分。全量 8 轮试跑的最佳 validation IoU 为 0.3776，最佳检查点保存在 `runs/deepglobe/segroad-s-full-probe/best.pt`；6 个固定验证样本的预测对照图已经生成。

训练全过程的现象、假设、排查实验、修正依据和逐轮趋势统一记录在 `09_training_journey.md`。

正式实验命令配方统一记录在 `../experiments/README.md`，包括 threshold sweep、低学习率短程续训和 PCS 消融。

## 记录规则

*   算法主线写入对应主题笔记；
*   Python / NumPy / PyTorch 语法只在首次出现时补充；
*   不使用当前环境无法稳定渲染的 LaTeX 分隔符；
*   公式使用反引号或 `text` 代码块；
*   每完成一个主题，更新本文件中的状态。
