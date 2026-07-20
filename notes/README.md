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

## 第二篇论文：SegRoadv2

参考论文已放在：

```text
docs/SegRoadv2 - hybrid deformable self-attention and convolutional network for road extraction with connectivity structure.pdf
```

该 PDF 按项目约定被 `.gitignore` 忽略，不上传到远程仓库。学习顺序暂定为：

```text
先完成 Seg-Road v1 的数据、训练、指标和复现闭环
-> 再阅读 SegRoadv2 的整体动机和方法改进
-> 对比普通 SRA 与 deformable self-attention
-> 对比两版网络的卷积、解码器和 connectivity structure
-> 最后整理可迁移到 Agent 开发的论文阅读方法
```

第二篇论文当前状态：**已归档，排队学习，尚未开始精读**。

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

下一阶段继续第一篇论文，结合真实数据和实验学习：

```text
真实数据集整理与无泄漏划分
正式训练和测试集评估
PCS 消融实验
参数量、FLOPs 和速度统计
```

训练工程已通过合成数据 Smoke Test。下一步需要整理真实数据集，并按原始大图或地理区域划分训练、验证和测试数据，避免相邻 patch 泄漏。

## 记录规则

*   算法主线写入对应主题笔记；
*   Python / NumPy / PyTorch 语法只在首次出现时补充；
*   不使用当前环境无法稳定渲染的 LaTeX 分隔符；
*   公式使用反引号或 `text` 代码块；
*   每完成一个主题，更新本文件中的状态。
