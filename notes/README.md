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

下一阶段继续 `04_srt_encoder.md`，结合 `code/srt.py` 学习：

```text
Softmax 和 Attention 权重的具体计算
MixFFN 的逐行代码细节
SRTEncoderStage 的运行验证
训练数据如何进入四级 SRT Encoder
```

## 记录规则

*   算法主线写入对应主题笔记；
*   Python / NumPy / PyTorch 语法只在首次出现时补充；
*   不使用当前环境无法稳定渲染的 LaTeX 分隔符；
*   公式使用反引号或 `text` 代码块；
*   每完成一个主题，更新本文件中的状态。
