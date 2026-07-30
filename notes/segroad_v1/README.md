# Seg-Road v1 笔记索引

本目录只放第一篇 Seg-Road 的笔记。

## 复习优先级

如果只是复习，优先看：

```text
07_review_summary.md        # 总复习速查，最重要
10_segroad_v1_closure.md    # 当前实验结论和为什么能过渡到 v2
04_srt_encoder.md           # SRA / MixFFN / SRTBlock
01_pcs.md                   # PCS 连通性监督
02_loss_and_training.md     # BCE、Dice、logits、loss
```

如果要查完整实验过程，再看：

```text
09_training_journey.md      # 从背景塌缩到 finetune 的完整过程
05_training_pipeline.md     # 训练脚本、Dataset、checkpoint
06_evaluation_and_experiments.md # 指标、消融、数据划分
```

## 当前阶段性结论

当前本地最优结果：

```text
checkpoint：runs/deepglobe/segroad-s-lr1e-4-finetune/best.pt
threshold：0.85
IoU：0.4779
F1：0.6468
Precision：0.6839
Recall：0.6135
```

这不是论文数值达标复现。论文报告 SegRoadv1 在 DeepGlobe 上 IoU 约 67.20%，我们当前实现仍有明显差距。

但它已经足够作为学习和后续 SegRoadV2 的 baseline，因为我们已经完成：

```text
背景塌缩诊断
weighted BCE + Dice + GroupNorm 修正
完整 threshold sweep
PCS alpha=0 消融
低学习率短程续训
summary 结果表
```

## 复习目标

复习第一篇时，不要求背代码。要能回答：

```text
Seg-Road 为什么需要 PCS？
SRA 和普通 attention 有什么区别？
MixFFN 为什么加 depthwise conv？
为什么普通 BCE 会背景塌缩？
pos_weight、alpha、threshold 分别控制什么？
为什么 Accuracy 在道路提取里会骗人？
我们当前复现和论文结果差距可能来自哪里？
为什么这些问题自然引出 SegRoadV2？
```
