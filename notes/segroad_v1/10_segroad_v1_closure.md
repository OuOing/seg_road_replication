# Seg-Road v1 阶段性收口报告

这份报告用于判断 Seg-Road v1 是否已经足够收口，可以过渡到 SegRoadV2。它不是最终论文复现报告，而是把当前 baseline、待完成实验和过渡理由放在同一页，避免后续学习时来回翻训练日志。

## 1. 当前已确认结果

数据与划分：

```text
Dataset：DeepGlobe Road Extraction public labeled train split
Train：4980
Validation：1246
Input：512 x 512
Device：Apple MPS for training
Model：Seg-Road-s
Parameters：3.979M
```

当前最佳 checkpoint：

```text
runs/deepglobe/segroad-s-full-probe/best.pt
epoch：8
threshold：0.5
IoU：0.3776
F1：0.5482
Precision：0.4128
Recall：0.8155
pred+：0.0795
target+：0.0402
```

完整 validation threshold sweep 后，验证集最优阈值为 `0.85`：

```text
threshold：0.85
IoU：0.4460
F1：0.6169
Precision：0.6033
Recall：0.6311
pred+：0.0421
target+：0.0402
```

核心结论：

```text
模型已经摆脱全背景塌缩
能够识别主要道路结构
Recall 较高，说明多数道路像素被找回
Precision 偏低，说明误检仍多
pred+ 约为 target+ 的两倍，说明预测道路偏粗
调高 threshold 到 0.85 后，pred+ 接近 target+，Precision 明显改善
```

## 2. 第一篇还差的最小收尾实验

### A. 完整 validation threshold sweep

目标：确定默认 `threshold=0.5` 是否偏宽松，并选择 validation 上更合理的后处理阈值。

状态：已完成。完整结果保存在：

```text
runs/deepglobe/segroad-s-full-probe/threshold_sweep_val.csv
```

产物：

```text
runs/deepglobe/segroad-s-full-probe/threshold_sweep_val.csv
runs/deepglobe/segroad-s-full-probe/summary.md
```

判断方式：

```text
0.85 的 IoU/F1 最高
0.5 明显偏宽松，pred+ 是 target+ 的约两倍
0.85 下 pred+ 接近 target+，Precision 从 0.4128 提升到 0.6033
代价是 Recall 从 0.8155 降到 0.6311
```

### B. PCS 消融

目标：验证 PCS 连通性监督在当前实现中是否有实际贡献。

状态：已完成。`--pcs-alpha 0` 的最佳 checkpoint 为 epoch 7：

```text
runs/deepglobe/segroad-s-no-pcs/best.pt
threshold：0.5
IoU：0.3729
F1：0.5432
Precision：0.4128
Recall：0.7942
pred+：0.0774
target+：0.0402
```

产物：

```text
runs/deepglobe/segroad-s-no-pcs/best.pt
runs/deepglobe/segroad-s-no-pcs/summary.md
```

判断方式：

```text
默认 threshold=0.5 下，no-pcs 略低于 baseline：
  baseline IoU/F1：0.3776 / 0.5482
  no-pcs IoU/F1：0.3729 / 0.5432

no-pcs early training 更激进：
  epoch 1 pred+：0.2996
  baseline epoch 1 pred+：0.2211

到 best epoch 时，no-pcs 的 Precision 接近 baseline，但 Recall 略低。
下一步仍需可视化检查道路是否更容易断裂。
```

### C. 低学习率短程续训

目标：判断 epoch 8 后用更小学习率是否能减少误检、提高 Precision。

状态：已完成。从 epoch 8 best checkpoint 继续训练到 epoch 12，使用 `--resume-learning-rate 1e-4` 和 `--eval-threshold 0.85`。最佳结果出现在 epoch 12：

```text
runs/deepglobe/segroad-s-lr1e-4-finetune/best.pt
threshold：0.85
IoU：0.4779
F1：0.6468
Precision：0.6839
Recall：0.6135
pred+：0.0361
target+：0.0402
```

产物：

```text
runs/deepglobe/segroad-s-lr1e-4-finetune/best.pt
runs/deepglobe/segroad-s-lr1e-4-finetune/summary.md
```

判断方式：

```text
相对 epoch 8 threshold-selected baseline：
  IoU：0.4460 -> 0.4779
  F1：0.6169 -> 0.6468
  Precision：0.6033 -> 0.6839
  Recall：0.6311 -> 0.6135
  pred+：0.0421 -> 0.0361

结论：低学习率短程续训有效，主要提升 Precision 和 IoU/F1。
代价是 Recall 小幅下降，模型比 epoch 8 更保守。
```

## 3. 过渡到 SegRoadV2 的最低条件

满足下面三项即可过渡：

```text
1. 完整 threshold sweep 已完成（已完成）
2. PCS alpha=0 消融已完成（已完成）
3. baseline 与消融都有统一 summary.md（已完成）
```

低学习率续训和推理速度统计建议完成，但不是进入第二篇的硬门槛。

当前低学习率续训也已完成，因此第一篇已经可以阶段性收口。若继续补充，优先做可视化对比和推理速度，而不是继续无限追加 epoch。

## 4. 为什么这些结果自然引出 SegRoadV2

Seg-Road v1 当前暴露出的主要问题：

```text
道路预测偏粗，线状背景容易被误检
狭窄支路、遮挡道路仍可能漏检
固定 SRA 对弯曲道路的空间采样不够灵活
普通局部卷积只能看固定网格
PCS 关注连通性，但主干特征提取仍有改进空间
```

SegRoadV2 的几个核心模块正好对应这些问题：

```text
DSA：让注意力采样位置可变，适应弯曲道路
GroupDCN：让局部卷积采样不再固定，适应形变结构
条带卷积：强化细长道路的方向性建模
继续保留 PCS：说明连通性监督仍是道路提取主线
```

因此，第一篇收口的目的不是把指标追到论文最高分，而是建立一个可信 baseline，并清楚说明它为什么需要第二篇的改进。

## 5. 当前推荐执行顺序

```text
1. 跑完整 threshold sweep
2. 用 threshold_sweep_val.csv 重新生成 baseline summary.md
3. 跑 PCS alpha=0 消融
4. 生成 no-pcs summary.md
5. 可选：跑低学习率短程续训
6. 开始 SegRoadV2 精读
```
