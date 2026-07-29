# DeepGlobe 正式训练全过程复盘

这份记录按真实发生顺序整理 Seg-Road 从第一次正式训练，到发现背景塌缩、逐项诊断、修正训练策略，再到完成 8 个 epoch 和预测可视化的全过程。重点不是只保留最好结果，而是说明每一步为什么做、证据是什么、结论如何影响下一步。

## 1. 实验起点：先建立可比较的正式基线

数据来自 DeepGlobe Road Extraction Dataset。公开镜像包含 6226 对有标签训练图像和 mask；官方 valid/test 只有图像，没有公开 mask，因此本地不能直接计算它们的 IoU/F1。

项目使用固定划分，避免每次运行随机改变验证集：

```text
train：4980
validation：1246
输入尺寸：512 x 512
模型：Seg-Road-s
设备：Apple MPS
```

真实道路像素只占约 4.02%，其余约 95.98% 都是背景。这一事实后来成为理解所有早期异常的关键。

第一次实验仍从普通 BCE 开始，因为正式实验需要一个未经补偿的 baseline，后续加权 BCE、Dice 和归一化修改才有对照对象。

## 2. 第一次异常：loss 下降，但 IoU/F1 始终为 0

普通 BCE、`lr=1e-4` 的前两轮日志：

```text
epoch 1：train loss 0.5706，val loss 0.4820，val IoU 0，F1 0
epoch 2：train loss 0.4080，val loss 0.3056，val IoU 0，F1 0
```

如果只看 loss，会误以为模型训练正常；但 IoU/F1 表明它没有提取任何道路。继续完成 20 个 epoch 后：

```text
val loss：0.1956
accuracy：0.9598
IoU / F1 / Precision / Recall：0
```

### 当时发生了什么

模型把全部像素都预测为背景。因为约 96% 像素本来就是背景，这种无用预测仍能获得约 96% Accuracy，也能让普通 BCE 持续下降。

### 得出的第一个结论

```text
loss 下降 != 模型学会目标
Accuracy 很高 != 道路提取有效
```

类别极不平衡时，必须同时观察 IoU、F1、Precision、Recall 和预测道路比例，不能用 Accuracy 作为主要判断标准。

## 3. 第一轮修正：用加权 BCE 强化道路像素

道路约占 4%，背景与道路的像素比约为 `22.5:1`。因此先给道路正样本更高损失权重：

```text
seg_pos_weight = 10
pcs_pos_weight = 10
learning_rate = 1e-4
```

这里没有直接使用理论比值 22.5，是因为权重过大可能使模型把大量背景误判为道路。

结果：训练到 epoch 5 仍然预测全背景。随后把学习率提高到 `1e-3`，模型曾短暂开始预测道路，但又退回背景。

### 得出的结论

类别权重是必要条件，但不是唯一问题。模型并非只缺少更大的道路梯度，还存在优化或归一化稳定性问题。

## 4. 阈值诊断：不是把 0.5 调低就能解决

对加权 BCE 的 checkpoint 检查 sigmoid 概率，发现输出接近空间常数：

```text
mean：约 0.316
max：约 0.343
threshold = 0.3：几乎全部像素变成道路
threshold = 0.4 / 0.5：几乎全部像素变成背景
```

如果模型已经学到道路结构，只是概率偏低，降低 threshold 应该能逐渐找回道路。但这里阈值稍微变化就从全背景跳到全道路，说明输出缺少空间区分能力。

### 得出的结论

阈值只能改变决策边界，不能把近似常数的概率图变成有道路形状的预测。此时继续调 threshold 没有意义。

## 5. 单图过拟合：验证数据、模型和反向传播链路

为了判断是代码链路错误，还是全量优化困难，选取一张真实 DeepGlobe 图像反复训练 100 步：

```text
IoU：约 0.38
Recall：约 0.95
predicted positive ratio：约 0.0425
```

模型能够记住单张图像，而且预测道路比例接近真实分布。

### 这一实验排除了什么

```text
图像和 mask 完全不匹配
PCS 标签无法生成
模型输出尺寸错误
loss 无法反向传播
优化器完全没有更新参数
```

### 得出的结论

整个数据流和梯度链路具备学习能力。问题集中在全量数据、小 batch 和训练稳定性，而不是基础代码完全失效。

## 6. 找到稳定性问题：BatchNorm 不适合当前小 batch

正式训练的 batch size 只有 2~4。Decoder 原先使用 BatchNorm，它依赖一个 batch 内的均值和方差；batch 太小时，训练统计量噪声很大，训练模式与验证模式还会使用不同统计方式。

实际表现是验证预测会在两种极端之间摆动：

```text
全背景
全道路或大量道路
```

因此把 Decoder 中的 BatchNorm 替换为 GroupNorm。GroupNorm 在通道组内归一化，不依赖 batch 内有多少样本，更适合当前显存和 batch size 条件。

### 得出的结论

问题不只是类别不平衡。小 batch 下的归一化方式会直接改变模型能否稳定地把训练能力迁移到验证模式。

## 7. 第二轮修正：Weighted BCE + Dice + GroupNorm

新的 segmentation loss：

```text
seg_loss = weighted_BCE + dice_weight * DiceLoss
total_loss = seg_loss + alpha * PCS_BCE
```

各部分分工：

```text
Weighted BCE：逐像素区分道路和背景，并提高道路错误的代价
Dice Loss：直接鼓励预测区域与真实道路区域重叠
PCS BCE：监督道路像素与八邻域的连接关系
GroupNorm：稳定小 batch 下的特征分布
```

同时加入以下诊断能力：

```text
Precision / Recall
predicted_positive_ratio / target_positive_ratio
batch 进度和每轮耗时
best.pt / last.pt
--resume 断点续训
```

## 8. 小规模控制变量实验：选择 pos_weight

先在固定的 200 train / 50 validation 上比较三个正样本权重，其他条件保持相同：

| pos_weight | 最佳 IoU | 主要现象 | 判断 |
| ---: | ---: | --- | --- |
| 22.5 | 0.1061 | `pred+` 约 0.30，大量误检 | 权重过强 |
| 10 | 0.1219 | `pred+` 约 0.15，Precision/Recall 相对平衡 | 当前最好 |
| 5 | 0.1042 | 容易重新偏向背景 | 道路梯度偏弱 |

因此确定全量候选配置：

```text
GroupNorm
seg_pos_weight = 10
pcs_pos_weight = 10
dice_weight = 1
learning_rate = 3e-4
batch_size = 4
num_workers = 2
```

这一步的意义不是证明 `10` 永远最优，而是在当前模型和训练预算下，为全量实验选择一个有证据支持的起点。

## 9. 全量训练：epoch 1~8 的完整趋势

| Epoch | Train loss | Val loss | IoU | F1 | Precision | Recall | pred+ |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.7606 | 1.5628 | 0.1181 | 0.2113 | 0.1248 | 0.6861 | 0.2211 |
| 2 | 1.3547 | 1.4092 | 0.2260 | 0.3687 | 0.3158 | 0.4428 | 0.0564 |
| 3 | 1.1381 | 1.0929 | 0.2546 | 0.4059 | 0.2697 | 0.8200 | 0.1224 |
| 4 | 1.0561 | 0.9779 | 0.3387 | 0.5060 | 0.3778 | 0.7658 | 0.0816 |
| 5 | 0.9911 | 1.0171 | 0.2662 | 0.4205 | 0.2781 | 0.8616 | 0.1247 |
| 6 | 0.9451 | 0.9179 | 0.3231 | 0.4884 | 0.3441 | 0.8408 | 0.0983 |
| 7 | 0.9209 | 0.8641 | 0.3738 | 0.5442 | 0.4084 | 0.8151 | 0.0803 |
| 8 | 0.9013 | 0.8673 | **0.3776** | **0.5482** | **0.4128** | 0.8155 | 0.0795 |

真实道路比例 `target+ = 0.0402`。

### 阶段 A：epoch 1，成功脱离全背景

IoU 首次达到 0.1181，证明新配置解决了“完全不预测道路”。但 `pred+=0.2211`，相当于把约 22% 像素判成道路，误检非常严重。

趋势含义：先让模型愿意预测道路，再逐步学习哪些线状区域才是真道路。

### 阶段 B：epoch 2，预测范围快速收缩

`pred+` 从 0.2211 降到 0.0564，Precision 从 0.1248 升到 0.3158，IoU 几乎翻倍。模型不再只是广泛猜测道路，开始学习位置和形状。

### 阶段 C：epoch 3~4，空间结构继续改善但仍有震荡

epoch 3 的 Recall 上升到 0.8200，同时 `pred+` 又升到 0.1224；epoch 4 则把 `pred+` 压回 0.0816，并将 IoU 提高到 0.3387。

趋势含义：加权 BCE 和 Dice 正在拉扯 Precision/Recall，模型尚未形成稳定决策边界。

### 阶段 D：epoch 5，出现明显回落

虽然 train loss 继续下降，但 val IoU 从 0.3387 回落到 0.2662，`pred+` 回升到 0.1247，Recall 很高但 Precision 下降。

这再次说明 train loss 下降不保证验证分割质量单调提高。固定学习率、较强正样本权重和数据增强会造成验证指标波动，因此必须保存 best checkpoint，而不能只保留最后一轮。

### 阶段 E：epoch 6~8，恢复并进入平台期

epoch 6 恢复到 0.3231；epoch 7 达到 0.3738；epoch 8 小幅刷新到 0.3776。epoch 7 和 8 已很接近，train loss 下降也变慢，说明当前 `3e-4` 固定学习率下开始进入平台。

最终最佳 checkpoint：

```text
runs/deepglobe/segroad-s-full-probe/best.pt
epoch = 8
val IoU = 0.3776
val F1 = 0.5482
```

## 10. 可视化验证：指标对应什么实际错误

使用 epoch 8 最佳 checkpoint，对 6 个固定 validation 样本生成四列对照：

```text
原始图像 | ground truth | prediction | 错误叠加图
绿色 TP | 红色 FP | 蓝色 FN
```

观察结果：

```text
已经能识别主干道路和部分交叉口
预测道路普遍比标注更粗
施工带、田埂、河岸等线状纹理容易成为 FP
狭窄支路、被建筑或树木遮挡的道路容易成为 FN
```

这与最终指标完全一致：Recall 0.8155 表示多数道路像素被找回，但 Precision 0.4128 表示误检仍多。`pred+=0.0795` 约为真实比例的两倍，也解释了为什么预测道路偏粗。

可视化文件：

```text
runs/deepglobe/segroad-s-full-probe/visualizations/comparison_sheet.png
```

## 11. 到当前训练结束，可以确认什么

### 已确认

```text
真实数据、mask、PCS、模型、loss 和梯度链路能够工作
普通 BCE 在当前类别比例下会发生全背景塌缩
单纯调 threshold 不能修复没有空间结构的概率图
GroupNorm 比当前小 batch 下的 BatchNorm 稳定
Weighted BCE + Dice 能使模型学习出道路空间结构
全量训练从 IoU 0 提升到了 0.3776
主要剩余错误已经从“不预测道路”变成“道路偏粗、误检过多”
```

### 还不能确认

```text
当前结果不是论文报告的完整复现结果
当前 4980/1246 split 与论文划分并不完全相同
公开 valid/test 缺少 mask，当前结果是本地 validation，不是官方 test 成绩
尚未完成 PCS 消融、学习率调度、阈值扫描和预训练 encoder 对照
尚未报告 APLS、TOPO-F1、参数量、FLOPs 和推理速度
```

Seg-Road 原论文报告的 DeepGlobe IoU 为 67.20%，当前 37.76% 仍有明显差距。除了训练策略，简化实现、随机初始化、数据增强、论文细节和数据划分都可能造成差异，不能把差距归因于单一因素。

## 12. 下一阶段的实验决策

不再直接用固定 `3e-4` 盲目追加很多 epoch。按成本和信息量排序：

```text
1. 对 epoch 8 checkpoint 做一次验证集 threshold sweep
2. 给 resume 训练加入明确的学习率覆盖或 scheduler
3. 降低学习率短跑 3~5 个 epoch，观察平台能否突破
4. 若 pred+ 仍明显偏高，再对照较低 pos_weight
5. 完成 PCS alpha=0 的消融实验
6. 评估预训练 encoder 或更贴近论文的架构实现
```

注意：resume 会恢复 optimizer state，其中包含旧学习率。仅在命令行重新填写 `--learning-rate` 并不能保证覆盖 checkpoint 内的学习率，因此训练脚本需要显式支持 resume 后覆盖学习率，或正式加入 scheduler。

### 已补充的收尾工具

当前已新增 `code/threshold_sweep.py`。它和反复运行 `evaluate.py --threshold ...` 的区别是：验证集只 forward 一遍，然后在同一批 sigmoid 概率图上同时统计多个 threshold 的 TP/FP/FN/TN。

这一步的实验意义：

```text
threshold 低：更容易判成道路，Recall 往往更高，但 FP 可能增加
threshold 高：更严格地判成道路，Precision 往往更高，但 FN 可能增加
最佳 threshold：通常选择 F1 或 IoU 最高的位置
```

阈值扫描只属于后处理，不会改变模型参数。如果低阈值和高阈值都无法得到合理道路结构，说明问题不在 decision boundary，而在模型输出概率图本身。

小样本 sanity check 使用 validation 前 10 张图，结果显示 threshold 从 0.25 提高到 0.80 时，Precision 持续上升、Recall 持续下降，IoU/F1 在 0.80 左右达到局部最好：

| Threshold | IoU | F1 | Precision | Recall | pred+ |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.25 | 0.2958 | 0.4566 | 0.3165 | 0.8193 | 0.0931 |
| 0.50 | 0.3602 | 0.5296 | 0.4187 | 0.7204 | 0.0619 |
| 0.70 | 0.3906 | 0.5618 | 0.5077 | 0.6289 | 0.0445 |
| 0.80 | 0.3953 | 0.5667 | 0.5669 | 0.5664 | 0.0359 |
| 0.95 | 0.3324 | 0.4990 | 0.7237 | 0.3807 | 0.0189 |

这组结果不能当作正式验证集成绩，因为样本数只有 10；它的作用是证明当前 checkpoint 的概率图具有可排序性，并提示默认 `threshold=0.5` 可能偏宽松。下一步应在完整 validation 上重点扫描 `0.65~0.85`。

在当前 Codex 执行环境中，PyTorch 能运行 CPU 评估，但看不到 Apple MPS；完整 validation 的 CPU 阈值扫描耗时过高。因此 `threshold_sweep.py` 已加入 `--log-interval`，正式运行时建议在本机 MPS 环境执行：

```bash
python3 code/threshold_sweep.py \
  --checkpoint runs/deepglobe/segroad-s-full-probe/best.pt \
  --image-dir data/deepglobe/images \
  --mask-dir data/deepglobe/masks \
  --split-list data/deepglobe_formal/splits/val.txt \
  --model-size s \
  --batch-size 4 \
  --thresholds 0.65,0.70,0.75,0.80,0.85 \
  --device mps \
  --log-interval 50 \
  --output-csv runs/deepglobe/segroad-s-full-probe/threshold_sweep_val.csv
```

如果 MPS 全量扫描结果仍显示 `0.75~0.80` 最好，最终报告中应同时说明两组成绩：默认 `threshold=0.5` 的模型输出表现，以及 validation-selected threshold 的后处理表现。

`threshold_sweep.py` 现在还会自动打印：

```text
best_iou_threshold=...
best_f1_threshold=...
```

并可用 `--output-csv` 保存完整表格。这样最终报告不需要从终端截图或聊天记录里抄数字，而是引用 CSV 作为可复查结果。

同时 `code/train.py` 已新增 `--resume-learning-rate`。这是因为 resume 时会恢复 optimizer state，旧学习率也在其中；如果要从 epoch 8 继续用更小学习率微调，需要显式覆盖：

```bash
python3 code/train.py \
  --resume runs/deepglobe/segroad-s-full-probe/best.pt \
  --resume-learning-rate 1e-4 \
  --reset-best-on-resume \
  --eval-threshold 0.5 \
  ...
```

这一步的实验意义：把“继续训练更多 epoch”和“用更小步长进入精修阶段”区分开。前者只是增加训练时间，后者是在改变优化策略。

如果续训写入新的 `--output-dir`，建议加 `--reset-best-on-resume`。否则训练脚本会沿用原 checkpoint 的 best IoU，可能导致新目录中只有 `last.pt`，没有这个续训实验自己的 `best.pt`。重置 best 追踪后，新实验目录会独立保存短程续训中的最佳 checkpoint，便于和 epoch 8 原始结果比较。

当前 `code/train.py` 还新增了 `--eval-threshold`，用于指定训练日志和 best checkpoint 选择时的二值化阈值。默认仍为 `0.5`，所以旧实验口径不变。等完整 threshold sweep 确认最佳阈值后，可以在续训或最终评估中使用同一阈值，例如：

```bash
--eval-threshold 0.8
```

注意：`--eval-threshold` 只影响指标计算和 best checkpoint 选择，不影响 loss，也不改变模型参数更新。训练时的 loss 仍然基于 logits 和真实 mask 计算；threshold 只是在统计 IoU/F1/Precision/Recall 时把概率图切成二值 mask。

### PCS 消融准备

当前 `code/train.py` 已新增 `--pcs-alpha`，默认仍是 `0.2`，与当前候选配置保持一致。若要验证 PCS 连通性监督的贡献，可以只改这一项：

```bash
python3 code/train.py \
  --image-dir data/deepglobe/images \
  --mask-dir data/deepglobe/masks \
  --train-list data/deepglobe_formal/splits/train.txt \
  --val-list data/deepglobe_formal/splits/val.txt \
  --output-dir runs/deepglobe/segroad-s-no-pcs \
  --model-size s \
  --epochs 8 \
  --batch-size 4 \
  --learning-rate 3e-4 \
  --seg-pos-weight 10 \
  --pcs-pos-weight 10 \
  --pcs-alpha 0 \
  --dice-weight 1 \
  --num-workers 2 \
  --device mps
```

这组实验要和 `segroad-s-full-probe` 对照。除了 IoU/F1，还要重点看可视化中的蓝色 FN 是否增多、道路是否更容易断裂。如果去掉 PCS 后像素指标接近但断裂明显增加，说明 PCS 的价值主要体现在拓扑连通，而不一定完全反映在 IoU 上。

### 参数量和推理速度

当前已新增 `code/model_stats.py`，用于报告模型参数量，并可选进行 forward latency 小基准：

```bash
python3 code/model_stats.py \
  --model-size s \
  --image-height 512 \
  --image-width 512 \
  --batch-size 1 \
  --device mps \
  --benchmark
```

参数量回答“模型有多大”，推理速度回答“实际使用有多快”。两者都不能替代 IoU/F1，但论文复现报告需要它们辅助判断：一个方法如果只提升一点指标却大幅增加参数或推理时间，实际价值就要重新评估。

### 正式结果表

当前已新增 `code/summarize_experiment.py`，用于从 checkpoint 自动生成 Markdown 结果表：

```bash
python3 code/summarize_experiment.py \
  --checkpoint runs/deepglobe/segroad-s-full-probe/best.pt \
  --model-size s \
  --output-md runs/deepglobe/segroad-s-full-probe/summary.md
```

如果已经完成全量 threshold sweep，还可以加入 validation-selected threshold 行：

```bash
python3 code/summarize_experiment.py \
  --checkpoint runs/deepglobe/segroad-s-full-probe/best.pt \
  --model-size s \
  --threshold-csv runs/deepglobe/segroad-s-full-probe/threshold_sweep_val.csv \
  --select-metric f1 \
  --output-md runs/deepglobe/segroad-s-full-probe/summary.md
```

这一步的意义是把 checkpoint、阈值扫描和参数量统一到一张表里。后面做 `segroad-s-no-pcs`、低学习率续训、SegRoadV2 对照时，都应该用同样格式记录，避免每次人工整理造成口径不一致。

### 实验配方记录

当前正式实验命令已整理到 `experiments/README.md`，包括：

```text
1. epoch 8 checkpoint 的完整 validation threshold sweep
2. 从 epoch 8 开始的低学习率短程续训
3. `--pcs-alpha 0` 的 PCS 消融
```

这一步不是新增算法，而是固定实验口径。后续每次跑实验，都应该优先从该文件复制命令，再只修改明确要测试的变量。

## 13. 过渡到 SegRoadV2 的判定标准

第一篇不需要做到“完全复现论文数值”才进入第二篇，但需要形成一个可靠 baseline。建议满足下面条件后开始 SegRoadV2：

```text
必须完成：
1. 完整 validation threshold sweep，并生成 threshold_sweep_val.csv
2. PCS 消融，至少跑出 `--pcs-alpha 0` 的对照结果
3. 用 summarize_experiment.py 生成统一 summary.md

建议完成：
4. 低学习率短程续训，判断 epoch 8 后是否还能提升
5. 至少一张 prediction/error overlay 对比 PCS 开关或续训前后
6. 参数量和推理速度记录
```

完成前 3 项后，即使当前 IoU 与论文报告仍有差距，也可以合理过渡。原因是我们已经知道当前实现的主要瓶颈：

```text
道路预测偏粗，FP 偏多
细小支路和遮挡道路仍容易断裂或漏检
固定 SRA 与普通局部卷积对弯曲、细长道路的适应性有限
PCS 能否稳定改善连通性仍需消融验证
```

这些问题正好引出 SegRoadV2：

```text
Deformable Self-Attention：让注意力采样位置适应弯曲道路
Groupable Deformable Convolution：让局部卷积不再只能看固定网格
条带卷积：更适合细长道路结构
继续保留 PCS：说明连通性监督仍是主线问题
```

因此，第一篇的收尾目标不是追到论文最高分，而是建立清楚的失败边界和对照基线。第二篇的学习就从“这些新模块分别想修第一篇哪个痛点”开始。

## 14. 这次训练最重要的方法论

```text
先建立 baseline，再修改；否则无法知道改动是否有效
同时看 loss、区域指标、类别比例和实际预测图
遇到失败先做单图过拟合，快速区分代码错误与优化问题
一次只改变一个主要变量，并使用固定 split
best checkpoint 比 last checkpoint 更可靠
阈值、loss 权重、归一化和学习率解决的是不同层面的问题
```

整个过程的核心不是“不断加参数直到指标上涨”，而是用可证伪的小实验逐层缩小问题范围：先证明失败现象，再证明链路能学，随后定位小 batch 稳定性，最后用控制变量选择可行配置并在全量数据上验证趋势。
