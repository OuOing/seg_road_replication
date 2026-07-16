# Seg-Road 学习笔记：Dataset、训练与评估

## 1. 为什么要把这部分单独写出来

前面的笔记已经解释了模型和损失函数，但还缺少把它们真正串起来的工程代码：

```text
磁盘图片和 mask
-> Dataset
-> DataLoader
-> model
-> SegRoadLoss
-> backward 和 optimizer.step
-> 验证集指标
```

本项目新增了四个基础脚本：

```text
code/dataset.py   读取 image、road_mask，并生成 pcs_target
code/metrics.py   计算二分类分割指标
code/train.py     训练和验证
code/evaluate.py  加载 checkpoint，独立评估
```

## 2. 参考了什么

作者公开的 SegRoadv2 仓库使用了 `Dataset`、训练循环、checkpoint 和 mIoU 评估脚本。本项目参考其工程组织方式，但没有直接复制 v2 的模型训练代码，因为 v2 的输出分支、标签格式和当前 v1 原型不同。

当前 v1 采用更简单的目录约定：

```text
data/images/road_001.jpg
data/masks/road_001.png
```

图片和 mask 只要文件名 stem 相同，就会被 `RoadDataset` 配成一对。

## 3. Dataset 做了什么

`RoadDataset.__getitem__` 每次返回一个样本：

```text
image:      (3, H, W)，浮点数，范围约为 0 到 1
road_mask:  (1, H, W)，0/1 浮点标签
pcs_target: (8, H, W)，由 road_mask 生成
```

处理步骤是：

```text
读取 RGB 图片和灰度 mask
-> resize 图片和 mask
-> 图片除以 255
-> mask 转成 mask > 0
-> 对训练样本随机水平或垂直翻转
-> 根据 mask 生成 PCS 标签
-> HWC 转成 CHW
```

图片 resize 使用双线性插值；mask 使用最近邻插值。原因是图片可以平滑变化，但 mask 只能保留离散的 0/1 类别，不能产生中间类别值。

## 4. 训练脚本的最小闭环

```text
optimizer.zero_grad()
seg_out, pcs_out = model(images)
total_loss = loss_fn(seg_out, masks, pcs_out, pcs_targets)
total_loss.backward()
optimizer.step()
```

验证阶段不更新参数：

```text
model.eval()
torch.no_grad()
模型预测
sigmoid
threshold
统计 TP、FP、FN、TN
```

训练脚本保存两个 checkpoint：

```text
runs/segroad/last.pt  最后一轮
runs/segroad/best.pt  验证集 IoU 最好的一轮
```

## 5. 当前实现的边界

这是一套用于学习和小规模复现的最小实现，暂时没有加入：

```text
随机裁剪和复杂颜色增强
多卡训练
学习率调度器
混合精度
PCS 预测的独立指标
完整论文级实验配置
```

这些功能可以在基础训练跑通后逐步加入。不要一开始把所有工程技巧混在一起，否则很难判断模型到底出了什么问题。

## 6. 运行方式

假设目录结构如下：

```text
data/
  images/
    road_001.jpg
  masks/
    road_001.png
```

训练：

```bash
python code/train.py \
  --image-dir data/images \
  --mask-dir data/masks \
  --epochs 20
```

评估：

```bash
python code/evaluate.py \
  --checkpoint runs/segroad/best.pt \
  --image-dir data/images \
  --mask-dir data/masks
```

## 7. 本节记住

```text
Dataset 负责把文件变成张量。
model 负责把输入张量变成预测。
loss 负责告诉模型预测错了多少。
metrics 负责告诉人模型效果如何。
```

---

## 8. 新术语普及

### 8.1 Dataset

`Dataset` 是“如何读取一个样本”的规则，不是整个数据集文件本身。

它主要回答两个问题：

```text
一共有多少个样本？       -> __len__
第 index 个样本是什么？  -> __getitem__
```

在本项目里，第 `index` 个样本由三部分组成：图片、道路 mask、PCS 标签。

### 8.2 DataLoader

`DataLoader` 是“批量取样器”。它从 `Dataset` 中取出多个样本，自动拼成一个 batch：

```text
Dataset:
  第 1 个样本 -> (3, H, W)
  第 2 个样本 -> (3, H, W)

DataLoader:
  两个样本 -> (B=2, 3, H, W)
```

这个自动拼接过程叫 `collate`，可以先理解为“把多个同形状样本堆叠起来”。

### 8.3 Batch、Iteration、Epoch

```text
batch:     一次送进模型的样本集合
iteration: 处理一个 batch 的一次循环
epoch:     完整遍历训练集一次
```

例如训练集有 100 张图，`batch_size=10`：

```text
1 个 epoch = 10 次 iteration
每次 iteration 处理 10 张图
```

### 8.4 Data Augmentation

`Data Augmentation` 是“数据增强”：对训练图片做保持语义不变的随机变化，例如水平翻转、垂直翻转、裁剪或颜色变化。

道路仍然是道路，但图片外观发生了变化，模型因此更不容易只记住训练图片的固定方向和颜色。

验证集通常不做随机增强，否则每次验证的输入都可能不同，指标不稳定。

### 8.5 Interpolation

`Interpolation` 是“插值”，用于 resize 图片。

```text
图片：使用双线性插值，让颜色变化平滑
mask：使用最近邻插值，避免生成 0 和 1 之间的伪标签
```

如果对二值 mask 使用双线性插值，边界可能出现 `0.3`、`0.7` 等值，标签含义就被污染了。

### 8.6 Checkpoint

`Checkpoint` 是训练过程中的保存文件，通常包含：

```text
模型参数
优化器参数
训练到第几轮
当时的验证指标
```

它的作用是：训练中断后继续、保存最佳模型、之后单独进行评估。

### 8.7 Smoke Test

`Smoke Test` 不是正式实验，而是“最小可运行检查”。在真正下载大数据集和训练几十个 epoch 前，先用几张图片确认：

```text
图片和 mask 能否正确配对
张量形状是否正确
PCS 标签是否能生成
模型能否前向传播
loss 能否 backward
checkpoint 能否保存和加载
```

它像先点火检查发动机，而不是马上开车跑长途。

## 9. Smoke Test 中的前向和反向传播

项目新增 `code/smoke_test.py`，它会临时生成两张简单的合成道路图片，不会修改真实数据集。

### 9.1 前向传播

```python
seg_out, pcs_out = model(images)
```

`forward pass`（前向传播）指数据从输入层一路经过网络，最后得到预测结果：

```text
images
-> Encoder
-> Decoder
-> seg_out、pcs_out
```

此时模型只是“做题”，还没有根据答案修改参数。

### 9.2 反向传播

```python
total_loss.backward()
```

`backward pass`（反向传播）指从损失开始，沿着刚才的计算过程反向计算每个参数的梯度。

梯度可以先理解成：

```text
这个参数应该增大还是减小？
改变多少比较合适？
```

### 9.3 参数更新

```python
optimizer.step()
```

`optimizer`（优化器）根据梯度修改模型参数。常见优化器有 SGD 和 AdamW，本项目当前使用 AdamW。

完整顺序必须是：

```text
optimizer.zero_grad()  清除上一次梯度
model(images)          前向传播
loss.backward()        反向传播，计算梯度
optimizer.step()       使用梯度更新参数
```

### 9.4 Smoke Test 预期形状

```text
images:      (2, 3, 64, 64)
masks:       (2, 1, 64, 64)
pcs_targets: (2, 8, 64, 64)
seg_out:     (2, 1, 64, 64)
pcs_out:     (2, 8, 64, 64)
```

如果这些形状和 loss 都正常，说明最基本的训练管道已经接通，但这不代表模型已经学会道路提取。

## 10. 什么是清理旧梯度

### 10.1 梯度默认会累加

在 PyTorch 中，参数的 `.grad` 默认不是每次自动覆盖，而是会累加新梯度。

假设第一次 iteration 算出的梯度是：

```text
旧梯度 = 2
```

第二次 iteration 算出的新梯度是：

```text
新梯度 = 3
```

如果不清理，参数中可能留下：

```text
累计梯度 = 2 + 3 = 5
```

但我们通常希望第二次更新只依据第二个 batch 的梯度 `3`，而不是把上一个 batch 的 `2` 也混进来。

### 10.2 `zero_grad()` 做什么

```python
optimizer.zero_grad()
```

它会把模型参数上一次保存的梯度清除或置零，使下一次反向传播从干净状态开始。

因此训练循环通常写成：

```python
for images, masks, pcs_targets in loader:
    optimizer.zero_grad()
    seg_out, pcs_out = model(images)
    total_loss, _, _ = loss_fn(
        seg_out, masks, pcs_out, pcs_targets
    )
    total_loss.backward()
    optimizer.step()
```

### 10.3 为什么反向传播不会自动清理

梯度累加是 PyTorch 的设计选择，因为有时我们确实希望使用多个小 batch 的梯度来模拟一个大 batch，这叫 `gradient accumulation`（梯度累积）。

但当前项目使用普通训练方式，每个 batch 都应该独立更新一次，因此需要在每次循环开始时调用 `zero_grad()`。

本节记住：

```text
zero_grad：清除上一批数据留下的梯度
backward：根据当前损失计算新梯度
step：用当前梯度更新参数
```

---

## 11. 训练集、验证集与测试集

### 11.1 三种数据集的职责

一个完整的机器学习项目通常把数据分成三部分：

```text
训练集 train：用于更新模型参数
验证集 val：用于选择模型和调整超参数
测试集 test：最后一次客观评价模型
```

当前 `train.py` 只自动划分训练集和验证集。测试集需要在正式实验时另外准备，不能反复拿测试集调参数，否则测试集就不再客观。

### 11.2 什么是泛化

`Generalization`（泛化）指模型不仅能处理训练时见过的图片，还能处理没有见过的新图片。

道路提取真正想要的是：

```text
训练图片 -> 学习道路规律
新图片   -> 也能正确提取道路
```

而不是：

```text
训练图片 -> 记住这几张图片的像素和背景
新图片   -> 表现很差
```

### 11.3 什么是过拟合

`Overfitting`（过拟合）指模型在训练集上越来越好，但在验证集或新数据上越来越差。

常见表现：

```text
训练 loss 持续下降
训练 IoU 持续上升
验证 loss 先下降后上升
验证 IoU 先上升后下降
```

可以把它看成模型“背答案”，而不是理解道路的共同特征。

### 11.4 什么是欠拟合

`Underfitting`（欠拟合）是另一种情况：模型连训练集都没有学好。

```text
训练 loss 很高
训练 IoU 很低
验证 IoU 也很低
```

可能原因包括模型太小、训练轮数太少、学习率不合适或数据预处理有问题。

### 11.5 验证集为什么不能反向传播

验证阶段只回答“当前模型效果如何”，不应该修改模型参数：

```python
model.eval()
with torch.no_grad():
    seg_out, pcs_out = model(images)
```

`eval()` 会切换 Dropout、BatchNorm 等层的行为；`no_grad()` 会关闭梯度记录，节省显存和计算。

验证集的结果可以用于选择：

```text
best.pt：验证 IoU 最好的模型
学习率：参数更新步长
batch_size：每批样本数
threshold：概率转 0/1 时的阈值
```

这些可调设置叫 `hyperparameters`（超参数），它们不是模型通过训练自动学出来的参数。

### 11.6 数据泄漏

`Data Leakage`（数据泄漏）指验证集或测试集的信息意外参与了训练。

道路数据尤其要注意相邻裁剪块：如果同一张大图裁出的相邻区域被分别放进训练集和验证集，验证分数可能虚高，因为两部分画面非常相似。

本节记住：

```text
训练集用来学习，验证集用来选择，测试集用来最后评价。
训练好不等于泛化好。
```

---

## 12. 学习率与 AdamW

### 12.1 参数为什么会改变

反向传播得到梯度后，优化器根据梯度调整参数。最简单的更新形式可以写成：

```text
new_parameter = old_parameter - learning_rate * gradient
```

`learning_rate`（学习率）就是每次更新的步长。

```text
gradient：告诉模型往哪个方向改
learning_rate：告诉模型每次改多大
```

### 12.2 学习率过大或过小

学习率过大：

```text
参数一步跨得太远
可能在最优点附近来回跳
loss 震荡，甚至变成 NaN
```

学习率过小：

```text
每次只移动一点点
loss 下降很慢
训练需要很长时间
```

因此学习率不是越大越好，也不是越小越稳定，而要和模型、数据、batch size 一起调整。

当前代码中的默认值是：

```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4,
)
```

`1e-4` 就是 `0.0001`。

### 12.3 AdamW 是什么

`AdamW` 是一种优化器名称，通常比最基础的 SGD 更容易作为深度网络的起点。

它会根据历史梯度维护两类信息：

```text
梯度的平均方向
梯度变化的大小
```

因此不同参数可以使用不同的有效步长，而不是所有参数都使用完全相同的更新方式。

可以把它理解成：

```text
SGD：所有参数拿着相同大小的步长前进
AdamW：根据每个参数过去的梯度情况，自适应调整步长
```

### 12.4 Weight Decay

`Weight Decay`（权重衰减）是一种正则化方法，会轻微限制参数变得过大，帮助降低过拟合风险。

`AdamW` 末尾的 `W` 表示它对权重衰减进行了更清晰的处理。

它不是让模型完全不学习，而是加入一种偏好：

```text
在能达到相近效果时，倾向于使用不要过度膨胀的参数
```

### 12.5 超参数

`Hyperparameter`（超参数）是训练前由人设置的值，不是模型通过数据自动学出来的参数。

当前项目中的超参数包括：

```text
learning_rate：学习率
batch_size：批大小
epochs：训练轮数
alpha：PCS 损失权重
threshold：二值化阈值
```

模型内部的卷积核权重则是 `parameter`（模型参数），会通过反向传播和优化器自动更新。

本节记住：

```text
梯度决定方向，学习率决定步长，AdamW 根据历史梯度帮助更新参数。
```

---

## 13. 学习率调度器

### 13.1 为什么不能一直使用同一个学习率

训练早期，模型参数通常离较好的位置还很远，需要较大的步子快速学习；训练后期，模型已经接近较好的位置，需要更小的步子细致调整。

```text
训练早期：快速靠近较好区域
训练后期：小步调整，减少震荡
```

如果整个训练过程都使用同一个学习率，可能出现：

```text
学习率一直很大：后期在最优点附近震荡
学习率一开始就很小：前期学习过慢
```

### 13.2 Scheduler

`Scheduler`（学习率调度器）是一个根据训练进度自动改变学习率的工具。

训练代码可能变成：

```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4,
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=epochs,
)
```

每个 epoch 结束后：

```python
scheduler.step()
```

### 13.3 常见调度策略

#### Step Decay

`Step Decay`（阶梯衰减）每隔固定轮数把学习率乘一个系数：

```text
epoch 1～10：0.001
epoch 11～20：0.0001
epoch 21～30：0.00001
```

优点是简单，缺点是变化突然。

#### Cosine Annealing

`Cosine Annealing`（余弦退火）让学习率按照余弦曲线平滑下降：

```text
开始较大
-> 平滑下降
-> 训练后期接近较小值
```

它不会像 Step Decay 那样突然跳变。

#### Warmup

`Warmup`（预热）指训练最开始先使用较小学习率，再逐渐升到目标学习率。

```text
初始几轮：小学习率
-> 慢慢增大
-> 进入正常训练阶段
```

当模型很大、batch 很大或训练刚开始不稳定时，Warmup 可能有帮助。

### 13.4 `scheduler.step()` 放在哪里

如果使用按 epoch 调度的 scheduler，通常写成：

```python
for epoch in range(epochs):
    train_one_epoch(...)
    validate(...)
    scheduler.step()
```

这样表示：完成一轮训练和验证后，再为下一轮调整学习率。

不同 scheduler 的调用频率可能不同，有些按 batch 调整，有些按 epoch 调整，必须看具体配置。

### 13.5 如何观察学习率

可以在训练日志中打印：

```python
current_lr = optimizer.param_groups[0]["lr"]
print(current_lr)
```

如果发现 loss 不下降，不能只看模型结构，还要检查：

```text
当前学习率是多少
学习率是否突然变成 0
学习率是否大到导致 loss 震荡
```

本节记住：

```text
学习率调度器让模型前期敢于快走，后期能够小步精调。
```

---

## 14. Checkpoint 与恢复训练

### 14.1 为什么要保存模型

训练深度模型可能需要很长时间。如果程序中断、电脑重启或想比较不同 epoch 的效果，就需要把训练状态保存下来。

```python
torch.save(checkpoint, "runs/segroad/best.pt")
```

这里的 `checkpoint` 就是一个保存文件，通常使用 `.pt` 或 `.pth` 后缀。

### 14.2 `state_dict`

`state_dict` 可以理解为“模型参数字典”：

```python
state_dict = model.state_dict()
```

它保存模型每一层的参数，例如：

```text
卷积层权重
卷积层 bias
Linear 层权重
LayerNorm 参数
```

加载模型参数：

```python
model.load_state_dict(checkpoint["model"])
```

注意：加载参数前，模型结构必须和保存参数时基本一致。

### 14.3 `best.pt` 和 `last.pt`

本项目保存两个文件：

```text
last.pt：最后一个 epoch 保存的状态
best.pt：验证集 IoU 最好的状态
```

它们的用途不同：

```text
继续训练：通常使用 last.pt
正式推理：通常使用 best.pt
```

最后一轮不一定是效果最好的一轮，因为训练后期可能发生过拟合。

### 14.4 为什么恢复训练还要保存优化器

如果只保存：

```python
model.state_dict()
```

可以用于推理，但不一定适合无缝恢复训练。

恢复训练通常还需要：

```python
optimizer.state_dict()
```

因为 AdamW 保存了历史梯度相关的状态。如果不恢复这些状态，优化器相当于忘记了之前的训练历史，继续训练时更新行为可能发生变化。

此外还可以保存：

```text
epoch：已经训练到第几轮
args：训练配置
metrics：当时的验证指标
```

### 14.5 加载 checkpoint 的两种目的

#### 只做推理

```python
checkpoint = torch.load("runs/segroad/best.pt", map_location=device)
model.load_state_dict(checkpoint["model"])
model.eval()
```

此时不需要创建优化器，也不需要反向传播。

#### 继续训练

```python
checkpoint = torch.load("runs/segroad/last.pt", map_location=device)
model.load_state_dict(checkpoint["model"])
optimizer.load_state_dict(checkpoint["optimizer"])
start_epoch = checkpoint["epoch"] + 1
```

之后从 `start_epoch` 继续训练。

### 14.6 `map_location`

```python
torch.load(path, map_location=device)
```

`map_location` 指定把保存的张量加载到哪里：

```text
GPU 保存的模型 -> 当前 CPU
CPU 保存的模型 -> 当前 GPU
```

这样可以减少模型保存设备和当前设备不同导致的加载错误。

本节记住：

```text
state_dict 保存模型参数。
best.pt 适合选择最佳模型做推理。
last.pt 适合从最后状态继续训练。
恢复训练时，最好同时恢复模型和优化器状态。
```

## 15. 恢复训练和推理不是一回事

### 15.1 恢复训练

`Resume Training`（恢复训练）指训练过程因为中断而停止后，从保存的 checkpoint 接着训练，而不是从随机参数重新开始。

```text
已经训练 20 个 epoch
-> 程序中断
-> 从 checkpoint 加载
-> 从第 21 个 epoch 继续
```

它通常需要恢复：

```text
模型参数
优化器状态
当前 epoch
学习率调度器状态（如果使用）
```

### 15.2 加载模型做推理

`Inference`（推理）指模型已经训练好后，只输入新图片并得到预测，不再更新参数。

```text
加载 best.pt
-> model.eval()
-> 输入新图片
-> 得到 seg_out、pcs_out
-> sigmoid 和 threshold
-> 输出道路掩膜
```

推理不需要：

```text
loss.backward()
optimizer.step()
```

### 15.3 为什么需要恢复训练

常见原因包括：

```text
训练时间太长，今天先暂停
电脑或程序意外中断
显存不足，需要修改 batch size
想从已有模型继续微调
```

恢复训练可以保留之前已经学到的内容，避免从随机参数重新开始。

### 15.4 当前项目的实际状态

当前 `code/train.py` 已经保存了模型、优化器、epoch 和指标，但还没有加入 `--resume` 命令行参数。因此它目前可以保存 checkpoint，也可以由其他代码手动加载，但默认重新运行训练脚本时仍会从头开始。

## 16. 为什么同一训练集要训练多个 epoch

### 16.1 参数不是每个 epoch 固定不变

训练过程中变化的是模型参数：

```text
第 1 次看到图片 -> 使用参数 W1 -> 得到预测 P1 -> 计算梯度 G1
第 2 次看到图片 -> 使用参数 W2 -> 得到预测 P2 -> 计算梯度 G2
```

即使图片和标签完全相同，`P1` 和 `P2` 通常也不同，因此梯度 `G1` 和 `G2` 也不同。

### 16.2 多个 epoch 是反复修正

```text
第 1 个 epoch：先学会比较明显的道路特征
第 2 个 epoch：修正第一轮剩下的错误
第 3 个 epoch：继续减少漏检和误检
...
```

每轮不是简单重复同一个计算，而是使用上一次更新后的参数重新预测和纠错。

### 16.3 即使数据不变，模型也在变

模型训练可以理解为不断重复：

```text
当前参数
-> 预测
-> 与标签比较
-> 计算当前错误
-> 根据当前错误修改参数
-> 新参数
```

因此：

```text
数据相同
模型状态不同
预测不同
梯度不同
更新方向也可能不同
```

### 16.4 batch 让每轮更新带有随机性

训练集通常会在每个 epoch 重新打乱：

```python
DataLoader(dataset, shuffle=True)
```

不同 epoch 中，batch 的组合和顺序可能不同。每个 batch 使用当前参数产生自己的梯度，优化器再逐步更新模型。

### 16.5 为什么不能只训练一个 epoch

一个 epoch 只表示每张训练图片被模型看过一次。第一次看时模型通常还没有学好，因此只更新一次远远不够。

如果过早停止：

```text
模型参数还不成熟
训练 loss 较高
训练 IoU 较低
```

但 epoch 也不是越多越好。训练太久可能导致过拟合，所以需要观察验证集并保存 best checkpoint。

### 16.6 参数更新的具体位置

当前项目中，参数不是在 epoch 结束时才统一变化，而是在每个 batch 后变化：

```text
一个 epoch
= 多个 batch
= 多次参数更新
```

例如 100 张图、batch_size=10：

```text
1 个 epoch = 10 次 iteration = 10 次 optimizer.step()
```

本节记住：

```text
epoch 重复的是数据遍历，不是重复使用同一组固定权重。
每个 batch 都会更新权重，所以后面的预测和梯度已经不同。
```

## 17. 什么时候停止训练

### 17.1 不能只看训练 loss

训练 loss 只反映模型在训练集上的表现。模型可能继续记忆训练图片，使训练 loss 继续下降，但对新图片的效果已经变差。

因此至少要同时观察：

```text
train_loss
val_loss
train_IoU
val_IoU
```

### 17.2 理想的训练曲线

比较理想的早期阶段是：

```text
train_loss 下降
val_loss   下降
train_IoU   上升
val_IoU     上升
```

这说明模型既在学习训练集，也在提升对未见图片的泛化能力。

### 17.3 过拟合曲线

过拟合通常表现为：

```text
train_loss 继续下降
val_loss   开始上升
train_IoU   继续上升
val_IoU     开始下降
```

这时继续训练通常没有好处，应优先保留验证集效果最好的 checkpoint。

### 17.4 Early Stopping

`Early Stopping`（早停）指验证集指标长时间没有改善时，提前结束训练。

例如设置：

```text
patience = 5
```

含义是：如果连续 5 个 epoch 的验证 IoU 都没有超过历史最好值，就停止训练。

```text
第 20 轮：val IoU 最好
第 21 轮：没有提升，等待 1
第 22 轮：没有提升，等待 2
...
第 25 轮：仍无提升，停止
```

`patience`（耐心轮数）不是模型参数，而是训练控制策略。

### 17.5 Best Checkpoint

即使不实现 Early Stopping，也可以保存验证集指标最好的模型：

```python
if val_metrics["iou"] > best_iou:
    best_iou = val_metrics["iou"]
    torch.save(checkpoint, output_dir / "best.pt")
```

这样训练结束后使用 `best.pt`，而不是盲目使用最后一轮。

### 17.6 当前项目如何判断

当前 `code/train.py` 已经实现了：

```text
每个 epoch 计算验证集 IoU
验证 IoU 变好时保存 best.pt
每轮都保存 last.pt
```

暂时还没有自动 Early Stopping。基础训练跑通后，再加入它更容易理解。

本节记住：

```text
训练 loss 告诉我们模型是否在记住训练集。
验证指标帮助我们判断模型是否真的变好。
best checkpoint 比最后 checkpoint 更值得信任。
```

## 18. 推理流程

### 18.1 什么是推理

`Inference`（推理）指训练完成后，使用模型处理没有参与参数更新的新图片。

```text
训练阶段：图片 + 标签 -> loss -> 更新参数
推理阶段：只有图片 -> 模型 -> 预测结果
```

推理阶段没有 `backward()`，也没有 `optimizer.step()`。

### 18.2 推理前的预处理

新图片必须尽量使用和训练时相同的预处理：

```text
读取 RGB 图片
-> resize 到模型要求的尺寸
-> 转成浮点数
-> 除以 255
-> HWC 转 CHW
-> 增加 batch 维度
```

一张图片经过处理后：

```text
原始图片：H x W x 3
模型输入：1 x 3 x H x W
```

如果训练时图片除以了 255，而推理时没有除，模型看到的数值范围就不同，预测可能明显变差。

### 18.3 `eval()` 和 `no_grad()`

```python
model.eval()
with torch.no_grad():
    seg_out, pcs_out = model(image)
```

`eval()` 让模型进入评估模式；`no_grad()` 关闭梯度记录。推理不需要保存反向传播所需的中间结果，因此更省内存。

### 18.4 从 logits 得到道路掩膜

模型输出的是 logits，不是 0/1：

```text
seg_out -> sigmoid -> road_probability -> threshold -> road_mask
```

代码逻辑：

```python
probability = torch.sigmoid(seg_out)
road_mask = probability >= 0.5
```

输出结果是：

```text
1：预测为道路
0：预测为背景
```

### 18.5 PCS 分支如何使用

PCS 输出包含 8 个方向的连通性预测：

```text
pcs_out: (1, 8, H, W)
```

它可以用于：

```text
观察道路连接关系
辅助分析道路断裂
通过 reverse mapping 得到连通区域提示
```

当前项目的主道路掩膜来自 `seg_out`；`pcs_out` 主要作为训练时的辅助任务和推理分析信号。不要把 PCS 预测直接当成普通的单通道道路 mask。

### 18.6 后处理

`Post-processing`（后处理）指模型输出之后的整理步骤，例如：

```text
阈值化
去除很小的孤立区域
填补局部空洞
恢复到原始图片尺寸
保存为 PNG
```

后处理不能替代模型训练，只能整理模型已经产生的预测。

### 18.7 训练尺寸和原图尺寸

如果训练时把图片 resize 到 `512 x 512`，但原始图片尺寸不是这个大小，推理结束后需要把预测 mask resize 回原始尺寸：

```text
原图：1500 x 1500
-> 模型输入：512 x 512
-> 模型输出：512 x 512
-> 最终结果：1500 x 1500
```

恢复尺寸时，二值 mask 应使用最近邻插值，避免边界产生新的灰度值。

本节记住：

```text
训练是学习参数，推理是使用参数。
seg_out 产生主要道路预测，pcs_out 描述八方向连通性。
推理时必须保持和训练一致的预处理。
```

---

## 19. 把推理流程写成脚本

项目新增 `code/predict.py`，用于对一张图片生成道路 mask：

```bash
python3 code/predict.py \
  --checkpoint runs/segroad/best.pt \
  --image data/demo.jpg \
  --output runs/segroad/demo_mask.png
```

脚本步骤是：

```text
加载 checkpoint
-> 创建同样的 SegRoad 模型
-> 加载模型参数
-> 读取并预处理图片
-> model.eval() 和 no_grad()
-> 得到 seg_out
-> sigmoid 和 threshold
-> 恢复原图尺寸
-> 保存 PNG mask
```

这里没有把 `pcs_out` 直接合并进最终 mask，因为当前 v1 的主要预测分支是 `seg_out`，PCS 主要作为训练中的结构监督。后续可以专门比较三种结果：

```text
只使用 seg_out
只使用 reverse_mapping(pcs_out)
融合 seg_out 和 PCS 结果
```

这类比较属于 `ablation study`（消融实验）：保持其他条件不变，只移除或替换一个模块，观察它对结果的影响。

本节记住：

```text
predict.py 是使用训练好模型的入口。
输出 mask 保存为 0 和 255，而不是直接保存概率。
恢复尺寸时，二值 mask 使用最近邻插值。
```

## 20. 概率阈值与二值化

### 20.1 为什么需要阈值

模型输出经过 sigmoid 后是概率：

```text
0.02、0.31、0.56、0.91
```

但道路 mask 通常需要离散标签：

```text
0：背景
1：道路
```

因此需要一个 `threshold`（阈值）把连续概率转换为 0/1：

```text
probability >= threshold -> 道路
probability <  threshold -> 背景
```

常用默认值是：

```text
threshold = 0.5
```

### 20.2 阈值降低会怎样

例如把阈值从 `0.5` 降到 `0.3`：

```text
更多像素会被判为道路
漏检可能减少
误检可能增加
```

这通常有利于提高 Recall，但可能降低 Precision。

### 20.3 阈值提高会怎样

例如把阈值从 `0.5` 提高到 `0.7`：

```text
只有模型非常确定的像素才被判为道路
误检可能减少
漏检可能增加
```

这通常有利于提高 Precision，但可能降低 Recall。

### 20.4 阈值和道路提取的关系

道路通常比较细，且容易被树木、阴影遮挡。如果阈值过高，模型给道路边缘的概率可能达不到阈值，结果出现断裂。

```text
阈值过高 -> 道路变细、断裂、漏检增加
阈值过低 -> 道路变粗、噪声和误检增加
```

### 20.5 如何选择更合适的阈值

不要只凭经验固定 `0.5`，可以在验证集上测试多个阈值：

```text
0.1、0.2、0.3、0.4、0.5、0.6、0.7、0.8、0.9
```

分别计算：

```text
IoU
F1
Precision
Recall
```

然后根据任务目标选择：

```text
希望道路尽量完整 -> 可以偏向较低阈值
希望误检尽量少   -> 可以偏向较高阈值
综合效果         -> 选择验证集 F1 或 IoU 较好的阈值
```

阈值只能在验证集上选择，不能反复根据测试集结果调阈值，否则会造成测试集信息泄漏。

### 20.6 概率不一定是真实概率

sigmoid 输出在 0 到 1 之间，但 `0.8` 不一定意味着“有 80% 的真实概率”。它首先是模型用于分类的置信分数。

`Calibration`（概率校准）研究的是：模型输出的分数是否和真实发生概率一致。当前项目先把 sigmoid 输出理解为道路置信度即可。

本节记住：

```text
阈值低：Recall 通常更高，但误检可能增加。
阈值高：Precision 通常更高，但漏检可能增加。
0.5 是默认起点，不一定是最终最优值。
```

## 21. 混淆矩阵

### 21.1 什么是混淆矩阵

`Confusion Matrix`（混淆矩阵）不是模型中的一层，而是统计预测结果的表格。

二分类道路分割中，它记录预测和真实标签的组合：

```text
                 真实道路       真实背景
预测道路           TP             FP
预测背景           FN             TN
```

### 21.2 用道路像素理解四种结果

```text
TP：模型说是道路，真实确实是道路
FP：模型说是道路，真实其实是背景
FN：模型说是背景，真实其实是道路
TN：模型说是背景，真实确实是背景
```

例子：

```text
真实 mask：    [道路, 道路, 背景, 背景]
预测 mask：    [道路, 背景, 道路, 背景]
```

逐个位置比较：

```text
第 1 个：TP
第 2 个：FN
第 3 个：FP
第 4 个：TN
```

所以：

```text
TP = 1
FN = 1
FP = 1
TN = 1
```

### 21.3 指标如何从矩阵得到

```text
Precision = TP / (TP + FP)
Recall    = TP / (TP + FN)
IoU       = TP / (TP + FP + FN)
```

每个指标忽略的部分不同：

```text
Precision 重点关注 FP
Recall    重点关注 FN
IoU       同时考虑 FP 和 FN
```

### 21.4 道路断裂属于哪种错误

假设真实道路是一条连续道路，但模型在中间漏掉了一段：

```text
真实：道路 道路 道路 道路 道路
预测：道路 道路 背景 背景 道路
```

中间漏掉的道路像素属于 `FN`：

```text
模型预测背景
真实却是道路
```

因此道路断裂通常会降低：

```text
Recall
IoU
F1
```

PCS 训练的目标之一，就是帮助模型减少这种道路连接位置的 FN。

### 21.5 道路周围误检属于哪种错误

如果模型把道路旁边的屋顶或阴影判成道路：

```text
真实：背景 背景 道路
预测：道路 背景 道路
```

第一个位置属于 `FP`：

```text
模型预测道路
真实却是背景
```

FP 增多通常会降低：

```text
Precision
IoU
F1
```

### 21.6 为什么 Accuracy 可能具有误导性

假设一张图有 10000 个像素，但道路只有 500 个：

```text
背景：9500 个
道路：500 个
```

如果模型把所有像素都预测为背景：

```text
TN = 9500
FN = 500
```

Accuracy 是：

```text
9500 / 10000 = 95%
```

看起来很高，但道路一个像素都没找出来：

```text
Recall = 0
IoU = 0
```

所以道路提取不能只看 Accuracy，要重点看道路 IoU、Recall、Precision 和 F1。

本节记住：

```text
FN 是漏掉真实道路，FP 是误报背景为道路。
道路断裂主要增加 FN，道路噪声主要增加 FP。
```

## 22. Precision、Recall 与 F1

### 22.1 Precision 和 Recall 关注不同问题

```text
Precision：我预测出来的道路，有多少是真的？
Recall：真实存在的道路，有多少被我找到了？
```

对应公式：

```text
Precision = TP / (TP + FP)
Recall    = TP / (TP + FN)
```

`Precision` 主要惩罚误检 `FP`；`Recall` 主要惩罚漏检 `FN`。

### 22.2 为什么两者会互相牵制

降低阈值时，模型更容易把像素判成道路：

```text
Recall 通常上升
FP 可能增加
Precision 可能下降
```

提高阈值时，模型更谨慎：

```text
Precision 通常上升
FN 可能增加
Recall 可能下降
```

因此不能只追求一个指标。

### 22.3 F1 是什么

`F1 score` 是 Precision 和 Recall 的综合指标：

```text
F1 = 2 * Precision * Recall / (Precision + Recall)
```

它使用的是 `Harmonic Mean`（调和平均），而不是普通的算术平均。

### 22.4 为什么不用算术平均

假设：

```text
Precision = 1.0
Recall = 0.0
```

算术平均是：

```text
(1.0 + 0.0) / 2 = 0.5
```

但这会掩盖 Recall 为 0 的严重问题。

F1 是：

```text
F1 = 2 * 1.0 * 0.0 / (1.0 + 0.0) = 0
```

只要 Precision 或 Recall 中有一个很低，F1 就会明显降低。

因此 F1 更适合表达：

```text
两个指标必须同时不错
```

### 22.5 一个具体例子

假设：

```text
TP = 80
FP = 20
FN = 40
```

那么：

```text
Precision = 80 / (80 + 20) = 0.80
Recall    = 80 / (80 + 40) = 0.67
F1        = 2 * 0.80 * 0.67 / (0.80 + 0.67) ≈ 0.73
IoU       = 80 / (80 + 20 + 40) ≈ 0.57
```

这里模型的误检和漏检都存在，因此 F1 和 IoU 都不会特别高。

### 22.6 F1 和 IoU 的关系

在同一个二分类任务、同一组 TP、FP、FN 下，F1 和 IoU 可以互相换算：

```text
F1 = 2 * IoU / (1 + IoU)
IoU = F1 / (2 - F1)
```

它们关注的核心都是道路预测与真实道路的重叠，但表达形式不同：

```text
F1：更强调 Precision 和 Recall 的平衡
IoU：更强调预测区域和真实区域的交集占并集比例
```

本节记住：

```text
Precision 防止乱报道路。
Recall 防止漏掉道路。
F1 要求两者同时保持较好。
```

## 23. IoU、MIoU 与 F1 的区别

### 23.1 IoU 关注一个类别的区域重叠

`IoU`（Intersection over Union，交并比）计算预测区域和真实区域的重叠程度：

```text
IoU = 交集面积 / 并集面积
    = TP / (TP + FP + FN)
```

在道路提取中，通常说的 `IoU_road` 是只针对道路类别计算的 IoU。

### 23.2 MIoU 是多个类别 IoU 的平均

`MIoU`（Mean Intersection over Union，平均交并比）通常先计算每个类别的 IoU，再取平均：

```text
MIoU = (IoU_background + IoU_road) / 2
```

如果有更多类别，就对所有类别的 IoU 求平均。

因此：

```text
IoU_road：只看道路
MIoU：同时看道路和背景的平均表现
```

### 23.3 为什么道路 IoU 和 MIoU 不一样

假设：

```text
IoU_background = 0.95
IoU_road = 0.60
```

那么：

```text
MIoU = (0.95 + 0.60) / 2 = 0.775
```

这时道路本身的 IoU 只有 `0.60`，但由于背景分割很好，MIoU 会更高。

所以看道路提取效果时，不能只看 MIoU，还要单独看 `IoU_road`。

### 23.4 F1 和 IoU 的共同点

F1 和 IoU 都主要使用：

```text
TP、FP、FN
```

它们都不直接依赖大量的 TN，因此比 Accuracy 更适合道路这种前景较少的任务。

对于同一组二分类预测：

```text
F1 = 2 * IoU / (1 + IoU)
```

两者数值不同，但通常变化方向相近：道路区域重叠变好时，IoU 和 F1 往往一起提高。

### 23.5 指标报告应该怎么看

论文表格中看到多个指标时，可以按这个顺序理解：

```text
IoU_road：道路区域本身分割得好不好
MIoU：道路和背景整体平均好不好
F1：道路 Precision 和 Recall 平衡得好不好
Precision：误检多不多
Recall：漏检多不多
```

例如：

```text
Precision 高、Recall 低：模型很谨慎，但漏掉很多道路
Recall 高、Precision 低：模型找得很全，但误检很多
IoU 和 F1 都高：道路区域整体重叠和查准查全都较好
```

### 23.6 指标不能脱离数据集比较

不同论文可能使用不同的：

```text
数据集划分
道路像素定义
阈值
预处理方式
评价代码
```

因此不能只看到某个数字更大，就直接断言模型一定更好。应先确认评价条件一致。

本节记住：

```text
IoU_road 看道路，MIoU 看类别平均，F1 看查准率和查全率平衡。
道路提取最重要的是结合多个指标，而不是只看一个最高数字。
```

## 24. 消融实验与实验表格

### 24.1 什么是消融实验

`Ablation Study`（消融实验）指保持数据集、训练配置和评价方式尽量一致，只移除或替换一个模块，观察性能变化。

它回答的不是“哪个模型分数最高”，而是：

```text
这个模块到底有没有贡献？
```

### 24.2 Seg-Road 中可以怎样设计消融

围绕当前论文，可以构造这样的实验组：

```text
A：CNN 基线，不使用 Transformer，不使用 PCS
B：CNN + Transformer，不使用 PCS
C：CNN + PCS，不使用 Transformer
D：CNN + Transformer + PCS，完整模型
```

如果 D 比 B 更好，可以支持：

```text
PCS 对道路连通结构有帮助
```

如果 B 比 A 更好，可以支持：

```text
Transformer 对全局道路依赖有帮助
```

如果 C 比 A 更好，可以支持：

```text
PCS 作为结构监督可以改善分割
```

### 24.3 消融实验必须控制变量

公平的消融实验应该尽量保持：

```text
训练集和验证集相同
训练轮数相同
随机种子相同或记录清楚
输入尺寸相同
优化器和学习率相同
评价代码相同
```

否则最后的分数差异可能来自训练条件，而不是来自被研究的模块。

`Control Variable`（控制变量）就是实验中刻意保持不变的因素。

### 24.4 如何读实验表格

看到一张实验表时，可以按以下顺序读：

```text
1. 行代表哪些模型或实验组？
2. 列代表哪些指标？
3. 是否使用了相同数据集和设置？
4. 完整模型是否优于去掉模块的版本？
5. 提升是否同时出现在 IoU、F1、Recall 等多个指标？
```

不要只盯着加粗的最大数字，还要看：

```text
模型规模是否更大
推理速度是否更慢
参数量是否更多
提升是否稳定
```

### 24.5 一个示意表

下面只是帮助理解实验逻辑的示意，不是论文原始结果：

| 实验组 | Transformer | PCS | IoU_road | Recall |
| :--- | :---: | :---: | :---: | :---: |
| A CNN baseline | 否 | 否 | 0.58 | 0.78 |
| B + Transformer | 是 | 否 | 0.62 | 0.82 |
| C + PCS | 否 | 是 | 0.61 | 0.85 |
| D full model | 是 | 是 | 0.66 | 0.88 |

从示意数据可以推测：

```text
Transformer 提升全局理解
PCS 提升道路连通和 Recall
两者结合得到更好的整体结果
```

这里的“推测”必须有对应的对照实验支持，不能只凭网络结构做结论。

### 24.6 与 v2 学习的连接

学习第二篇 SegRoadv2 时，也要问同样的问题：

```text
deformable self-attention 比普通 attention 改善了什么？
卷积模块改善了什么？
connectivity structure 是否被保留或重新设计？
去掉某个模块后指标如何变化？
```

这样读论文就不是记模块名称，而是在追踪：

```text
问题 -> 方法 -> 对照实验 -> 证据 -> 结论
```

本节记住：

```text
消融实验的核心是一次只改变一个因素，用对照证明模块贡献。
```

## 25. 基线模型与公平比较

### 25.1 什么是 Baseline

`Baseline`（基线模型）是用来比较的参考方案。

它可以是：

```text
一个简单的传统方法
一个普通 CNN 分割模型
论文改进前的基础版本
已有论文中的代表模型
```

没有 baseline，就无法判断新方法到底带来了多少提升。

### 25.2 Seg-Road 的 baseline 思路

阅读 Seg-Road 实验时，可以把比较分成三层：

```text
基础 CNN：观察普通局部特征提取的效果
CNN + Transformer：观察全局特征是否有帮助
CNN + Transformer + PCS：观察连通结构监督是否有额外帮助
```

这样就能把论文贡献拆开，而不是只看最终模型的一个分数。

### 25.3 什么是 SOTA

`SOTA` 是 `State Of The Art` 的缩写，表示某个任务或时间范围内的先进水平。

注意：

```text
SOTA 不是永远有效的称号
```

因为新数据、新模型、新评价方式出现后，先进水平会变化。阅读论文时要确认：

```text
SOTA 针对哪个数据集？
SOTA 使用什么评价指标？
SOTA 是论文发表时的结果，还是当前最新结果？
```

### 25.4 公平比较需要哪些条件

两个模型的分数只有在评价条件接近时才有意义：

```text
相同数据集
相同训练集和测试集划分
相同或可比的预处理
相同评价指标定义
相同阈值策略
```

如果一个模型使用了更大的输入图片、更强的数据增强或额外的后处理，就不能简单说它只因为网络结构更好。

### 25.5 不只看精度

论文表格除了精度指标，还可能报告：

```text
Params：参数量
FLOPs：浮点运算量
FPS：每秒处理帧数
Inference Time：单张推理时间
Memory：显存或内存占用
```

这些指标体现效率和部署成本。

```text
参数量大：通常模型容量更强，但存储和计算成本更高
FLOPs 高：计算量更大，推理可能更慢
FPS 高：处理速度更快
```

### 25.6 精度和速度的权衡

`Trade-off`（权衡）表示一个方面变好时，另一个方面可能变差。

例如：

```text
大模型：IoU 可能更高，FPS 可能更低
小模型：速度更快，精度可能略低
```

因此实际应用不一定选择分数最高的模型：

```text
无人机实时处理 -> 更关注 FPS 和显存
离线高精度制图 -> 更关注 IoU 和 Recall
```

### 25.7 读论文表格的正确问题

看到新模型超过 baseline 时，问：

```text
提升了多少？
提升来自哪个模块？
是否牺牲了速度？
是否使用了更多参数？
实验条件是否真的公平？
```

本节记住：

```text
Baseline 提供参照，SOTA 表示先进水平，公平比较决定分数是否可信。
```
