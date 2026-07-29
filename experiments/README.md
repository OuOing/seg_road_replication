# Experiment Recipes

This folder records reproducible command recipes for the current Seg-Road
DeepGlobe experiments. The commands are designed for the local Apple MPS
environment; run outputs are written under `runs/`, which is ignored by git.

## 1. Threshold Sweep For The Epoch 8 Checkpoint

Purpose: choose a validation threshold for converting sigmoid probabilities to
binary road masks. This does not change model weights.

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

After it finishes, regenerate the summary table with a threshold-selected row:

```bash
python3 code/summarize_experiment.py \
  --checkpoint runs/deepglobe/segroad-s-full-probe/best.pt \
  --model-size s \
  --threshold-csv runs/deepglobe/segroad-s-full-probe/threshold_sweep_val.csv \
  --select-metric f1 \
  --output-md runs/deepglobe/segroad-s-full-probe/summary.md
```

## 2. Low-Learning-Rate Short Finetune

Purpose: resume from epoch 8 and refine with smaller optimizer steps. This can
improve precision if the model is currently over-predicting roads.

```bash
python3 code/train.py \
  --image-dir data/deepglobe/images \
  --mask-dir data/deepglobe/masks \
  --train-list data/deepglobe_formal/splits/train.txt \
  --val-list data/deepglobe_formal/splits/val.txt \
  --output-dir runs/deepglobe/segroad-s-lr1e-4-finetune \
  --resume runs/deepglobe/segroad-s-full-probe/best.pt \
  --resume-learning-rate 1e-4 \
  --reset-best-on-resume \
  --eval-threshold 0.5 \
  --model-size s \
  --epochs 12 \
  --batch-size 4 \
  --learning-rate 3e-4 \
  --seg-pos-weight 10 \
  --pcs-pos-weight 10 \
  --pcs-alpha 0.2 \
  --dice-weight 1 \
  --num-workers 2 \
  --device mps \
  --log-interval 200
```

Interpretation checklist:

```text
IoU/F1 should beat or approach epoch 8.
Precision should increase if false positives are reduced.
pred+ should move closer to target+ without collapsing recall.
If recall drops sharply, the finetune became too conservative.
```

## 3. PCS Ablation

Purpose: verify whether the Pixel Connectivity Structure loss contributes to
road continuity. Only `--pcs-alpha` changes; the rest should match the baseline
as closely as possible.

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
  --device mps \
  --log-interval 200
```

Interpretation checklist:

```text
Compare IoU/F1 with segroad-s-full-probe.
Inspect visualizations for more blue FN gaps or broken roads.
If pixel metrics are close but roads break more often, PCS helps topology.
```
