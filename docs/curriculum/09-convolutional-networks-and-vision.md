# 09 — Convolutional Networks & Computer Vision

> Back to [index](../../README.md) · Prev: [08 Training](08-training-and-regularizing-networks.md) · Next: [10 RNNs & Sequences](10-rnn-and-sequence-modeling.md)

## 1. Convolution — the operation

A small kernel (filter) slides over the input, computing dot products at every position:

```
output[i,j] = ΣΣ input[i+u, j+v] · kernel[u,v] + bias
```

**Inductive biases exploited:** **locality** (nearby pixels relate) and **translation equivariance** (a cat shifted is still a cat; the feature map shifts with it). **Parameter sharing** means a edge detector works everywhere → orders of magnitude fewer parameters than a dense layer on images.

Key knobs: **kernel size** (3×3 standard), **stride** (step; stride 2 downsamples), **padding** (preserve borders), **dilation** (holes → bigger receptive field, e.g. DeepLab), **channels** (one kernel per output feature; stack many kernels = feature maps). Receptive field grows with depth.

**Pooling:** max-pool (dominant) / average-pool — spatial downsampling, mild translation invariance, cheaper compute. Modern trend: strided convs replace pooling.

## 2. Architecture evolution (know this lineage)

| Year | Net | Breakthrough |
|---|---|---|
| 1998 | **LeNet-5** | blueprint: conv→pool→fc |
| 2012 | **AlexNet** | ImageNet win, ReLU, dropout, 2-GPU — kickoff of deep learning |
| 2014 | **VGG** | tiny 3×3 stacks, depth matters, very uniform |
| 2014 | **GoogLeNet/Inception** | parallel multi-scale kernels, 1×1 bottlenecks |
| 2015 | **ResNet** | **skip connections** — train 100+ layers; solves degradation |
| 2017 | **DenseNet, MobileNet, EfficientNet** | dense links; depthwise-separable (mobile); compound scaling |
| 2020+ | **ConvNeXt** | modernized CNN competitive with ViT |
| 2020+ | **Vision Transformer (ViT)** | image → 16×16 patches → standard transformer; needs big data (or pretrained) |
| 2023+ | **Swin, hierarchical ViTs** | windowed attention, multi-scale |

**Takeaway:** ResNets taught us identity shortcuts keep gradients alive — the same trick became *the* transformer ingredient ([11](11-transformers-and-foundation-models.md)).

## 3. Vision tasks taxonomy

- **Classification:** ImageNet-style; top-1/top-5 accuracy.
- **Object detection:** predict boxes + classes.
  - *Two-stage:* R-CNN → Fast → **Faster R-CNN** (RPN).
  - *One-stage:* **YOLO** family (real-time), SSD, RetinaNet (focal loss).
  - Metrics: **IoU**, **mAP@0.5:0.95**.
- **Semantic segmentation:** per-pixel class (FCN, U-Net, DeepLab w/ atrous conv).
- **Instance segmentation:** per-pixel *instance* (**Mask R-CNN** = Faster R-CNN + mask branch).
- **Panoptic segmentation:** both — stuff + things.
- **Keypoint/pose, tracking (SORT/DeepSORT), OCR, depth estimation, image retrieval.**

## 4. Transfer learning in practice (the default workflow)

```python
# 1. Start from pretrained backbone (ImageNet weights / CLIP / DINOv2)
# 2. Freeze early blocks, replace the head for your K classes
# 3. Train head (lr ~1e-3), then unfreeze top blocks ("fine-tune") with lr ~1e-5
```

- **Why it works:** early features (edges/textures) are universal; task-specific knowledge lives late.
- Self-supervised pretrained backbones (**DINOv2, MAE, CLIP**) now beat pure ImageNet supervision for many tasks. CLIP ([20](20-multimodal-ai.md)) also enables **zero-shot classification**: compare image embedding to text embeddings of class names.

## 5. Data-centric vision details

- Input normalization to model's expected mean/std; consistent train/val/aug pipelines (use `albumentations`/torchvision.transforms v2).
- Augmentation ladder: flips/crops → color jitter → RandAugment → Mixup/CutMix.
- Watch class imbalance (focal loss, weighted sampling) and label noise (small-loss trick, confident learning).

## Mastery Checklist

- [ ] Computes output shapes of a conv layer given kernel/stride/padding
- [ ] Explains locality, translation equivariance, and parameter sharing with numbers
- [ ] Recites the AlexNet→VGG→ResNet lineage and *why* each mattered
- [ ] Selects detection/segmentation heads and their metrics for a given task
- [ ] Implements the freeze→unfreeze transfer-learning schedule with proper LRs
- [ ] Explains when ViT beats CNN and what data regime each prefers
