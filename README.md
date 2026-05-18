# LLIE-LMT: Semi-supervised Low-light Image Enhancement Using Latent Mean-Teacher

This repository is the official implementation of the paper *"LLIE-LMT: Semi-supervised Low-light Image Enhancement Using Latent Mean-Teacher"*.

## Abstract

Although recent low-light image enhancement (LLIE) methods have made significant advancements, they still face challenges in terms of low image quality and weak generalization ability when applied to complex scenarios. Facing these issues, we propose a semi-supervised low-light image enhancement method, called LLIE-LMT. Specifically, we introduce a latent mean-teacher framework that integrates labeled and unlabeled data, as well as their latent vectors, into the network learning. Meanwhile, we use a mean-teacher-assisted Gaussian process (GP) learning strategy to establish a connection between the latent vectors obtained from the labeled and unlabeled data, which improves the generalization ability. To guide the learning process, we utilize an assisted Gaussian process regression (GPR) loss. Moreover, we design a pseudo-ground truth memory (PGTM) module to ensure the reliability of the network learning and a metric evaluation strategy (MES) to select non-reference image quality assessment (NR-IQA) metric for PGTM. To demonstrate our method's generalization ability and effectiveness, we construct a large-scale low-light vehicle model (LLVM) dataset. We apply LLIE-LMT to multiple LLIE datasets and high-level vision tasks. Experiment results demonstrate that our method achieves high generalization performance and image quality.

## Dataset Structure

```
├── data
│   ├── Labeled data    # Training
│   │   └── our485
│   │       ├── low
│   │       └── high
│   ├── Unlabeled data  # Training
│   │   └── low
│   └── val             # Testing
│       ├── low
│       └── high
```

### Datasets

The LLVM-Cap dataset can be downloaded from [Baidu Pan](https://pan.baidu.com/s/1HdrxoGe0t-SX6lbSdmqnlQ) (Extraction code: `3u3e`).
The LLVM-Syn dataset can be downloaded from [Baidu Pan](https://pan.baidu.com/s/1QOp2RTuXiNjowvT_mpxiyQ) (Extraction code: `8i2g`).
## Quick Start

### Train

```
python train.py
```

### Test

```
python test.py
```

## File Description

| File | Description |
|------|-------------|
| `GP.py` | Implementation of the learning strategy. |
| `trainer.py` | Training pipeline. |
| `losses.py` | Loss functions. |

`assets/Original/` contains the original figures used in our paper.
`assets/LLVM_Cap` & `assets/LLVM_Syn` contain the sample images from our dataset.
## Acknowledgements

* Thanks to the creators of [IQA-pytorch](https://github.com/chaofengc/IQA-PyTorch) for the awesome work.
* Thanks to the creators [Semi-UIR](https://github.com/Huang-ShiRui/Semi-UIR), [Syn2real](https://github.com/rajeevyasarla/Syn2Real), and [MIMO-UNet](https://github.com/chosj95/MIMO-UNet) for providing such elegant codes.
