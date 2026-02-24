# Symmetry-Aware Framework for Bidirectional Modeling of 3D Photonic Crystals

This repository contains the official implementation of the paper: **"Symmetry-Aware Framework for Bidirectional Modeling of 3D Photonic Crystals and Band Structures"**.

## Overview

This project provides a closed-loop deep learning framework for the inverse design of 3D Photonic Crystals (PhCs). It addresses the "curse of dimensionality" in 3D material design by integrating crystallographic symmetry groups directly into the generative process.

The framework consists of two main components:

1. **Forward Prediction Model**: A **ResNet-Transformer** architecture (`CNNNetWork.py`) that accurately predicts photonic band structures (PBS) from 3D voxelized geometries.
2. **Inverse Design Model**: A **Symmetry-Aware Conditional Diffusion Model** (DDIM/DDPM) that generates physically realizable 3D structures from target band structures, enforcing strict space group symmetries (Nos. 195-230).

## Project Structure

Recommended directory structure:

```
Project_Root/
├── data/
│   └── saved_dataset/          # Dataset files (Tensor.pt, Label.pt, etc.)
│       ├── A_matrices.pt
│       ├── crystal_name.pt
│       ├── epsvalue.pt
│       ├── K_a_features.pt
│       ├── Label.pt
│       ├── metadata.pt
│       └── Tensor.pt
│   └── dataset_test.py         # Dataset loading utility
│   └── test_data.pt            # Test set sample
├── Models/
│   ├── ForwardPrediction/
│   │   ├── CNNNetWork.py       # Combined ResNet + Transformer architecture
│   │   ├── ResNet.py           # 3D ResNet backbone (based on MedicalNet)
│   │   └── train_cnn.py        # Training script for forward model
│   ├── InverseDesign/
│   │   ├── DiffusionModel.py   # 3D U-Net with Time/Freq embeddings
│   │   ├── DiffusionProcess.py # Gaussian Diffusion & DDIM sampling logic
│   │   ├── EMA.py              # Exponential Moving Average
│   │   ├── LossFunction.py     # Custom loss (MSE + L1 + Perceptual)
│   │   └── train_Diffusion.py  # Main diffusion training logic
│   │   └── Visualization.py    # Results visualization
├── utils/
│   ├── Enforce_Symmetry.py     # Core logic for Space Group symmetry enforcement
│   ├── reconstruct_diffusion_test.py # Post-processing, thresholding & visualization
│   └── SymmetryAwareAtt.py     # Symmetry-Aware Attention mechanism
├── Weights/                    # Pre-trained checkpoints
│   ├── diffusionunet_checkpoint100.pth
│   └── resnet_model.pth
├── requirements.txt            # Python dependencies
└── Save_Sample.py              # Main inference script for generation
```

## Data Preparation

The dataset should be placed in `data/saved_dataset/`. The framework expects the following PyTorch tensors:

- **Tensor.pt**: 3D voxelized structures (Shape: `[N, 1, 60, 60, 60]`).

- **Label.pt**: Band structure data (Frequencies).

- **crystal_name.pt**: List of space group names/IDs (e.g., "No225_FCC_BiF3").

- **K_a_features.pt** & **A_matrices.pt**: K-path and Lattice information.

## Usage

### 1. Inverse Design (Generation & Selection)

To generate 3D structures from a target band structure, use the `Save_Sample.py` script. This script performs a "Generate-Evaluate-Select" loop:

1. Generates candidates using the Diffusion model (with DDIM acceleration).

2. Enforces strict space group symmetry and performs morphological post-processing.

3. Predicts the band structure of generated candidates using the Forward CNN.

4. Selects the best matching structures based on MSE loss.

**Example Command:**

```
python Save_Sample.py \
  --num-generations 1000 \
  --num-best 5 \
  --sample-idx 0 \
  --batch-size 1 \
  --use-ddim \
  --ddim-steps 20 \
  --save-images \
  --out-dir Save_Sample_out \
  --diffusion-ckpt Weights/diffusionunet_checkpoint100.pth \
  --cnn-ckpt Weights/resnet_model.pth
```

**Key Parameters:**

- `--num-generations`: Total number of candidates to generate (e.g., 1000).

- `--num-best`: Number of top candidates to save (e.g., 5).

- `--use-ddim`: Enable DDIM for faster sampling (20 steps vs 1000+ steps).

- `--save-images`: Save 3D visualizations (PNG) using PyVista.

- `--sample-idx`: Index of the target sample from the test dataset to reconstruct.

- `--seg-p`: Percentile threshold for binarization (default: 92.0).

### 2. Training the Forward Model (ResNet)

To train the ResNet+Transformer model for band structure prediction:

```code
python -m Models.ForwardPrediction.train_resnet
```

### 3. Training the Inverse Model (Diffusion)

To train the Symmetry-Aware Diffusion model:

```code
python -m Models.InverseDesign.train_Diffusion
```

Note: The script supports multi-GPU training and will automatically detect available GPUs.

## Key Features

- **Symmetry-Aware Attention (SAA)**: Located in `utils/SymmetryAwareAtt.py`. Explicitly aggregates features across 48 symmetry operations of the cubic point group within the neural network layers.

- **Symmetry Enforcement**: Located in `utils/Enforce_Symmetry.py`. Uses `spglib` logic to strictly enforce crystallographic constraints (Space Groups 195-230) on generated voxels during post-processing, ensuring physical validity.

- **Robust Post-Processing**: Located in `utils/reconstruct_diffusion_test.py`. Includes percentile thresholding, connected component analysis, and boundary smoothing to convert grayscale diffusion outputs into binary material masks.


