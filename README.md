# TTA-Flow: Flow Matching for OCT Image Reconstruction

Flow Matching model for generating optical coherence tomography (OCT) images across different imaging devices. This project implements conditional flow matching trained on volumetric OCT data from the RETOUCH dataset.

![Visual Abstract](data/figures/visual_abstract_paper.pdf)

# Project Overview

This repository contains a complete pipeline for:
- **Training** Flow Matching models on 3D OCT volumes converted to 2D slices
- **Testing** on multiple OCT imaging devices (Spectralis, Cirrus, Topcon)
- **Inference** on new data with pre-trained checkpoints

The model architecture uses a UNet backbone with configurable parameters, optimized for multi-modal medical imaging generation tasks.

# Repository Structure

```
├── src/                          # Main source code
│   ├── train.py                 # Training script
│   ├── test.py                  # Inference script
│   ├── train.sh                 # Training wrapper
│   ├── inference.sh             # Inference wrapper
│   ├── models/                  # Model definitions (UNet wrapper)
│   └── util/                    # Utilities
│       ├── datasets.py          # Data loading and augmentation
│       ├── model_util.py        # Model initialization and checkpointing
│       ├── losses.py            # Loss functions
│       ├── metrics.py           # Evaluation metrics
│       └── checkpoint_manager.py
│
├── preferences/                 # Hydra configuration files
│   ├── config.yaml              # Main configuration
│   ├── data/                    # Dataset configs (retouch_*.yaml)
│   ├── model/                   # Model configs
│   ├── optimizer/               # Optimizer configs
│   ├── loss/                    # Loss configs
│   ├── transforms/              # Augmentation transform configs
│   └── train_parameters/        # Training hyperparameters
│
├── data/
│   ├── dataset_configs/         # CSV files mapping volume/mask paths
│   └── figures/                 # Generated visualizations
│
├── runs/                        # Training outputs (created at runtime)
│   └── {timestamp}-{comment}/
│       ├── checkpoints/         # Model weights at various steps
│       ├── inference/           # Inference outputs
│       └── val/                 # Validation visualizations
│
├── notebooks/                   # Jupyter notebooks for data exploration
├── Dockerfile                   # Docker image definition
└── requirements.txt             # Python dependencies
```

# Data Requirements

### Dataset Structure

Data must be organized with the following structure:

```
/path/to/retouch/
├── volumes/
│   ├── image_001.npy           # 3D volumetric data (H, W, D)
│   ├── image_002.npy
│   └── ...
└── annotations/
    ├── mask_001.npy            # 3D binary masks (H, W, D)
    ├── mask_002.npy
    └── ...
```

### Dataset Configuration

Create a CSV file in `data/dataset_configs/` with the following columns:

```csv
volume,mask
volumes/image_001.npy,annotations/mask_001.npy
volumes/image_002.npy,annotations/mask_002.npy
```

**Notes:**
- The `mask` column is optional and only used for downstream evaluation (Dice Similarity Coefficient).
- For Flow Matching training/testing, only the `volume` column is required.
- Pre-configured datasets: `retouch_spectralis`, `retouch_cirrus`, `retouch_topcon`
- Data should be saved as `.npy` NumPy arrays

# Setup

### Building the Docker Image

```bash
docker build -t tta-flow:latest .
```

### Prerequisites

- Docker with NVIDIA GPU support (`nvidia-docker` or Docker 19.03+ with `--gpus` flag)
- CUDA 12.6+ compatible GPU
- The RETOUCH dataset (or your own OCT volumes in the specified format)

# Training

### Basic Training Command

```bash
docker run -it --rm --gpus all \
  -v /path/to/retouch:/app/data/retouch \
  -v $(pwd)/data/dataset_configs:/app/data/dataset_configs \
  -v $(pwd)/runs:/app/runs \
  tta-flow:latest \
  ./src/train.sh retouch_spectralis my_experiment
```

**Arguments:**
- `retouch_spectralis` - Dataset configuration (see `preferences/data/`)
- `my_experiment` - Experiment name for logging and checkpoint organization

### Customization

Override Hydra configuration via command-line arguments:

```bash
docker run -it --rm --gpus all \
  -v /path/to/retouch:/app/data/retouch \
  -v $(pwd)/data/dataset_configs:/app/data/dataset_configs \
  -v $(pwd)/runs:/app/runs \
  tta-flow:latest \
  ./src/train.sh retouch_spectralis my_experiment \
    train_parameters.num_steps=50000 \
    optimizer.lr=1e-4
```

### Output

Training outputs are saved to `runs/{timestamp}-{comment}/`:
- `checkpoints/` - Model weights saved at regular intervals
- `events.out.tfevents.*` - TensorBoard logs
- `val/` - Validation visualizations during training

# Inference

### Basic Inference Command

```bash
docker run -it --rm --gpus all \
  -v /path/to/retouch:/app/data/retouch \
  -v $(pwd)/data/dataset_configs:/app/data/dataset_configs \
  -v $(pwd)/runs:/app/runs \
  tta-flow:latest \
  ./src/inference.sh runs/2026-03-05_11-47-31-spectralis_128_test retouch_spectralis
```

**Arguments:**
- `runs/2026-03-05_11-47-31-spectralis_128_test` - Path to trained model directory
- `retouch_spectralis` - Dataset configuration for inference

### Output

Inference results are saved to `{experiment_path}/inference/outputs/`:
- `pred_*.npy` - Generated images as NumPy arrays

# Configuration

All training hyperparameters and model settings are managed through Hydra YAML configuration files in the `preferences/` directory:

| Directory | Purpose |
|-----------|---------|
| `model/` | Model architecture (channels, depth, attention, etc.) |
| `optimizer/` | Optimizer settings and learning rate |
| `loss/` | Loss function configuration |
| `transforms/` | Data augmentation pipelines |
| `train_parameters/` | Training specifics (steps, batch size, seed) |
| `data/` | Dataset-specific configurations |

Modify these files or override via command-line arguments (see customization example above).

# Development

### Local Setup (without Docker)

1. Create a Python environment: `python3.12 -m venv venv`
2. Activate: `source venv/bin/activate`
3. Install: `pip install -r requirements.txt`
4. Run training: `python src/train.py data=retouch_spectralis`

### TensorBoard Monitoring

```bash
tensorboard --logdir runs/
```

Then open `http://localhost:6006` in your browser.

# Dependencies

Key dependencies include:
- PyTorch with CUDA 12.6 support
- torchcfm (Conditional Flow Matching)
- Hydra (configuration management)
- Albumentations (data augmentation)
- PyTorch Lightning (training utilities)

See `requirements.txt` for the complete list.

# References

```bibtex
@ARTICLE{bogunovic2019retouch,
  author={Bogunović, Hrvoje and Venhuizen, Freerk and Klimscha, Sophie and Apostolopoulos, Stefanos and Bab-Hadiashar, Alireza and Bagci, Ulas and Beg, Mirza Faisal and Bekalo, Loza and Chen, Qiang and Ciller, Carlos and Gopinath, Karthik and Gostar, Amirali K. and Jeon, Kiwan and Ji, Zexuan and Kang, Sung Ho and Koozekanani, Dara D. and Lu, Donghuan and Morley, Dustin and Parhi, Keshab K. and Park, Hyoung Suk and Rashno, Abdolreza and Sarunic, Marinko and Shaikh, Saad and Sivaswamy, Jayanthi and Tennakoon, Ruwan and Yadav, Shivin and De Zanet, Sandro and Waldstein, Sebastian M. and Gerendas, Bianca S. and Klaver, Caroline and Sánchez, Clara I. and Schmidt-Erfurth, Ursula},
  journal={IEEE Transactions on Medical Imaging}, 
  title={{RETOUCH}: The Retinal OCT Fluid Detection and Segmentation Benchmark and Challenge}, 
  year={2019},
  volume={38},
  number={8},
  pages={1858-1874},
  keywords={Retina;Image segmentation;Diseases;Biomedical imaging;Image analysis;Fluids;Benchmark testing;Evaluation;image segmentation;image classification;optical coherence tomography;retina},
  doi={10.1109/TMI.2019.2901398}
}
```