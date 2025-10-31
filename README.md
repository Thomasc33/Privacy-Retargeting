# Privacy-centric Motion Retargeting (PMR)

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Anonymizing skeleton-based motion data while preserving action utility through adversarial deep learning**

[Paper](https://github.com/Thomasc33/Privacy-Retargeting) | [Demo](https://thomasc33.github.io/Privacy-Retargeting/) | [Documentation](https://github.com/Thomasc33/Privacy-Retargeting/wiki)

## 🎯 Overview

PMR is a deep learning framework that anonymizes skeleton-based motion data by transferring motion from an original skeleton to a "dummy" skeleton. This effectively masks personally identifiable information (PII) such as body shape, gait patterns, and limb lengths while maintaining the recognizability of actions.

### Key Features

- 🔒 **Privacy Protection**: Reduces re-identification risk from 87.8% to 7.8%
- ✅ **Action Preservation**: Maintains 35.7% action recognition accuracy (vs 2-3% for baselines)
- 🎭 **Motion Retargeting**: Transfers motion between different skeletal structures
- ⚔️ **Adversarial Learning**: Disentangles identity from motion without prior knowledge of attack models
- 🚀 **Fast Inference**: 0.006s for 75 frames (~2.5s of motion)

## 📊 Results

| Method | MSE ↓ | AR Top-1 ↑ | AR Top-5 ↑ | Re-ID Top-1 ↓ | Re-ID Top-5 ↓ | Gender ↓ | Linkage ↓ |
|--------|-------|------------|------------|---------------|---------------|----------|-----------|
| Original | - | 82.2% | 85.0% | 87.8% | 97.3% | 88.7% | 69.6% |
| UNet (Moon) | 0.0834 | 2.6% | 11.1% | **3.0%** | 26.8% | **3.0%** | **50.0%** |
| DMR | **0.0071** | **49.1%** | **73.1%** | 25.7% | 60.3% | 25.7% | **50.0%** |
| **PMR (Ours)** | 0.0138 | 35.7% | 63.0% | **7.8%** | **26.4%** | **7.8%** | **50.0%** |

## 🏗️ Architecture

PMR uses a two-encoder/one-decoder architecture with adversarial and cooperative classifiers:

- **Motion Encoder (E_M)**: Captures action-specific temporal information
- **Privacy Encoder (E_P)**: Extracts skeleton structure and style attributes (PII)
- **Decoder (D)**: Reconstructs skeleton sequences from concatenated embeddings
- **Motion Classifier (M)**: Ensures action information is preserved (cooperative with E_M, adversarial with E_P)
- **Privacy Classifier (P)**: Ensures PII is captured (cooperative with E_P, adversarial with E_M)
- **Quality Controller (Q)**: GAN-style discriminator for realistic outputs

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/Thomasc33/Privacy-Retargeting.git
cd Privacy-Retargeting

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```bash
# Display CLI help
python cli.py --help

# Train PMR model on NTU-60
python cli.py train --dataset ntu60 --model pmr

# Evaluate trained model
python cli.py evaluate --model-path checkpoints/pmr_best.pt --dataset ntu60

# Anonymize a skeleton sequence
python cli.py anonymize --model-path checkpoints/pmr_best.pt \
                        --input data/sample.pkl \
                        --output data/anonymized.pkl

# Visualize results
python cli.py visualize --input data/anonymized.pkl --output video.gif

# Create comparison video
python cli.py compare --original data/sample.pkl \
                      --anonymized data/anonymized.pkl \
                      --output comparison.gif
```

## 📖 Documentation

### Training

PMR training consists of 4 stages:

1. **Stage 1: Autoencoder Warm-up** (5 paired + 20 unpaired epochs)
   - Trains encoders and decoder to reconstruct skeletons
   - Learns basic motion representation

2. **Stage 2: Classifier Pre-training** (20 paired + 50 unpaired epochs)
   - Pre-trains motion and privacy classifiers
   - Prepares for adversarial training

3. **Stage 3: Unpaired Cooperative-Adversarial** (100 epochs)
   - Adversarial training to disentangle motion and identity
   - Quality controller ensures realistic outputs

4. **Stage 4: Paired Motion Retargeting** (100 epochs)
   - Cross-reconstruction for anonymization
   - Fine-tunes with triplet and latent consistency losses

```bash
# Full training with custom parameters
python cli.py train \
    --dataset ntu60 \
    --model pmr \
    --batch-size 32 \
    --lr 1e-5 \
    --device cuda:0 \
    --checkpoint-dir checkpoints \
    --use-mlflow
```

### Configuration

Modify `configs/default_config.py` or create custom configs:

```python
from configs.default_config import Config

config = Config()
config.training.batch_size = 64
config.training.lr = 5e-5
config.training.alpha_emb = 1.0  # Increase adversarial strength
```

### Python API

```python
from models.pmr import PMRModel
import torch

# Load model
model = PMRModel(T=75, encoded_channels=(256, 32))
model.load_state_dict(torch.load('checkpoints/pmr_best.pt'))
model.eval()

# Anonymize skeleton
original = torch.randn(1, 75, 25, 3)  # (batch, frames, joints, coords)
dummy = torch.randn(1, 75, 25, 3)
anonymized = model.cross_reconstruct(original, dummy)

# Get embeddings for analysis
motion_emb, privacy_emb = model.get_embeddings(original)
```

## 📁 Project Structure

```
Privacy-Retargeting/
├── models/              # Model architectures
│   ├── pmr.py          # PMR encoders and decoder
│   ├── classifiers.py  # Motion/Privacy classifiers, Quality controller
│   └── sgn_wrapper.py  # SGN model wrapper for evaluation
├── training/            # Training scripts
│   ├── trainer.py      # Main training loop
│   ├── evaluator.py    # Evaluation metrics
│   └── losses.py       # Loss functions
├── data/                # Data loading and preprocessing
│   ├── dataset.py      # PyTorch datasets
│   └── preprocessing.py # Data preprocessing utilities
├── utils/               # Utility functions
│   ├── visualization.py # Skeleton visualization
│   └── metrics.py      # Evaluation metrics
├── configs/             # Configuration files
│   └── default_config.py # Default hyperparameters
├── scripts/             # Helper scripts
│   ├── download_data.sh # Download NTU dataset
│   └── preprocess.py   # Preprocess data
├── docs/                # Documentation
├── fig/                 # Paper figures
├── pretrained/          # Pretrained models
├── cli.py               # Command-line interface
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

## 🎓 Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{carr2025pmr,
  title={Privacy-centric Deep Motion Retargeting for Anonymization of Skeleton-Based Motion Visualization},
  author={Carr, Thomas and Xu, Depeng and Yuan, Shuhan and Lu, Aidong},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  year={2025}
}
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

- Thomas Carr - tcarr23@charlotte.edu
- Project Link: [https://github.com/Thomasc33/Privacy-Retargeting](https://github.com/Thomasc33/Privacy-Retargeting)

## 🙏 Acknowledgments

- NTU RGB+D dataset creators
- SGN (Semantics-Guided Neural Network) authors
- PyTorch team

## 📄 License

This project is licensed under the MIT License.

## 📚 Citation

**Published at ICCV 2025** - International Conference on Computer Vision

If you use this work in your research, please cite:

```bibtex
@InProceedings{Carr_2025_ICCV,
    author    = {Carr, Thomas and Xu, Depeng and Yuan, Shuhan and Lu, Aidong},
    title     = {Privacy-centric Deep Motion Retargeting for Anonymization of Skeleton-Based Motion Visualization},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2025},
    pages     = {13162-13170}
}
```

**Paper:** [CVF Open Access](https://openaccess.thecvf.com/content/ICCV2025/papers/Carr_Privacy-centric_Deep_Motion_Retargeting_for_Anonymization_of_Skeleton-Based_Motion_Visualization_ICCV_2025_paper.pdf)

