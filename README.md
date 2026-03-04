# Privacy-centric Motion Retargeting (PMR)

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![ICCV 2025](https://img.shields.io/badge/ICCV-2025-green.svg)](https://openaccess.thecvf.com/content/ICCV2025/papers/Carr_Privacy-centric_Deep_Motion_Retargeting_for_Anonymization_of_Skeleton-Based_Motion_Visualization_ICCV_2025_paper.pdf)

> **Anonymizing skeleton-based motion data while preserving action utility through adversarial deep learning**

[Paper](https://openaccess.thecvf.com/content/ICCV2025/papers/Carr_Privacy-centric_Deep_Motion_Retargeting_for_Anonymization_of_Skeleton-Based_Motion_Visualization_ICCV_2025_paper.pdf) | [Demo](https://thomasc33.github.io/Privacy-Retargeting/) | [Documentation](https://github.com/Thomasc33/Privacy-Retargeting/wiki)

## Overview

PMR is a deep learning framework that anonymizes skeleton-based motion data by transferring motion from an original skeleton to a "dummy" skeleton. This effectively masks personally identifiable information (PII) such as body shape, gait patterns, and limb lengths while maintaining the recognizability of actions.

### Key Features

- **Privacy Protection**: Reduces re-identification risk from 87.8% to 7.8%
- **Action Preservation**: Maintains 35.7% action recognition accuracy (vs 2-3% for baselines)
- **Motion Retargeting**: Transfers motion between different skeletal structures
- **Adversarial Learning**: Disentangles identity from motion without prior knowledge of attack models
- **Fast Inference**: 0.006s for 75 frames (~2.5s of motion)

## Results

| Method | MSE | AR Top-1 | AR Top-5 | Re-ID Top-1 | Re-ID Top-5 | Gender | Linkage |
|--------|-----|----------|----------|-------------|-------------|--------|---------|
| Original | - | 82.2% | 85.0% | 87.8% | 97.3% | 88.7% | 69.6% |
| UNet (Moon) | 0.0834 | 2.6% | 11.1% | **3.0%** | 26.8% | **3.0%** | **50.0%** |
| DMR | **0.0071** | **49.1%** | **73.1%** | 25.7% | 60.3% | 25.7% | **50.0%** |
| **PMR (Ours)** | 0.0138 | 35.7% | 63.0% | **7.8%** | **26.4%** | **7.8%** | **50.0%** |

## Architecture

PMR uses a two-encoder/one-decoder architecture with adversarial and cooperative classifiers:

- **Motion Encoder (E_M)**: Captures action-specific temporal information
- **Privacy Encoder (E_P)**: Extracts skeleton structure and style attributes (PII)
- **Decoder (D)**: Reconstructs skeleton sequences from concatenated embeddings
- **Motion Classifier (M)**: Cooperative with E_M, adversarial with E_P
- **Privacy Classifier (P)**: Cooperative with E_P, adversarial with E_M
- **Quality Controller (Q)**: GAN-style discriminator for realistic outputs

## Quick Start

### Automated Setup (Recommended)

The interactive setup script handles everything: Python version management, virtual environment, dependencies, and NTU RGB+D skeleton data download.

```bash
git clone https://github.com/Thomasc33/Privacy-Retargeting.git
cd Privacy-Retargeting
./setup.sh
```

The script will:
1. Install Python 3.11 via [pyenv](https://github.com/pyenv/pyenv) if needed (PyTorch requires Python 3.8-3.12)
2. Create and activate a virtual environment
3. Install all dependencies from `requirements.txt`
4. Download NTU RGB+D 60/120 skeleton data from Google Drive (~11 GB total)
5. Extract `.skeleton` files to `NTU/SGN/raw_skeletons/` (handles varying zip structures)
6. Delete zip archives to save disk space
7. Run a full environment validation

### Manual Setup

```bash
# Requires Python 3.8-3.12 (3.11 recommended)
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Data Preprocessing

After downloading the raw skeleton files (via `setup.sh` or manually), run the SGN preprocessing pipeline to generate the data files used for training:

```bash
cd NTU/SGN
python get_raw_skes_data.py       # Extract raw skeleton data
python get_raw_denoised_data.py   # Denoise skeleton sequences
python seq_transformation.py      # Generate X_full.pkl
cd ../..
```

This produces `NTU/SGN/X_full.pkl` (NTU-60). For NTU-120, update `skes_available_name.txt` to include S018-S032 setups and re-run to generate `X_full_120.pkl`.

### Usage

```bash
# Display CLI help
python cli.py --help

# Train PMR model on NTU-60
python cli.py train --dataset ntu60 --model pmr --device cuda:0

# Test a trained model
python cli.py evaluate --model-path checkpoints/pmr_best.pt --dataset ntu60

# Anonymize a skeleton sequence
python cli.py anonymize --model-path pretrained/PMR.pt \
                        --input data/sample.pkl \
                        --output data/anonymized.pkl

# Visualize results
python cli.py visualize --input data/anonymized.pkl --output video.gif

# Create comparison video
python cli.py compare --original data/sample.pkl \
                      --anonymized data/anonymized.pkl \
                      --output comparison.gif
```

### Python API

```python
from models.pmr import PMRModel
import torch

# Load pretrained model
model = PMRModel(T=75, encoded_channels=(128, 16))
model.load_state_dict(torch.load('pretrained/PMR.pt'))

# Anonymize: transfer motion from original onto dummy skeleton
original = torch.randn(1, 75, 25, 3)  # (batch, frames, joints, coords)
dummy = torch.randn(1, 75, 25, 3)
anonymized = model.cross_reconstruct(original, dummy)

# Get embeddings for analysis
motion_emb, privacy_emb = model.get_embeddings(original)
```

## Training

PMR training consists of 4 stages:

1. **Stage 1: Autoencoder Warm-up** (5 paired + 20 unpaired epochs)
   - Trains encoders and decoder to reconstruct skeletons

2. **Stage 2: Classifier Pre-training** (20 paired + 50 unpaired epochs)
   - Pre-trains motion and privacy classifiers

3. **Stage 3: Unpaired Cooperative-Adversarial** (100 epochs)
   - Adversarial training to disentangle motion and identity

4. **Stage 4: Paired Motion Retargeting** (100 epochs)
   - Cross-reconstruction for anonymization with triplet and latent consistency losses

```bash
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
from configs.default_config import get_default_config, get_ntu120_config, get_dmr_config

config = get_default_config()              # NTU-60, 60 action classes
config = get_ntu120_config()               # NTU-120, 120 action classes
config = get_dmr_config()                  # DMR baseline (alpha_emb=0.0)
```

### Pretrained Models

Six pretrained checkpoints are included in `pretrained/`:

| Checkpoint | Dataset | Model | Description |
|------------|---------|-------|-------------|
| `PMR.pt` | NTU-60 | PMR | Full adversarial model |
| `NTU120.pt` | NTU-120 | PMR | Full adversarial model |
| `DMR.pt` | NTU-60 | DMR | Baseline (no adversarial) |
| `DMR_NTU120.pt` | NTU-120 | DMR | Baseline (no adversarial) |
| `ETRI.pt` | ETRI | PMR | Custom dataset model |
| `ETRI_DMR.pt` | ETRI | DMR | Custom dataset baseline |

## Project Structure

```
Privacy-Retargeting/
├── models/                  # Model architectures (implemented)
│   ├── pmr.py              # MotionEncoder, PrivacyEncoder, Decoder, PMRModel
│   ├── classifiers.py      # MotionClassifier, PrivacyClassifier, QualityController
│   └── sgn_wrapper.py      # SGN model wrapper
├── configs/
│   └── default_config.py   # Dataclass-based hyperparameter config
├── pretrained/              # 6 pretrained checkpoints
├── SGN/                     # Semantics-Guided Neural Network (used for assessment)
│   ├── model.py
│   └── pretrained/          # Pre-trained SGN models
├── NTU/                     # NTU dataset utilities
│   ├── datagen.ipynb        # Data generation notebook
│   └── SGN/                 # SGN preprocessing pipeline
│       ├── get_raw_skes_data.py
│       ├── get_raw_denoised_data.py
│       ├── seq_transformation.py
│       ├── raw_skeletons/   # .skeleton files (created by setup.sh)
│       └── statistics/      # Dataset metadata
├── legacy_scripts/          # Original monolithic implementation (reference)
├── notebooks/               # Archived Jupyter notebooks
├── fig/                     # Paper figures
├── cli.py                   # Command-line interface
├── setup.sh                 # Interactive setup script
├── requirements.txt         # Python dependencies
└── README.md
```

## Citation

**Published at ICCV 2025** - International Conference on Computer Vision

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

[Paper (CVF Open Access)](https://openaccess.thecvf.com/content/ICCV2025/papers/Carr_Privacy-centric_Deep_Motion_Retargeting_for_Anonymization_of_Skeleton-Based_Motion_Visualization_ICCV_2025_paper.pdf)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Contact

- Thomas Carr - tcarr23@charlotte.edu
- Project Link: [https://github.com/Thomasc33/Privacy-Retargeting](https://github.com/Thomasc33/Privacy-Retargeting)

## Acknowledgments

- NTU RGB+D dataset creators
- SGN (Semantics-Guided Neural Network) authors
- PyTorch team

## License

This project is licensed under the MIT License.
