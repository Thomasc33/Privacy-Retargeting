# Privacy-Retargeting Project Structure

This document describes the organization of the Privacy-centric Motion Retargeting (PMR) codebase after refactoring.

## Directory Structure

```
Privacy-Retargeting/
├── cli.py                      # Interactive command-line interface
├── README.md                   # Project documentation
├── index.html                  # GitHub Pages website
├── requirements.txt            # Python dependencies
├── CNAME                       # GitHub Pages custom domain
│
├── models/                     # Model architectures
│   ├── __init__.py            # Package initialization
│   ├── pmr.py                 # PMR encoders and decoder
│   ├── classifiers.py         # Motion/Privacy classifiers, Quality controller
│   └── sgn_wrapper.py         # SGN model wrapper for evaluation
│
├── configs/                    # Configuration files
│   └── default_config.py      # Default hyperparameters and settings
│
├── training/                   # Training scripts (to be implemented)
│   ├── trainer.py             # Main training loop
│   ├── evaluator.py           # Evaluation metrics
│   └── losses.py              # Loss functions
│
├── data/                       # Data loading and preprocessing (to be implemented)
│   ├── dataset.py             # PyTorch datasets
│   └── preprocessing.py       # Data preprocessing utilities
│
├── utils/                      # Utility functions (to be implemented)
│   ├── visualization.py       # Skeleton visualization
│   └── metrics.py             # Evaluation metrics
│
├── scripts/                    # Helper scripts
│   └── (to be added)
│
├── docs/                       # Documentation
│   ├── paper.tex              # Main paper LaTeX source
│   └── appendix.tex           # Appendix LaTeX source
│
├── fig/                        # Paper figures and visualizations
│   ├── Intro.png              # Overview figure
│   ├── PMR.png                # Architecture diagram
│   ├── cross.png              # Cross-reconstruction illustration
│   ├── action clustering.png  # Motion embedding visualization
│   ├── actor clustering.png   # Privacy embedding visualization
│   ├── res scatter.png        # Results scatter plot
│   ├── embedding priv acc.png # Privacy classifier accuracy
│   └── embedding utility acc.png # Utility classifier accuracy
│
├── pretrained/                 # Pre-trained model checkpoints
│   ├── PMR.pt                 # PMR model (NTU-60)
│   ├── NTU120.pt              # PMR model (NTU-120)
│   ├── DMR.pt                 # DMR baseline (NTU-60)
│   └── DMR_NTU120.pt          # DMR baseline (NTU-120)
│
├── SGN/                        # SGN model for evaluation
│   ├── model.py               # SGN architecture
│   ├── data.py                # Data loading utilities
│   ├── util.py                # Helper functions
│   ├── pretrain.ipynb         # SGN training notebook
│   └── pretrained/            # Pre-trained SGN models
│       ├── action.pt          # Action recognition model
│       └── privacy.pt         # Re-identification model
│
├── NTU/                        # NTU dataset utilities
│   ├── datagen.ipynb          # Data generation notebook
│   └── SGN/                   # SGN-specific data processing
│
├── notebooks/                  # Jupyter notebooks (archived)
│   ├── Eval.ipynb             # Evaluation notebook
│   ├── Motion Retargeting.ipynb # Training notebook
│   ├── Skeleton Visualization.ipynb # Visualization notebook
│   └── view_sample.ipynb      # Sample viewing notebook
│
└── legacy_scripts/             # Old Python scripts (archived)
    ├── eval.py                # Old evaluation script
    ├── motion retargeting.py  # Old training script
    ├── view_sample.py         # Old sample viewer
    ├── vis.py                 # Old visualization utilities
    └── model_ae.py            # Old monolithic model file
```

## Key Components

### Command-Line Interface (`cli.py`)

The CLI provides the following commands:

- `train` - Train PMR or DMR models
- `evaluate` - Evaluate trained models
- `anonymize` - Anonymize skeleton sequences
- `visualize` - Create skeleton visualizations
- `compare` - Create side-by-side comparison videos
- `info` - Display framework information

Example usage:
```bash
python cli.py train --dataset ntu60 --model pmr
python cli.py evaluate --model-path checkpoints/pmr_best.pt
python cli.py anonymize --model-path checkpoints/pmr_best.pt --input data/sample.pkl
```

### Model Architecture (`models/`)

#### PMR Model (`pmr.py`)
- `MotionEncoder`: Extracts action-specific temporal information
- `PrivacyEncoder`: Extracts skeleton structure and style (PII)
- `Decoder`: Reconstructs skeletons from concatenated embeddings
- `PMRModel`: Complete model with cross-reconstruction capability

#### Classifiers (`classifiers.py`)
- `MotionClassifier`: Predicts action labels (cooperative with E_M, adversarial with E_P)
- `PrivacyClassifier`: Predicts actor IDs (cooperative with E_P, adversarial with E_M)
- `QualityController`: GAN-style discriminator for realism
- `DMRModel`: Baseline model without adversarial training

#### SGN Wrapper (`sgn_wrapper.py`)
- Wrapper for SGN model used in evaluation
- Functions for loading action recognition and re-identification models

### Configuration (`configs/default_config.py`)

Dataclass-based configuration system with sections for:
- Data configuration (dataset, sequence length, cameras)
- Model configuration (architecture parameters)
- Training configuration (learning rates, loss weights, epochs)
- Evaluation configuration (metrics, batch size)

### GitHub Pages (`index.html`)

Hypermodern static website featuring:
- Dark mode theme with animated gradients
- Interactive tabs for different sections
- Mermaid diagrams for architecture and data flow
- Results tables and visualizations
- Code examples with copy-to-clipboard
- Smooth animations and scroll effects
- Responsive design

## Training Stages

PMR training consists of 4 stages:

1. **Stage 1: Autoencoder Warm-up** (25 epochs)
   - 5 paired + 20 unpaired epochs
   - Trains encoders and decoder for basic reconstruction

2. **Stage 2: Classifier Pre-training** (70 epochs)
   - 20 paired + 50 unpaired epochs
   - Pre-trains motion and privacy classifiers

3. **Stage 3: Unpaired Cooperative-Adversarial** (100 epochs)
   - Adversarial training to disentangle motion and identity
   - Quality controller ensures realistic outputs

4. **Stage 4: Paired Motion Retargeting** (100 epochs)
   - Cross-reconstruction for anonymization
   - Fine-tunes with triplet and latent consistency losses

Total training time: ~6.5 hours on NVIDIA RTX 3090

## Results (NTU RGB+D 60)

| Method | MSE ↓ | AR Top-1 ↑ | AR Top-5 ↑ | Re-ID Top-1 ↓ | Re-ID Top-5 ↓ | Gender ↓ | Linkage ↓ |
|--------|-------|------------|------------|---------------|---------------|----------|-----------|
| Original | - | 82.2% | 85.0% | 87.8% | 97.3% | 88.7% | 69.6% |
| UNet (Moon) | 0.0834 | 2.6% | 11.1% | 3.0% | 26.8% | 3.0% | 50.0% |
| DMR | 0.0071 | 49.1% | 73.1% | 25.7% | 60.3% | 25.7% | 50.0% |
| **PMR (Ours)** | 0.0138 | 35.7% | 63.0% | 7.8% | 26.4% | 7.8% | 50.0% |

## Migration Notes

### Old Files → New Structure

- `model_ae.py` → `models/pmr.py` + `models/classifiers.py`
- `eval.py` → `training/evaluator.py` (to be implemented)
- `motion retargeting.py` → `training/trainer.py` (to be implemented)
- `vis.py` → `utils/visualization.py` (to be implemented)
- Jupyter notebooks → `notebooks/` (archived)

### Breaking Changes

- Model imports changed from `from model_ae import *` to `from models.pmr import PMRModel`
- Configuration now uses dataclasses instead of dictionaries
- CLI replaces direct script execution

## Next Steps

1. Implement training scripts in `training/`
2. Implement data loading in `data/`
3. Implement utility functions in `utils/`
4. Add helper scripts in `scripts/`
5. Write comprehensive tests
6. Add CI/CD pipeline

## Citation

```bibtex
@inproceedings{carr2025pmr,
  title={Privacy-centric Deep Motion Retargeting for Anonymization of Skeleton-Based Motion Visualization},
  author={Carr, Thomas and Xu, Depeng and Yuan, Shuhan and Lu, Aidong},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  year={2025}
}
```

## Contact

- Thomas Carr - tcarr23@charlotte.edu
- GitHub: https://github.com/Thomasc33/Privacy-Retargeting

