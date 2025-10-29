"""
Default Configuration for PMR Training

This file contains all hyperparameters and settings for training the PMR model.
"""

from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class DataConfig:
    """Data-related configuration"""
    # Dataset settings
    dataset_name: str = 'NTU-60'  # 'NTU-60' or 'NTU-120'
    data_path: str = 'ntu/SGN/X_full.pkl'
    only_use_pos: bool = True  # Use SGN preprocessing
    remove_two_actor_actions: bool = True
    
    # Train/test split
    train_cameras: List[int] = field(default_factory=lambda: [2, 3])
    test_cameras: List[int] = field(default_factory=lambda: [1])
    train_actors: List[int] = field(default_factory=lambda: [
        1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28,
        31, 34, 35, 38, 45, 46, 47, 49, 50, 52, 53, 54, 55, 56, 57,
        58, 59, 70, 74, 78, 80, 81, 82, 83, 84, 85, 86, 89, 91, 92,
        93, 94, 95, 97, 98, 100, 103
    ])
    
    # Data processing
    T: int = 75  # Number of frames
    num_joints: int = 25
    num_coords: int = 3  # x, y, z
    
    # Cross-reconstruction samples
    cross_samples_train: int = 50000
    cross_samples_test: int = 5000


@dataclass
class ModelConfig:
    """Model architecture configuration"""
    # Encoder/Decoder settings
    encoded_channels: Tuple[int, int] = (256, 32)  # (channels, spatial_dim)
    use_2d_conv: bool = True  # Use 2D convolutions (vs 1D)
    
    # Number of classes
    utility_classes: int = 60  # Action classes (60 for NTU-60, 120 for NTU-120)
    privacy_classes: int = 40  # Actor IDs (40 for NTU-60, 106 for NTU-120)
    
    # SGN settings
    seg: int = 20  # Temporal segments for SGN


@dataclass
class TrainingConfig:
    """Training hyperparameters"""
    # Optimization
    lr: float = 1e-5  # Learning rate for autoencoder
    adv_lr: float = 1e-5  # Learning rate for adversarial components
    batch_size: int = 32
    num_workers: int = 0
    
    # Training stages (epochs)
    stage1_ae_warmup_paired: int = 5
    stage1_ae_warmup_unpaired: int = 20
    stage2_classifier_pretrain_paired: int = 20
    stage2_classifier_pretrain_unpaired: int = 50
    stage3_unpaired_training: int = 100
    stage4_paired_training: int = 100
    
    # Loss weights (alphas)
    alpha_rec: float = 2.0  # Reconstruction loss
    alpha_smooth: float = 3.0  # Smoothness loss
    alpha_qc: float = 0.5  # Quality controller loss
    alpha_cross: float = 0.1  # Cross-reconstruction loss
    alpha_trip: float = 1.0  # Triplet loss
    alpha_latent: float = 10.0  # Latent consistency loss
    alpha_ee: float = 1.0  # End-effector loss
    alpha_emb: float = 0.5  # Embedding classifier loss (cooperative + adversarial)
    
    # Classifier update frequency
    emb_clf_update_per_epoch_paired: int = 1
    emb_clf_update_per_epoch_unpaired: int = 3
    
    # Validation
    validation_acc_freq: int = -1  # -1 to disable, >0 for frequency
    
    # Device
    device: str = 'cuda:0'
    
    # Checkpointing
    checkpoint_dir: str = 'checkpoints'
    save_freq: int = 10  # Save every N epochs
    
    # Logging
    log_dir: str = 'logs'
    use_mlflow: bool = False
    experiment_name: str = 'Privacy Retargeting'


@dataclass
class EvaluationConfig:
    """Evaluation settings"""
    # Pretrained model paths
    sgn_action_model_path: str = 'SGN/pretrained/action_recognition.pt'
    sgn_reid_model_path: str = 'SGN/pretrained/reidentification.pt'
    sgn_gender_model_path: str = 'SGN/pretrained/gender_classification.pt'
    
    # Evaluation metrics
    compute_mse: bool = True
    compute_action_recognition: bool = True
    compute_reidentification: bool = True
    compute_gender_classification: bool = True
    compute_linkage_attack: bool = True
    
    # Dummy skeleton settings
    use_constant_dummy: bool = True  # vs random dummy
    constant_dummy_id: int = 21  # Actor ID to use as constant dummy


@dataclass
class Config:
    """Complete configuration"""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    
    def __post_init__(self):
        """Update dependent parameters"""
        if self.data.dataset_name == 'NTU-120':
            self.model.utility_classes = 120
            self.model.privacy_classes = 106


def get_default_config():
    """Get default configuration"""
    return Config()


def get_ntu120_config():
    """Get configuration for NTU-120 dataset"""
    config = Config()
    config.data.dataset_name = 'NTU-120'
    config.data.data_path = 'ntu/SGN/X_full_120.pkl'
    config.model.utility_classes = 120
    config.model.privacy_classes = 106
    return config


def get_dmr_config():
    """Get configuration for DMR baseline (no privacy components)"""
    config = Config()
    config.training.alpha_emb = 0.0  # Disable adversarial training
    return config

