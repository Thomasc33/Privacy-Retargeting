"""
SGN Model Wrapper for Action Recognition and Re-identification

Wraps the Semantics-Guided Neural Network (SGN) for evaluation purposes.
"""

import torch
import torch.nn as nn
from SGN.model import SGN


class SGNModel:
    """
    Wrapper for SGN model used in evaluation.
    
    Args:
        num_classes: Number of classes (actions or actors)
        dataset: Dataset name ('NTU')
        seg: Temporal segments (default: 20)
        batch_size: Batch size for training
        model_path: Path to pretrained model weights
        device: torch device
    """
    def __init__(self, num_classes, dataset='NTU', seg=20, batch_size=32, 
                 model_path=None, device='cuda'):
        self.num_classes = num_classes
        self.dataset = dataset
        self.seg = seg
        self.batch_size = batch_size
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Initialize SGN model
        self.model = SGN(
            num_classes=num_classes,
            dataset=dataset,
            seg=seg,
            batch_size=batch_size,
            train=False,
            bias=True
        ).to(self.device)
        
        # Load pretrained weights if provided
        if model_path:
            self.load_weights(model_path)
    
    def load_weights(self, model_path):
        """Load pretrained model weights"""
        checkpoint = torch.load(model_path, map_location=self.device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        self.model.eval()
    
    def predict(self, x):
        """
        Make predictions on input data.
        
        Args:
            x: Input skeleton data
        Returns:
            Predictions (logits or probabilities)
        """
        self.model.eval()
        with torch.no_grad():
            if not isinstance(x, torch.Tensor):
                x = torch.tensor(x, dtype=torch.float32)
            x = x.to(self.device)
            outputs = self.model(x)
        return outputs
    
    def evaluate(self, dataloader):
        """
        Evaluate model on a dataloader.
        
        Args:
            dataloader: PyTorch DataLoader
        Returns:
            Dictionary with accuracy metrics
        """
        self.model.eval()
        correct_top1 = 0
        correct_top5 = 0
        total = 0
        
        with torch.no_grad():
            for data, labels in dataloader:
                data = data.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(data)
                
                # Top-1 accuracy
                _, predicted = outputs.max(1)
                correct_top1 += predicted.eq(labels).sum().item()
                
                # Top-5 accuracy
                _, top5_pred = outputs.topk(5, 1, True, True)
                top5_pred = top5_pred.t()
                correct = top5_pred.eq(labels.view(1, -1).expand_as(top5_pred))
                correct_top5 += correct[:5].reshape(-1).float().sum(0).item()
                
                total += labels.size(0)
        
        return {
            'top1_accuracy': correct_top1 / total,
            'top5_accuracy': correct_top5 / total,
            'total_samples': total
        }


def load_action_recognition_model(model_path, num_classes=60, device='cuda'):
    """
    Load pretrained SGN model for action recognition.
    
    Args:
        model_path: Path to model weights
        num_classes: Number of action classes
        device: torch device
    Returns:
        SGNModel instance
    """
    return SGNModel(
        num_classes=num_classes,
        dataset='NTU',
        seg=20,
        batch_size=32,
        model_path=model_path,
        device=device
    )


def load_reidentification_model(model_path, num_classes=40, device='cuda'):
    """
    Load pretrained SGN model for re-identification.
    
    Args:
        model_path: Path to model weights
        num_classes: Number of actors
        device: torch device
    Returns:
        SGNModel instance
    """
    return SGNModel(
        num_classes=num_classes,
        dataset='NTU',
        seg=20,
        batch_size=32,
        model_path=model_path,
        device=device
    )

