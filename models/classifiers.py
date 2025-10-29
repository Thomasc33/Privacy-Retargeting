"""
Classifier and Discriminator Models for PMR

Contains Motion Classifier, Privacy Classifier, and Quality Controller (GAN discriminator).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


class MotionClassifier(nn.Module):
    """
    Motion Classifier (M) - Predicts action label from embeddings.
    
    Acts cooperatively with motion encoder and adversarially with privacy encoder.
    
    Args:
        num_classes: Number of action classes
        encoded_channels: Tuple of (channels, spatial_dim) for input embedding
    """
    def __init__(self, num_classes, encoded_channels=(256, 32)):
        super(MotionClassifier, self).__init__()
        self.channels = [encoded_channels[0], 128, 256, 512]
        
        self.conv1 = nn.ConvTranspose1d(self.channels[0], self.channels[1], 
                                       3, stride=2, padding=1, output_padding=1)
        self.conv2 = nn.ConvTranspose1d(self.channels[1], self.channels[2], 
                                       3, stride=2, padding=1, output_padding=1)
        self.conv3 = nn.ConvTranspose1d(self.channels[2], self.channels[3], 
                                       3, stride=2, padding=1, output_padding=1)
        
        self.bn1 = nn.BatchNorm1d(self.channels[1])
        self.bn2 = nn.BatchNorm1d(self.channels[2])
        self.bn3 = nn.BatchNorm1d(self.channels[3])
        
        self.pool = nn.AdaptiveAvgPool1d(1)
        
        self.fc1 = nn.Linear(self.channels[3], 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, num_classes)
        
        self.dropout = nn.Dropout(p=0.5)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose1d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = F.dropout(F.relu(self.fc1(x)), p=0.5, training=self.training)
        x = F.dropout(F.relu(self.fc2(x)), p=0.5, training=self.training)
        x = F.softmax(self.fc3(x), dim=1)
        return x


class PrivacyClassifier(nn.Module):
    """
    Privacy Classifier (P) - Predicts actor ID from embeddings.
    
    Acts cooperatively with privacy encoder and adversarially with motion encoder.
    
    Args:
        num_classes: Number of actor IDs
        encoded_channels: Tuple of (channels, spatial_dim) for input embedding
    """
    def __init__(self, num_classes, encoded_channels=(256, 32)):
        super(PrivacyClassifier, self).__init__()
        # Same architecture as MotionClassifier but for different purpose
        self.channels = [encoded_channels[0], 128, 256, 512]
        
        self.conv1 = nn.ConvTranspose1d(self.channels[0], self.channels[1], 
                                       3, stride=2, padding=1, output_padding=1)
        self.conv2 = nn.ConvTranspose1d(self.channels[1], self.channels[2], 
                                       3, stride=2, padding=1, output_padding=1)
        self.conv3 = nn.ConvTranspose1d(self.channels[2], self.channels[3], 
                                       3, stride=2, padding=1, output_padding=1)
        
        self.bn1 = nn.BatchNorm1d(self.channels[1])
        self.bn2 = nn.BatchNorm1d(self.channels[2])
        self.bn3 = nn.BatchNorm1d(self.channels[3])
        
        self.pool = nn.AdaptiveAvgPool1d(1)
        
        self.fc1 = nn.Linear(self.channels[3], 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, num_classes)
        
        self.dropout = nn.Dropout(p=0.5)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose1d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = F.dropout(F.relu(self.fc1(x)), p=0.5, training=self.training)
        x = F.dropout(F.relu(self.fc2(x)), p=0.5, training=self.training)
        x = F.softmax(self.fc3(x), dim=1)
        return x


class QualityController(nn.Module):
    """
    Quality Controller (Q) - GAN-style discriminator for realism.
    
    Distinguishes real vs generated skeletons to improve output quality.
    
    Args:
        T: Number of frames (default: 75)
    """
    def __init__(self, T=75):
        super(QualityController, self).__init__()
        self.T = T
        
        self.enc1 = nn.Conv1d(T, 64, kernel_size=3, stride=1, padding=1)
        self.enc2 = nn.Conv1d(64, 32, kernel_size=3, stride=1, padding=1)
        self.enc3 = nn.Conv1d(32, 16, kernel_size=3, stride=1, padding=1)
        self.enc4 = nn.Conv1d(16, 8, kernel_size=3, stride=1, padding=1)
        
        self.ref1 = nn.ReflectionPad1d(3)
        self.ref2 = nn.ReflectionPad1d(3)
        self.ref3 = nn.ReflectionPad1d(3)
        self.ref4 = nn.ReflectionPad1d(3)
        
        self.fc1 = nn.Linear(80, 32)
        self.fc2 = nn.Linear(32, 1)
        
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.acti = nn.LeakyReLU(0.2)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight)
                init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Args:
            x: Skeleton sequence (real or generated)
        Returns:
            Probability that input is real (1) vs fake (0)
        """
        x = self.ref1(x)
        x = self.acti(self.enc1(x))
        x = self.pool(x)
        
        x = self.ref2(x)
        x = self.acti(self.enc2(x))
        x = self.pool(x)
        
        x = self.ref3(x)
        x = self.acti(self.enc3(x))
        x = self.pool(x)
        
        x = self.ref4(x)
        x = self.acti(self.enc4(x))
        x = self.pool(x)
        
        # Flatten
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        
        return x


class DMRModel(nn.Module):
    """
    Deep Motion Retargeting (DMR) baseline model.
    
    Similar to PMR but without adversarial privacy components.
    """
    def __init__(self, T=75, encoded_channels=(256, 32), use_2d=True):
        super(DMRModel, self).__init__()
        from .pmr import MotionEncoder, PrivacyEncoder, Decoder
        
        self.motion_encoder = MotionEncoder(T, encoded_channels, use_2d)
        self.privacy_encoder = PrivacyEncoder(T, encoded_channels, use_2d)
        self.decoder = Decoder(T, encoded_channels, use_2d)
    
    def forward(self, x):
        """Standard forward pass for reconstruction"""
        motion_emb = self.motion_encoder(x)
        privacy_emb = self.privacy_encoder(x)
        reconstructed = self.decoder(motion_emb, privacy_emb)
        return reconstructed
    
    def cross_reconstruct(self, x_motion, x_privacy):
        """Cross-reconstruction without privacy constraints"""
        motion_emb = self.motion_encoder(x_motion)
        privacy_emb = self.privacy_encoder(x_privacy)
        retargeted = self.decoder(motion_emb, privacy_emb)
        return retargeted

