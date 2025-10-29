"""
Privacy-centric Motion Retargeting (PMR) Model

This module contains the core PMR architecture including encoders and decoder.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


class MotionEncoder(nn.Module):
    """
    Motion Encoder (E_M) - Extracts action-specific temporal information.
    
    Args:
        T: Number of frames (default: 75)
        encoded_channels: Tuple of (channels, spatial_dim) for latent space
        use_2d: Whether to use 2D convolutions (default: True)
    """
    def __init__(self, T=75, encoded_channels=(256, 32), use_2d=True):
        super(MotionEncoder, self).__init__()
        self.T = T
        self.encoded_channels = encoded_channels
        self.use_2d = use_2d
        
        if use_2d:
            self._build_2d_encoder()
        else:
            self._build_1d_encoder()
    
    def _build_2d_encoder(self):
        """Build 2D convolutional encoder"""
        self.enc1 = nn.Conv2d(self.T, 12, kernel_size=3, stride=1, padding=1)
        self.enc2 = nn.Conv2d(12, 24, kernel_size=3, stride=1, padding=1)
        self.enc3 = nn.Conv2d(24, 32, kernel_size=3, stride=1, padding=1)
        self.enc4 = nn.Conv2d(32, self.encoded_channels[0], kernel_size=3, stride=1, padding=1)
        
        self.ref = nn.ReflectionPad2d(1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.acti = nn.LeakyReLU(0.2)
        
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(self.encoded_channels[0], 
                            self.encoded_channels[0] * self.encoded_channels[1])
        
        self._initialize_weights()
    
    def _build_1d_encoder(self):
        """Build 1D convolutional encoder"""
        self.enc1 = nn.Conv1d(self.T, 128, kernel_size=3, stride=1, padding=1)
        self.enc2 = nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1)
        self.enc3 = nn.Conv1d(256, 512, kernel_size=3, stride=1, padding=1)
        self.enc4 = nn.Conv1d(512, self.encoded_channels[0], kernel_size=3, stride=1, padding=1)
        
        self.ref1 = nn.ReflectionPad1d(3)
        self.ref2 = nn.ReflectionPad1d(3)
        self.ref3 = nn.ReflectionPad1d(3)
        self.ref4 = nn.ReflectionPad1d(3)
        
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.acti = nn.LeakyReLU(0.2)
        
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(self.encoded_channels[0], 
                            self.encoded_channels[0] * self.encoded_channels[1])
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv1d)):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight)
                init.constant_(m.bias, 0)
    
    def forward(self, x):
        if self.use_2d:
            return self._forward_2d(x)
        else:
            return self._forward_1d(x)
    
    def _forward_2d(self, x):
        x = self.ref(x)
        x = self.acti(self.enc1(x))
        x = self.pool(x)
        
        x = self.ref(x)
        x = self.acti(self.enc2(x))
        x = self.pool(x)
        
        x = self.ref(x)
        x = self.acti(self.enc3(x))
        x = self.pool(x)
        
        x = self.ref(x)
        x = self.acti(self.enc4(x))
        x = self.pool(x)
        
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = x.view(-1, *self.encoded_channels)
        
        return x
    
    def _forward_1d(self, x):
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
        
        x = self.global_avg_pool(x)
        x = x.squeeze(-1)
        x = self.fc1(x)
        x = x.view(-1, *self.encoded_channels)
        
        return x


class PrivacyEncoder(nn.Module):
    """
    Privacy Encoder (E_P) - Extracts skeleton structure and style attributes (PII).
    
    Architecture is identical to MotionEncoder but serves a different purpose.
    """
    def __init__(self, T=75, encoded_channels=(256, 32), use_2d=True):
        super(PrivacyEncoder, self).__init__()
        self.encoder = MotionEncoder(T, encoded_channels, use_2d)
    
    def forward(self, x):
        return self.encoder(x)


class Decoder(nn.Module):
    """
    Decoder (D) - Reconstructs skeleton sequence from concatenated embeddings.
    
    Args:
        T: Number of frames (default: 75)
        encoded_channels: Tuple of (channels, spatial_dim) for latent space
        use_2d: Whether to use 2D convolutions (default: True)
    """
    def __init__(self, T=75, encoded_channels=(256, 32), use_2d=True):
        super(Decoder, self).__init__()
        self.T = T
        self.encoded_channels = encoded_channels
        self.use_2d = use_2d
        
        if use_2d:
            self._build_2d_decoder()
        else:
            self._build_1d_decoder()
    
    def _build_2d_decoder(self):
        """Build 2D transpose convolutional decoder"""
        self.dec1 = nn.ConvTranspose2d(self.encoded_channels[0] * 2, 256, 
                                       kernel_size=3, stride=1, padding=1)
        self.dec2 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.dec3 = nn.ConvTranspose2d(128, 96, kernel_size=3, stride=1, padding=1)
        self.dec4 = nn.ConvTranspose2d(96, self.T, kernel_size=3, stride=1, padding=1)
        
        self.ref = nn.ReflectionPad2d(1)
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.up_final = nn.Upsample(size=(25, 3), mode='nearest')
        self.acti = nn.LeakyReLU(0.2)
        
        self._initialize_weights()
    
    def _build_1d_decoder(self):
        """Build 1D transpose convolutional decoder"""
        self.dec1 = nn.ConvTranspose1d(self.encoded_channels[0] * 2, 256, 
                                       kernel_size=3, stride=1, padding=1)
        self.dec2 = nn.ConvTranspose1d(256, 128, kernel_size=3, stride=1, padding=1)
        self.dec3 = nn.ConvTranspose1d(128, 96, kernel_size=3, stride=1, padding=1)
        self.dec4 = nn.ConvTranspose1d(96, self.T, kernel_size=3, stride=1, padding=1)
        
        self.ref1 = nn.ReflectionPad1d(3)
        self.ref2 = nn.ReflectionPad1d(3)
        self.ref3 = nn.ReflectionPad1d(3)
        self.ref4 = nn.ReflectionPad1d(3)
        
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.up75 = nn.Upsample(size=75, mode='nearest')
        self.acti = nn.LeakyReLU(0.2)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.ConvTranspose2d, nn.ConvTranspose1d)):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
    
    def forward(self, motion_emb, privacy_emb):
        """
        Args:
            motion_emb: Motion embedding from E_M
            privacy_emb: Privacy embedding from E_P
        Returns:
            Reconstructed skeleton sequence
        """
        # Concatenate embeddings
        x = torch.cat([motion_emb, privacy_emb], dim=1)
        
        if self.use_2d:
            return self._forward_2d(x)
        else:
            return self._forward_1d(x)
    
    def _forward_2d(self, x):
        x = self.ref(x)
        x = self.acti(self.dec1(x))
        x = self.up(x)
        
        x = self.ref(x)
        x = self.acti(self.dec2(x))
        x = self.up(x)
        
        x = self.ref(x)
        x = self.acti(self.dec3(x))
        x = self.up(x)
        
        x = self.ref(x)
        x = self.acti(self.dec4(x))
        x = self.up_final(x)
        
        return x
    
    def _forward_1d(self, x):
        x = self.ref1(x)
        x = self.acti(self.dec1(x))
        x = self.up(x)
        
        x = self.ref2(x)
        x = self.acti(self.dec2(x))
        x = self.up(x)
        
        x = self.ref3(x)
        x = self.acti(self.dec3(x))
        x = self.up(x)
        
        x = self.ref4(x)
        x = self.acti(self.dec4(x))
        x = self.up75(x)
        
        return x


class PMRModel(nn.Module):
    """
    Complete Privacy-centric Motion Retargeting Model.
    
    Combines motion encoder, privacy encoder, and decoder for anonymization.
    """
    def __init__(self, T=75, encoded_channels=(256, 32), use_2d=True):
        super(PMRModel, self).__init__()
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
        """
        Cross-reconstruction for anonymization.
        
        Args:
            x_motion: Original skeleton (for motion)
            x_privacy: Dummy skeleton (for privacy/structure)
        Returns:
            Anonymized skeleton with original motion but dummy structure
        """
        motion_emb = self.motion_encoder(x_motion)
        privacy_emb = self.privacy_encoder(x_privacy)
        anonymized = self.decoder(motion_emb, privacy_emb)
        return anonymized
    
    def get_embeddings(self, x):
        """Get both embeddings for analysis"""
        motion_emb = self.motion_encoder(x)
        privacy_emb = self.privacy_encoder(x)
        return motion_emb, privacy_emb

