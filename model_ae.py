#!/usr/bin/env python3
"""
model_ae.py

Contains the motion retargeting model (AutoEncoder) and related classes.
Also shows how to load the SGN-based models for action recognition (AR)
and re-identification (RI) if desired.
"""
import os
import pickle
import random
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from SGN.model import SGN
from SGN.data import NTUDataLoaders, AverageMeter
from sklearn.metrics import f1_score, precision_score, recall_score

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# -------------- Hyperparameters --------------
only_use_pos = True  # True uses SGN preprocessing, False uses your preprocessing
remove_two_actor_actions = True
one_dimension_conv = False
ntu_120 = False
only_ntu_120 = False
seperate_train_test = True
sgn_eval_after_each_stage = False
binary_data = False

train_cameras = [2, 3]
test_cameras = [1]
train_actors = [
    1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28,
    31, 34, 35, 38, 45, 46, 47, 49, 50, 52, 53, 54, 55, 56, 57,
    58, 59, 70, 74, 78, 80, 81, 82, 83, 84, 85, 86, 89, 91, 92,
    93, 94, 95, 97, 98, 100, 103
]
T = 75
k = 5
setting = 'cv'
dataset = 'NTU'
metric = 'val_utility_acc_coop'
matric_minimize = False
seg = 20
lr = 1e-5
adv_lr = 1e-5
util_classifier_alpha = 10
priv_classifier_alpha = 0.1
if ntu_120:
    utility_classes = 120
    privacy_classes = 106
else:
    utility_classes = 60
    privacy_classes = 40
validation_acc_freq = -1  # -1 to disable
emb_clf_update_per_epoch_paired = 1
emb_clf_update_per_epoch_unpaired = 3
encoded_channels = (128, 16)  # default
dmr_encoded_channels = (256, 32)  # dmr
batch_size = 32
workers = 0
cross_samples_train = 50000
cross_samples_test = 5000

# Input is size of latent space
class Adversary_Emb(nn.Module):
    def __init__(self, num_classes):
        super(Adversary_Emb, self).__init__()
        self.channels = [encoded_channels[0], 128, 256, 512]
        self.conv1 = nn.ConvTranspose1d(self.channels[0], self.channels[1], 3, stride=2, padding=1, output_padding=1)
        self.conv2 = nn.ConvTranspose1d(self.channels[1], self.channels[2], 3, stride=2, padding=1, output_padding=1)
        self.conv3 = nn.ConvTranspose1d(self.channels[2], self.channels[3], 3, stride=2, padding=1, output_padding=1)
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
        # x = self.fc3(x)
        return x
    
class Discriminator(nn.Module): # 1 = real, 0 = fake
    def __init__(self):
        super(Discriminator, self).__init__()

        self.enc1 = nn.Conv1d(in_channels=T, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.enc2 = nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.enc3 = nn.Conv1d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.enc4 = nn.Conv1d(in_channels=16, out_channels=8, kernel_size=3, stride=1, padding=1)
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

        #flatten
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))

        return x
class Encoder1D(nn.Module):
    def __init__(self):
        super(Encoder1D, self).__init__()

        self.enc1 = nn.Conv1d(in_channels=T, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.enc2 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.enc3 = nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.enc4 = nn.Conv1d(in_channels=512, out_channels=encoded_channels[0], kernel_size=3, stride=1, padding=1)
        self.ref1 = nn.ReflectionPad1d(3)
        self.ref2 = nn.ReflectionPad1d(3)
        self.ref3 = nn.ReflectionPad1d(3)
        self.ref4 = nn.ReflectionPad1d(3)

        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

        self.acti = nn.LeakyReLU(0.2)

        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(encoded_channels[0], encoded_channels[0] * encoded_channels[1])

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight)
                init.constant_(m.bias, 0)

    def forward(self, x):
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
        x = x.view(-1, *encoded_channels)

        return x

class Decoder1D(nn.Module):
    def __init__(self):
        super(Decoder1D, self).__init__()

        self.dec1 = nn.ConvTranspose1d(in_channels=encoded_channels[0]*2, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.dec2 = nn.ConvTranspose1d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.dec3 = nn.ConvTranspose1d(in_channels=128, out_channels=96, kernel_size=3, stride=1, padding=1)
        self.dec4 = nn.ConvTranspose1d(in_channels=96, out_channels=T, kernel_size=3, stride=1, padding=1)

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
            if isinstance(m, nn.ConvTranspose1d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
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
    
class Encoder2D(nn.Module):
    def __init__(self):
        super(Encoder2D, self).__init__()

        self.enc1 = nn.Conv2d(in_channels=T, out_channels=12, kernel_size=(3,3), stride=1, padding=1)
        self.enc2 = nn.Conv2d(in_channels=12, out_channels=24, kernel_size=(3,3), stride=1, padding=1)
        self.enc3 = nn.Conv2d(in_channels=24, out_channels=32, kernel_size=(3,3), stride=1, padding=1)
        self.enc4 = nn.Conv2d(in_channels=32, out_channels=encoded_channels[0], kernel_size=(3,3), stride=1, padding=1)

        self.ref = nn.ReflectionPad2d(1)
        self.pool = nn.MaxPool2d(kernel_size=(2,2), stride=2)
        self.acti = nn.LeakyReLU(0.2)

        self.global_avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc1 = nn.Linear(encoded_channels[0], encoded_channels[0] * encoded_channels[1])

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight)
                init.constant_(m.bias, 0)

    def forward(self, x):
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
        x = x.view(-1, *encoded_channels)

        return x

class Decoder2D(nn.Module):
    def __init__(self):
        super(Decoder2D, self).__init__()

        self.dec1 = nn.ConvTranspose2d(in_channels=encoded_channels[0]*2, out_channels=256, kernel_size=(3,3), stride=1, padding=1)
        self.dec2 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=(3,3), stride=1, padding=1)
        self.dec3 = nn.ConvTranspose2d(in_channels=128, out_channels=96, kernel_size=(3,3), stride=1, padding=1)
        self.dec4 = nn.ConvTranspose2d(in_channels=96, out_channels=75, kernel_size=(3,3), stride=1, padding=1)

        self.ref = nn.ReflectionPad2d(3)
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.up75 = nn.Upsample(size=75, mode='nearest') 
        self.acti = nn.LeakyReLU(0.2)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
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
        x = self.up75(x)
        
        return x

class AutoEncoder(nn.Module):
    def __init__(self, adv_lr=1e-4, use_adv=True):
        super(AutoEncoder, self).__init__()

        # AutoEncoder Models
        if one_dimension_conv:
            self.static_encoder = Encoder1D()
            self.advamic_encoder = Encoder1D()
            self.decoder = Decoder1D()
        else:
            self.static_encoder = Encoder2D()
            self.dynamic_encoder = Encoder2D()
            self.decoder = Decoder1D()

        # Adversarial Models
        self.use_adv = use_adv
        if use_adv:
            self.priv_adv = Adversary_Emb(privacy_classes).to(device) # input = dynamic embedding, output = privacy class
            self.priv_coop = Adversary_Emb(privacy_classes).to(device) # input = static embedding, output = privacy class
            self.util_adv = Adversary_Emb(utility_classes).to(device) # input = static embedding, output = utility class
            self.util_coop = Adversary_Emb(utility_classes).to(device) # input = dynamic embedding, output = utility class
            self.discriminator = Discriminator().to(device)

            self.priv_optim = torch.optim.AdamW(self.priv_adv.parameters(), lr=adv_lr)
            self.priv_coop_optim = torch.optim.AdamW(self.priv_coop.parameters(), lr=adv_lr)
            self.util_optim = torch.optim.AdamW(self.util_adv.parameters(), lr=adv_lr)
            self.util_coop_optim = torch.optim.AdamW(self.util_coop.parameters(), lr=adv_lr)
            self.discriminator_optim = torch.optim.AdamW(self.discriminator.parameters(), lr=adv_lr)

            # Freeze Adversarial Models
            self.priv_adv.eval()
            self.priv_coop.eval()
            self.util_adv.eval()
            self.util_coop.eval()
            self.discriminator.eval()

        # Loss Functions
        self.triplet_loss = nn.TripletMarginLoss()
        self.bce_loss = nn.BCELoss()
        self.cross_entropy = nn.CrossEntropyLoss()
        
        # Info for loss functions
        self.end_effectors = torch.tensor([19, 15, 23, 24, 21, 22, 3]).to(device) * 3
        self.chain_lengths = torch.tensor([5, 5, 8, 8, 8, 8, 5]).to(device)

        # Lambdas for discounted loss
        self.lambda_rec = 2
        self.lambda_cross = 0.1
        self.lambda_ee = 1
        self.lambda_smoothing = 3
        self.lambda_trip = 1
        self.lambda_latent = 10
        self.lambda_adv_util_coop = util_classifier_alpha
        self.lambda_adv_priv_coop = priv_classifier_alpha
        self.lambda_adv_util_adv = util_classifier_alpha
        self.lambda_adv_priv_adv = priv_classifier_alpha
        self.lambda_adv_disc = 1

        # Loss Toggles
        self.use_rec_loss = True
        self.use_cross_loss = True
        self.use_ee_loss = True 
        self.use_trip_loss_paired = True 
        self.use_trip_loss_unpaired = True
        self.use_smoothing_loss = True
        self.use_latent_consistency = True

    def get_loss_params(self):
        return {
            'lambda_rec': self.lambda_rec,
            'lambda_cross': self.lambda_cross,
            'lambda_ee': self.lambda_ee,
            'lambda_trip': self.lambda_trip,
            'lambda_latent': self.lambda_latent,
            'lambda_adv_util_coop': self.lambda_adv_util_coop,
            'lambda_adv_priv_coop': self.lambda_adv_priv_coop,
            'lambda_adv_util_adv': self.lambda_adv_util_adv,
            'lambda_adv_priv_adv': self.lambda_adv_priv_adv,
            'lambda_adv_disc': self.lambda_adv_disc,
            'use_rec_loss': self.use_rec_loss,
            'use_cross_loss': self.use_cross_loss,
            'use_ee_loss': self.use_ee_loss,
            'use_trip_loss_paired': self.use_trip_loss_paired,
            'use_trip_loss_unpaired': self.use_trip_loss_unpaired,
            'use_smoothing_loss': self.use_smoothing_loss,
            'use_latent_consistency': self.use_latent_consistency
        }

    def cross(self, x1, x1_rot, x2, x2_rot):
        d1 = self.dynamic_encoder(x1_rot)
        d2 = self.dynamic_encoder(x2_rot)
        s1 = self.static_encoder(x1)
        s2 = self.static_encoder(x2)
        
        x1_hat = self.decoder(torch.cat((d1, s1), dim=1))
        x2_hat = self.decoder(torch.cat((d2, s2), dim=1))
        y1_hat = self.decoder(torch.cat((d1, s2), dim=1))
        y2_hat = self.decoder(torch.cat((d2, s1), dim=1))

        return x1_hat, x2_hat, y1_hat, y2_hat
    
    def eval(self, x1_rot, x2):
        dynamic = self.dynamic_encoder(x1_rot)
        static = self.static_encoder(x2)
        return self.decoder(torch.cat((dynamic, static), dim=1))

    def rec_loss(self, x, x_rot):
        d = self.dynamic_encoder(x_rot)
        s = self.static_encoder(x)
        x_hat = self.decoder(torch.cat((d, s), dim=1))
        if not one_dimension_conv:
            x_ = x.reshape(x.size(0), T, -1)
        return self.reconstruction_loss(x_, x_hat)
    
    def loss_paired(self, x1, x1_rot, x2, x2_rot, y1, y1_rot, y2, y2_rot, actors, actions, cross = True, reconstruction = True, emb_adv = True, discrim_adv = True, verbose = False):
        d1 = self.dynamic_encoder(x1_rot) # A1
        d2 = self.dynamic_encoder(x2_rot) # A2
        s1 = self.static_encoder(x1) # P1
        s2 = self.static_encoder(x2) # P2

        x1_hat = self.decoder(torch.cat((d1, s1), dim=1)) # P1, A1
        x2_hat = self.decoder(torch.cat((d2, s2), dim=1)) # P2, A2
        y1_hat = self.decoder(torch.cat((d1, s2), dim=1)) # P2, A1
        y2_hat = self.decoder(torch.cat((d2, s1), dim=1)) # P1, A2

        d12 = self.dynamic_encoder(y1_rot) # A1
        d21 = self.dynamic_encoder(y2_rot) # A2
        s12 = self.static_encoder(y1) # P2
        s21 = self.static_encoder(y2) # P1

        x1_hat_ = self.decoder(torch.cat((d12, s21), dim=1)) # P1, A1
        x2_hat_ = self.decoder(torch.cat((d21, s12), dim=1)) # P2, A2
        y1_hat_ = self.decoder(torch.cat((d12, s12), dim=1)) # P2, A1
        y2_hat_ = self.decoder(torch.cat((d21, s21), dim=1)) # P1, A2

        # x1_hat is reconstruction of x1
        # x2_hat is reconstruction of x2
        # y1_hat is cross reconstruction from x1 and x2
        # y2_hat is cross reconstruction from x2 and x1
        # x1_hat_ is cross reconstruction from y1 and y2
        # x2_hat_ is cross reconstruction from y2 and y1
        # y1_hat_ is reconstruction of y1
        # y2_hat_ is reconstruction of y2
        # d1 = A1
        # d2 = A2
        # d12 = A1
        # d21 = A2
        # s1 = P1
        # s2 = P2
        # s12 = P2
        # s21 = P1

        # flatten data if 2D
        if not one_dimension_conv:
            x1 = x1.view(x1.size(0), T, -1)
            x2 = x2.view(x2.size(0), T, -1)
            y1 = y1.view(y1.size(0), T, -1)
            y2 = y2.view(y2.size(0), T, -1)
        
        # initialize all losses to 0 tensor
        rec_loss = torch.zeros(1).to(device)
        cross_loss = torch.zeros(1).to(device)
        end_effector_loss = torch.zeros(1).to(device)
        triplet_loss = torch.zeros(1).to(device)
        smoothing_loss = torch.zeros(1).to(device)
        latent_consistency_loss = torch.zeros(1).to(device)
        privacy_loss = torch.zeros(1).to(device)
        privacy_loss_adv = torch.zeros(1).to(device)
        privacy_loss_coop = torch.zeros(1).to(device)
        utility_loss = torch.zeros(1).to(device)
        utility_loss_adv = torch.zeros(1).to(device)
        utility_loss_coop = torch.zeros(1).to(device)
        privacy_acc_adv = torch.zeros(1).to(device)
        privacy_acc_coop = torch.zeros(1).to(device)
        utility_acc_adv = torch.zeros(1).to(device)
        utility_acc_coop = torch.zeros(1).to(device)
        discriminator_loss = torch.zeros(1).to(device)
        discriminator_acc = torch.zeros(1).to(device)
                        
        # reconstruction loss
        if self.use_rec_loss and reconstruction:
            rec_loss = (self.reconstruction_loss(x1, x1_hat) + self.reconstruction_loss(x2, x2_hat) + self.reconstruction_loss(y1, y1_hat_) + self.reconstruction_loss(y2, y2_hat_)) / 4
            if verbose: print('Reconstruction Loss: ', rec_loss.item())
        
        # cross reconstruction loss
        if self.use_cross_loss and cross:
            # could move this to its own function, but since cross is basically reconstruction, its fine like this
            cross_loss = (self.reconstruction_loss(y1, y1_hat) + self.reconstruction_loss(y2, y2_hat) + self.reconstruction_loss(x1, x1_hat_) + self.reconstruction_loss(x2, x2_hat_)) / 4
            if verbose: print('Cross Reconstruction Loss: ', cross_loss.item())
        
        # end effector loss
        if self.use_ee_loss:
            if reconstruction:
                end_effector_loss += (self.end_effector_loss(x1_hat, x1) + self.end_effector_loss(x2_hat, x2)) / 2
            if cross:
                end_effector_loss += (self.end_effector_loss(y1_hat, y1) + self.end_effector_loss(y2_hat, y2)) / 2
            if verbose: print('End Effector Loss: ', end_effector_loss.item())

        # triplet loss
        if self.use_trip_loss_paired: # anchor, positive, negative
            # d1 = A1, d2 = A2, d12 = A1, d21 = A2
            # s1 = P1, s2 = P2, s12 = P2, s21 = P1
            # d12,s12 = y1, d21,s21 = y2
            # y1 = jk, y2 = il
            triplet_loss = self.triplet_loss(d12, d1, d2) \
                            + self.triplet_loss(d21, d2, d1) \
                            + self.triplet_loss(s12, s2, s1) \
                            + self.triplet_loss(s21, s1, s2) 
            if verbose: print('Triplet Loss: ', triplet_loss.item())

        if self.use_smoothing_loss:
            smoothing_loss = (self.smoothing_loss(x1, x1_hat) + self.smoothing_loss(x2, x2_hat) + self.smoothing_loss(y1, y1_hat_) + self.smoothing_loss(y2, y2_hat_) + \
                                self.smoothing_loss(x1, x1_hat_) + self.smoothing_loss(x2, x2_hat_) + self.smoothing_loss(y1, y1_hat) + self.smoothing_loss(y2, y2_hat)) / 8
            if verbose: print('Smoothing Loss: ', smoothing_loss.item())

        # latent consistency loss
        if self.use_latent_consistency:
            latent_consistency_loss = (self.latent_consistency_loss(d1, d12) + self.latent_consistency_loss(d2, d21) + self.latent_consistency_loss(s1, s21) + self.latent_consistency_loss(s2, s12)) / 4
            if verbose: print('Latent Consistency Loss: ', latent_consistency_loss.item())

        # adversarial loss
        if self.use_adv and emb_adv:
            actor_y1, actor_y2 = actors[0] - 1, actors[1] - 1
            actor_y1, actor_y2 = torch.eye(privacy_classes)[actor_y1.long()].to(device), torch.eye(privacy_classes)[actor_y2.long()].to(device)
            action_y1, action_y2 = actions[0] - 1, actions[1] - 1
            action_y1, action_y2 = torch.eye(utility_classes)[action_y1.long()].to(device), torch.eye(utility_classes)[action_y2.long()].to(device)

            # x1 => d1 s1
            # x2 => d2 s2

            # d1 => p1
            # d2 => p2
            # s1 => a1
            # s2 => a2
            
            # actor_y1 = p1
            # actor_y2 = p2

            # action_y1 = a1
            # action_y2 = a2


            # privacy loss (adversarial)
            privacy_loss_adv = (-self.adv_loss(self.priv_adv, d1, actor_y1) -self.adv_loss(self.priv_adv, d2, actor_y2))/2
            privacy_acc_adv = (self.adv_accuracy(self.priv_adv, d1, actor_y1) + self.adv_accuracy(self.priv_adv, d2, actor_y2))/2

            # privacy loss (coop)
            privacy_loss_coop = (self.adv_loss(self.priv_coop, s1, actor_y1) + self.adv_loss(self.priv_coop, s2, actor_y2))/2
            privacy_acc_coop = (self.adv_accuracy(self.priv_coop, s1, actor_y1) + self.adv_accuracy(self.priv_coop, s2, actor_y2))/2

            # utility loss (adversarial)
            utility_loss_adv = (-self.adv_loss(self.util_adv, s1, action_y1) -self.adv_loss(self.util_adv, s2, action_y2))/2
            utility_acc_adv = (self.adv_accuracy(self.util_adv, s1, action_y1) + self.adv_accuracy(self.util_adv, s2, action_y2))/2

            # utility loss (coop)
            utility_loss_coop = (self.adv_loss(self.util_coop, d1, action_y1) + self.adv_loss(self.util_coop, d2, action_y2))/2
            utility_acc_coop = (self.adv_accuracy(self.util_coop, d1, action_y1) + self.adv_accuracy(self.util_coop, d2, action_y2))/2

            privacy_loss = privacy_loss_adv * self.lambda_adv_priv_adv + privacy_loss_coop * self.lambda_adv_priv_coop
            utility_loss = utility_loss_adv * self.lambda_adv_util_adv + utility_loss_coop * self.lambda_adv_util_coop

            if verbose: 
                print('Privacy Loss (Adversarial): ', privacy_loss_adv.item(), '\tPrivacy Loss (Coop): ', privacy_loss_coop.item())
                print('Utility Loss (Adversarial): ', utility_loss_adv.item(), '\tUtility Loss (Coop): ', utility_loss_coop.item())
                print('Privacy Accuracy (Adversarial): ', privacy_acc_adv.item(), '\tPrivacy Accuracy (Coop): ', privacy_acc_coop.item())
                print('Utility Accuracy (Adversarial): ', utility_acc_adv.item(), '\tUtility Accuracy (Coop): ', utility_acc_coop.item())
            

        if self.use_adv and discrim_adv:
            # discrimnator (adversarial)
            discrim_out_fake = self.discriminator(torch.cat((x1_hat, x2_hat, y1_hat, y2_hat, x1_hat_, x2_hat_, y1_hat_, y2_hat_)))
            discriminator_loss = self.bce_loss(discrim_out_fake, torch.ones_like(discrim_out_fake))
            discriminator_acc = torch.sum(torch.round(discrim_out_fake) == 0).float() / (8 * batch_size)
            if verbose: print('Discriminator Loss: ', discriminator_loss.item(), '\tDiscriminator Accuracy: ', discriminator_acc.item())

        losses = {
            'rec_loss': rec_loss.item(),
            'cross_loss': cross_loss.item(),
            'end_effector_loss': end_effector_loss.item(),
            'triplet_loss': triplet_loss.item(),
            'smoothing_loss': smoothing_loss.item(),
            'latent_consistency_loss': latent_consistency_loss.item(),
            'privacy_loss': privacy_loss.item(),
            'privacy_loss_adv': privacy_loss_adv.item(),
            'privacy_loss_coop': privacy_loss_coop.item(),
            'privacy_acc_adv': privacy_acc_adv.item(),
            'privacy_acc_coop': privacy_acc_coop.item(),
            'utility_loss': utility_loss.item(),
            'utility_loss_adv': utility_loss_adv.item(),
            'utility_loss_coop': utility_loss_coop.item(),
            'utility_acc_adv': utility_acc_adv.item(),
            'utility_acc_coop': utility_acc_coop.item(),
            'discriminator_loss': discriminator_loss.item(),
            'discriminator_acc': discriminator_acc.item()
        }

        return rec_loss * self.lambda_rec \
                + cross_loss * self.lambda_cross \
                + end_effector_loss * self.lambda_ee \
                + triplet_loss * self.lambda_trip \
                + latent_consistency_loss * self.lambda_latent \
                + privacy_loss \
                + utility_loss \
                + discriminator_loss * self.lambda_adv_disc \
                + smoothing_loss * self.lambda_smoothing, \
                x1_hat, x2_hat, y1_hat, y2_hat, losses

    def loss_unpaired(self, x_pos, x_rot, actors, actions, reconstruction = True, emb_adv = False, discrim_adv = False, ee = False, triplet = False, verbose = False):
        d = self.dynamic_encoder(x_rot)
        s = self.static_encoder(x_pos)
        x_hat = self.decoder(torch.cat((d, s), dim=1))

        if not one_dimension_conv:
            x = x_pos.reshape(x_pos.size(0), T, -1)

        # initialize all losses to 0 tensor
        rec_loss = torch.zeros(1).to(device)
        end_effector_loss = torch.zeros(1).to(device)
        triplet_loss = torch.zeros(1).to(device)
        smoothing_loss = torch.zeros(1).to(device)
        privacy_loss = torch.zeros(1).to(device)
        privacy_loss_adv = torch.zeros(1).to(device)
        privacy_loss_coop = torch.zeros(1).to(device)
        utility_loss = torch.zeros(1).to(device)
        utility_loss_adv = torch.zeros(1).to(device)
        utility_loss_coop = torch.zeros(1).to(device)
        privacy_acc_adv = torch.zeros(1).to(device)
        privacy_acc_coop = torch.zeros(1).to(device)
        utility_acc_adv = torch.zeros(1).to(device)
        utility_acc_coop = torch.zeros(1).to(device)
        discriminator_loss = torch.zeros(1).to(device)
        discriminator_acc = torch.zeros(1).to(device)

        # Reconstruction Loss
        if self.use_rec_loss and reconstruction:
            rec_loss = self.reconstruction_loss(x, x_hat)
            if verbose: print('Reconstruction Loss: ', rec_loss.item())

        # End Effector Loss
        if self.use_ee_loss and ee:
            end_effector_loss = self.end_effector_loss(x_hat, x)
            if verbose: print('End Effector Loss: ', end_effector_loss.item())

        # Triplet Loss
        if self.use_trip_loss_unpaired and triplet: # anchor, positive, negative
            triplet_loss = (self.triplet_loss(d, d, s) + self.triplet_loss(s, s, d)) / 2
            if verbose: print('Triplet Loss: ', triplet_loss.item())

        # Smoothing Loss
        if self.use_smoothing_loss:
            smoothing_loss = self.smoothing_loss(x, x_hat)
            if verbose: print('Smoothing Loss: ', smoothing_loss.item())

        # Adversarial Loss
        if self.use_adv and emb_adv:
            actor_y = actors - 1
            actor_y = torch.eye(privacy_classes)[actor_y.long()].to(device)
            action_y = actions - 1
            action_y = torch.eye(utility_classes)[action_y.long()].to(device)

            # latent privacy loss (adv)
            privacy_loss_adv = -self.adv_loss(self.priv_adv, d, actor_y)
            privacy_acc_adv = self.adv_accuracy(self.priv_adv, d, actor_y)

            # latent privacy loss (coop)
            privacy_loss_coop = self.adv_loss(self.priv_coop, s, actor_y)
            privacy_acc_coop = self.adv_accuracy(self.priv_coop, s, actor_y)

            # latent utility loss (adv)
            utility_loss_adv = -self.adv_loss(self.util_adv, s, action_y)
            utility_acc_adv = self.adv_accuracy(self.util_adv, s, action_y)

            # latent utility loss (coop)
            utility_loss_coop = self.adv_loss(self.util_coop, d, action_y)
            utility_acc_coop = self.adv_accuracy(self.util_coop, d, action_y)

            privacy_loss = privacy_loss_adv * self.lambda_adv_priv_adv + privacy_loss_coop * self.lambda_adv_priv_coop
            utility_loss = utility_loss_adv * self.lambda_adv_util_adv + utility_loss_coop * self.lambda_adv_util_coop

            if verbose: 
                print('Privacy Loss Adv: ', privacy_loss_adv.item(), '\tPrivacy Loss Coop: ', privacy_loss_coop.item(), '\tPrivacy Loss: ', privacy_loss.item())
                print('Utility Loss Adv: ', utility_loss_adv.item(), '\tUtility Loss Coop: ', utility_loss_coop.item(), '\tUtility Loss: ', utility_loss.item())
                print('Privacy Accuracy Adv: ', privacy_acc_adv.item(), '\tPrivacy Accuracy Coop: ', privacy_acc_coop.item())
                print('Utility Accuracy Adv: ', utility_acc_adv.item(), '\tUtility Accuracy Coop: ', utility_acc_coop.item())


        if self.use_adv and discrim_adv:
            # discrimnator (adversarial)
            discrim_out_fake = self.discriminator(x_hat)
            discriminator_loss = self.bce_loss(discrim_out_fake, torch.ones_like(discrim_out_fake))
            discriminator_acc = torch.sum(torch.round(discrim_out_fake) == 0).float() / (batch_size)
            if verbose: print('Discriminator Loss: ', discriminator_loss.item(), '\tDiscriminator Accuracy: ', discriminator_acc.item())

        losses = {
            'rec_loss': rec_loss.item(),
            'end_effector_loss': end_effector_loss.item(),
            'triplet_loss': triplet_loss.item(),
            'smoothing_loss': smoothing_loss.item(),
            'privacy_loss': privacy_loss.item(),
            'privacy_loss_adv': privacy_loss_adv.item(),
            'privacy_loss_coop': privacy_loss_coop.item(),
            'privacy_acc_adv': privacy_acc_adv.item(),
            'privacy_acc_coop': privacy_acc_coop.item(),
            'utility_loss': utility_loss.item(),
            'utility_loss_adv': utility_loss_adv.item(),
            'utility_loss_coop': utility_loss_coop.item(),
            'utility_acc_adv': utility_acc_adv.item(),
            'utility_acc_coop': utility_acc_coop.item(),
            'discriminator_loss': discriminator_loss.item(),
            'discriminator_acc': discriminator_acc.item()
        }

        return rec_loss * self.lambda_rec \
                + end_effector_loss * self.lambda_ee \
                + triplet_loss * self.lambda_trip \
                + privacy_loss \
                + utility_loss \
                + discriminator_loss * self.lambda_adv_disc \
                + smoothing_loss * self.lambda_smoothing, \
                x_hat, losses

    def reconstruction_loss(self, x, y):
        # return F.mse_loss(x, y)
        return torch.square(torch.norm(x - y, dim=1)).mean()
    
    def latent_consistency_loss(self, x, y):
        return F.mse_loss(x, y)
    
    def end_effector_loss(self, x, y):
        # slice to get the end effector joints
        x_ee = x[:, :, self.end_effectors.unsqueeze(-1) + torch.arange(3).to(device)] 
        y_ee = y[:, :, self.end_effectors.unsqueeze(-1) + torch.arange(3).to(device)]

        # calculate velocities
        x_vel = torch.norm(x_ee[:, 1:] - x_ee[:, :-1], dim=-1) / self.chain_lengths.unsqueeze(0)
        y_vel = torch.norm(y_ee[:, 1:] - y_ee[:, :-1], dim=-1) / self.chain_lengths.unsqueeze(0)
        
        # compute mse loss for each joint
        losses = F.mse_loss(x_vel, y_vel, reduction='none')

        # take sum over end effectors
        loss = losses.sum(dim=1)

        # take mean over batch
        loss = loss.mean()
        
        return loss
    
    def smoothing_loss(self, y, y_pred):
        # (batch, T, 75)
        # Calculate the squared sum of differences for y and y_pred
        diff_y = torch.sum(y[:, :-1] - y[:, 1:], dim=2) ** 2
        diff_y_pred = torch.sum(y_pred[:, :-1] - y_pred[:, 1:], dim=2) ** 2

        # Calculate the absolute difference
        abs_diff = torch.abs(diff_y - diff_y_pred)

        # Sum over all batches and sequence elements
        loss = torch.sum(abs_diff)

        # Normalize by the total number of elements (batch_size * sequence_length)
        total_loss = torch.sqrt(loss) / (y.size(0) * y.size(1))

        return total_loss

    def adv_loss(self, model, x, y):
        return self.cross_entropy(model(x), y)#.long().to(device))
    
    def adv_accuracy(self, model, x, y):
        return (model(x).argmax(dim=1) == y.argmax(dim=1).to(device)).float().mean()

    def train_adv_paired(self, x1, x1_rot, x2, x2_rot, y1, y1_rot, y2, y2_rot, actors, actions, train_emb = True, train_discrim = True):
        if not self.use_adv: return 0,0
        # freeze encoders/decoder
        self.dynamic_encoder.eval()
        self.static_encoder.eval()
        self.decoder.eval()

        # unfreeze adversaries
        self.priv_adv.train()
        self.util_adv.train()
        self.discriminator.train()

        # zero out gradients
        self.priv_optim.zero_grad()
        self.util_optim.zero_grad()
        self.discriminator_optim.zero_grad()

        # encode
        d1 = self.dynamic_encoder(x1_rot) # A1
        d2 = self.dynamic_encoder(x2_rot) # A2
        d3 = self.dynamic_encoder(y1_rot) # A2
        d4 = self.dynamic_encoder(y2_rot) # A1
        s1 = self.static_encoder(x1) # P1
        s2 = self.static_encoder(x2) # P2
        s3 = self.static_encoder(y1) # P1
        s4 = self.static_encoder(y2) # P2

        # decode
        x1_hat = self.decoder(torch.cat((d1, s1), dim=1)) # P1, A1
        x2_hat = self.decoder(torch.cat((d2, s2), dim=1)) # P2, A2
        y1_hat = self.decoder(torch.cat((d3, s3), dim=1)) # P1, A2
        y2_hat = self.decoder(torch.cat((d4, s4), dim=1)) # P2, A1

        # instantiate losses
        priv_loss = torch.zeros(1).to(device)
        priv_coop_loss = torch.zeros(1).to(device)
        priv_acc = torch.zeros(1).to(device)
        priv_coop_acc = torch.zeros(1).to(device)
        util_loss = torch.zeros(1).to(device)
        util_coop_loss = torch.zeros(1).to(device)
        util_acc = torch.zeros(1).to(device)
        util_coop_acc = torch.zeros(1).to(device)
        discriminator_loss = torch.zeros(1).to(device)
        discriminator_acc = torch.zeros(1).to(device)

        if train_emb:
            # train privacy adversary
            p1, p2 = actors[0] - 1, actors[1] - 1
            p1, p2 = torch.eye(privacy_classes)[p1.long()].to(device), torch.eye(privacy_classes)[p2.long()].to(device)
            priv_loss = (self.cross_entropy(self.priv_adv(d1), p1) + \
                        self.cross_entropy(self.priv_adv(d2), p2) + \
                        self.cross_entropy(self.priv_adv(d3), p1) + \
                        self.cross_entropy(self.priv_adv(d4), p2)) / 4
            priv_acc = (self.adv_accuracy(self.priv_adv, d1, p1) + self.adv_accuracy(self.priv_adv, d2, p2) + self.adv_accuracy(self.priv_adv, d3, p1) + self.adv_accuracy(self.priv_adv, d4, p2)) / 4
            priv_loss.backward(retain_graph=True)
            self.priv_optim.step()

            # train privacy cooperative
            priv_coop_loss = (self.cross_entropy(self.priv_coop(s1), p1) + \
                            self.cross_entropy(self.priv_coop(s2), p2) + \
                            self.cross_entropy(self.priv_coop(s3), p1) + \
                            self.cross_entropy(self.priv_coop(s4), p2)) / 4
            priv_coop_acc = (self.adv_accuracy(self.priv_coop, s1, p1) + self.adv_accuracy(self.priv_coop, s2, p2) + self.adv_accuracy(self.priv_coop, s3, p1) + self.adv_accuracy(self.priv_coop, s4, p2)) / 4
            priv_coop_loss.backward(retain_graph=True)
            self.priv_coop_optim.step()
                        
            # train utility adversary
            a1, a2 = actions[0] - 1, actions[1] - 1
            a1, a2 = torch.eye(utility_classes)[a1.long()].to(device), torch.eye(utility_classes)[a2.long()].to(device)
            util_loss = (self.cross_entropy(self.util_adv(s1), a1) + \
                        self.cross_entropy(self.util_adv(s2), a2) + \
                        self.cross_entropy(self.util_adv(s3), a2) + \
                        self.cross_entropy(self.util_adv(s4), a1)) / 4
            util_acc = (self.adv_accuracy(self.util_adv, s1, a1) + self.adv_accuracy(self.util_adv, s2, a2) + self.adv_accuracy(self.util_adv, s3, a2) + self.adv_accuracy(self.util_adv, s4, a1)) / 4
            util_loss.backward(retain_graph=True)
            self.util_optim.step()

            # train utility cooperative
            util_coop_loss = (self.cross_entropy(self.util_coop(d1), a1) + \
                            self.cross_entropy(self.util_coop(d2), a2) + \
                            self.cross_entropy(self.util_coop(d3), a2) + \
                            self.cross_entropy(self.util_coop(d4), a1)) / 4
            util_coop_acc = (self.adv_accuracy(self.util_coop, d1, a1) + self.adv_accuracy(self.util_coop, d2, a2) + self.adv_accuracy(self.util_coop, d3, a2) + self.adv_accuracy(self.util_coop, d4, a1)) / 4
            util_coop_loss.backward(retain_graph=True)
            self.util_coop_optim.step()


        if train_discrim:
            # train discriminator
            output_real = self.discriminator(torch.cat((x1.view(x1.size(0), T, -1), x2.view(x2.size(0), T, -1), y1.view(y1.size(0), T, -1), y2.view(y1.size(0), T, -1))))
            output_fake = self.discriminator(torch.cat((x1_hat, x2_hat, y1_hat, y2_hat)))
            discriminator_loss = self.bce_loss(output_real, torch.ones_like(output_real)) + self.bce_loss(output_fake, torch.zeros_like(output_fake))
            discriminator_acc = ((torch.sum(torch.round(output_fake) == 0).float() / (4 * batch_size)) + (torch.sum(torch.round(output_real) == 1).float() / (4 * batch_size))) / 2
            discriminator_loss.backward()
            self.discriminator_optim.step()

        # unfreeze encoders/decoder
        self.dynamic_encoder.train()
        self.static_encoder.train()
        self.decoder.train()

        # freeze adversaries
        self.priv_adv.eval()
        self.priv_coop.eval()
        self.util_adv.eval()
        self.util_coop.eval()
        self.discriminator.eval()

        return priv_loss.item(), priv_coop_loss.item(), util_loss.item(), util_coop_loss.item(), discriminator_loss.item(), priv_acc.item(), util_acc.item(), priv_coop_acc.item(), util_coop_acc.item(), discriminator_acc.item()

    def train_adv_unpaired(self, x_pos, x_rot, actor, action, train_emb = True, train_discrim = True):
        # ensure one training method is enabled
        assert train_emb or train_discrim, 'At least one training method must be enabled'

        # freeze encoders/decoder
        self.dynamic_encoder.eval()
        self.static_encoder.eval()
        self.decoder.eval()

        # unfreeze adversaries
        self.priv_adv.train()
        self.priv_coop.train()
        self.util_adv.train()
        self.util_coop.train()
        self.discriminator.train()

        # zero out gradients
        self.priv_optim.zero_grad()
        self.priv_coop_optim.zero_grad()
        self.util_optim.zero_grad()
        self.util_coop_optim.zero_grad()
        self.discriminator_optim.zero_grad()

        # instantiate losses
        priv_loss = torch.zeros(1).to(device)
        priv_coop_loss = torch.zeros(1).to(device)
        priv_acc = torch.zeros(1).to(device)
        priv_coop_acc = torch.zeros(1).to(device)
        util_loss = torch.zeros(1).to(device)
        util_coop_loss = torch.zeros(1).to(device)
        util_acc = torch.zeros(1).to(device)
        util_coop_acc = torch.zeros(1).to(device)
        discriminator_loss = torch.zeros(1).to(device)
        discriminator_acc = torch.zeros(1).to(device)

        if train_emb:
            p = actor - 1
            p = torch.eye(privacy_classes)[p.long()].to(device)
            a = action - 1
            a = torch.eye(utility_classes)[a.long()].to(device)

            # train privacy adversary
            priv_loss = self.adv_loss(self.priv_adv, self.dynamic_encoder(x_rot), p)
            priv_acc = self.adv_accuracy(self.priv_adv, self.dynamic_encoder(x_rot), p)
            priv_loss.backward()
            self.priv_optim.step()

            # tain privacy cooperative
            priv_coop_loss = self.adv_loss(self.priv_coop, self.static_encoder(x_pos), p)
            priv_coop_acc = self.adv_accuracy(self.priv_coop, self.static_encoder(x_pos), p)
            priv_coop_loss.backward()
            self.priv_coop_optim.step()
            
            # train utility adversary
            util_loss = self.adv_loss(self.util_adv, self.static_encoder(x_pos), a)
            util_acc = self.adv_accuracy(self.util_adv, self.static_encoder(x_pos), a)
            util_loss.backward()
            self.util_optim.step()

            # train utility cooperative
            util_coop_loss = self.adv_loss(self.util_coop, self.dynamic_encoder(x_rot), a)
            util_coop_acc = self.adv_accuracy(self.util_coop, self.dynamic_encoder(x_rot), a)
            util_coop_loss.backward()
            self.util_coop_optim.step()

        if train_discrim:
            # encode
            d = self.dynamic_encoder(x_rot)
            s = self.static_encoder(x_pos)

            # decode
            x_hat = self.decoder(torch.cat((d, s), dim=1))

            # train discriminator
            output_real = self.discriminator(x_pos.reshape(x_pos.size(0), T, -1))
            output_fake = self.discriminator(x_hat)
            discriminator_loss = self.bce_loss(output_real, torch.ones_like(output_real)) + self.bce_loss(output_fake, torch.zeros_like(output_fake))
            discriminator_acc = ((torch.sum(torch.round(output_fake) == 0).float() / batch_size) + (torch.sum(torch.round(output_real) == 1).float() / batch_size)) / 2
            discriminator_loss.backward()
            self.discriminator_optim.step()

        # unfreeze encoders/decoder
        self.dynamic_encoder.train()
        self.static_encoder.train()
        self.decoder.train()

        # freeze adversaries
        self.priv_adv.eval()
        self.priv_coop.eval()
        self.util_adv.eval()
        self.util_coop.eval()
        self.discriminator.eval()

        return priv_loss.item(), priv_coop_loss.item(), util_loss.item(), util_coop_loss.item(), discriminator_loss.item(), priv_acc.item(), util_acc.item(), priv_coop_acc.item(), util_coop_acc.item(), discriminator_acc.item()

    def val_adv_paired(self, x1, x1_rot, x2, x2_rot, y1, y1_rot, y2, y2_rot, actors, actions, train_emb = True, train_discrim = True):
        if not self.use_adv: return 0,0

        # freeze encoders/decoder
        self.set_eval()

        # Encode
        d1, d2, d3, d4 = [self.dynamic_encoder(x) for x in [x1_rot, x2_rot, y1_rot, y2_rot]]
        s1, s2, s3, s4 = [self.static_encoder(x) for x in [x1, x2, y1, y2]]

        # Decode
        x1_hat, x2_hat, y1_hat, y2_hat = [self.decoder(torch.cat((d, s), dim=1)) for d, s in zip([d1, d2, d3, d4], [s1, s2, s3, s4])]

        # instantiate losses
        priv_loss = torch.zeros(1).to(device)
        priv_coop_loss = torch.zeros(1).to(device)
        priv_acc = torch.zeros(1).to(device)
        priv_coop_acc = torch.zeros(1).to(device)
        util_loss = torch.zeros(1).to(device)
        util_coop_loss = torch.zeros(1).to(device)
        util_acc = torch.zeros(1).to(device)
        util_coop_acc = torch.zeros(1).to(device)
        discriminator_loss = torch.zeros(1).to(device)
        discriminator_acc = torch.zeros(1).to(device)

        if train_emb:
            # privacy adversary
            p1, p2 = actors[0] - 1, actors[1] - 1
            p1, p2 = torch.eye(privacy_classes)[p1.long()].to(device), torch.eye(privacy_classes)[p2.long()].to(device)
            priv_loss = (self.cross_entropy(self.priv_adv(d1), p1) + \
                        self.cross_entropy(self.priv_adv(d2), p2) + \
                        self.cross_entropy(self.priv_adv(d3), p1) + \
                        self.cross_entropy(self.priv_adv(d4), p2)) / 4
            priv_acc = (self.adv_accuracy(self.priv_adv, d1, p1) + self.adv_accuracy(self.priv_adv, d2, p2) + self.adv_accuracy(self.priv_adv, d3, p1) + self.adv_accuracy(self.priv_adv, d4, p2)) / 4

            # privacy cooperative
            priv_coop_loss = (self.cross_entropy(self.priv_coop(s1), p1) + \
                            self.cross_entropy(self.priv_coop(s2), p2) + \
                            self.cross_entropy(self.priv_coop(s3), p1) + \
                            self.cross_entropy(self.priv_coop(s4), p2)) / 4
            priv_coop_acc = (self.adv_accuracy(self.priv_coop, s1, p1) + self.adv_accuracy(self.priv_coop, s2, p2) + self.adv_accuracy(self.priv_coop, s3, p1) + self.adv_accuracy(self.priv_coop, s4, p2)) / 4
                        
            # utility adversary
            a1, a2 = actions[0] - 1, actions[1] - 1
            a1, a2 = torch.eye(utility_classes)[a1.long()].to(device), torch.eye(utility_classes)[a2.long()].to(device)
            util_loss = (self.cross_entropy(self.util_adv(s1), a1) + \
                        self.cross_entropy(self.util_adv(s2), a2) + \
                        self.cross_entropy(self.util_adv(s3), a2) + \
                        self.cross_entropy(self.util_adv(s4), a1)) / 4
            util_acc = (self.adv_accuracy(self.util_adv, s1, a1) + self.adv_accuracy(self.util_adv, s2, a2) + self.adv_accuracy(self.util_adv, s3, a2) + self.adv_accuracy(self.util_adv, s4, a1)) / 4

            # utility cooperative
            util_coop_loss = (self.cross_entropy(self.util_coop(d1), a1) + \
                            self.cross_entropy(self.util_coop(d2), a2) + \
                            self.cross_entropy(self.util_coop(d3), a2) + \
                            self.cross_entropy(self.util_coop(d4), a1)) / 4
            util_coop_acc = (self.adv_accuracy(self.util_coop, d1, a1) + self.adv_accuracy(self.util_coop, d2, a2) + self.adv_accuracy(self.util_coop, d3, a2) + self.adv_accuracy(self.util_coop, d4, a1)) / 4


        if train_discrim:
            # discriminator
            output_real = self.discriminator(torch.cat((x1.view(x1.size(0), T, -1), x2.view(x2.size(0), T, -1), y1.view(y1.size(0), T, -1), y2.view(y1.size(0), T, -1))))
            output_fake = self.discriminator(torch.cat((x1_hat, x2_hat, y1_hat, y2_hat)))
            discriminator_loss = self.bce_loss(output_real, torch.ones_like(output_real)) + self.bce_loss(output_fake, torch.zeros_like(output_fake))
            discriminator_acc = ((torch.sum(torch.round(output_fake) == 0).float() / (4 * batch_size)) + (torch.sum(torch.round(output_real) == 1).float() / (4 * batch_size))) / 2

        # unfreeze encoders/decoder
        self.set_eval(False)

        return priv_loss.item(), priv_coop_loss.item(), util_loss.item(), util_coop_loss.item(), discriminator_loss.item(), priv_acc.item(), util_acc.item(), priv_coop_acc.item(), util_coop_acc.item(), discriminator_acc.item()

    def val_adv_unpaired(self, x_pos, x_rot, actor, action, train_emb = True, train_discrim = True):
        # ensure one training method is enabled
        assert train_emb or train_discrim, 'At least one training method must be enabled'

        # freeze encoders/decoder
        self.set_eval()

        # Encode
        d = self.dynamic_encoder(x_rot)
        s = self.static_encoder(x_pos)

        # Decode
        x_hat = self.decoder(torch.cat((d, s), dim=1))

        # instantiate losses
        priv_loss = torch.zeros(1).to(device)
        priv_coop_loss = torch.zeros(1).to(device)
        priv_acc = torch.zeros(1).to(device)
        priv_coop_acc = torch.zeros(1).to(device)
        util_loss = torch.zeros(1).to(device)
        util_coop_loss = torch.zeros(1).to(device)
        util_acc = torch.zeros(1).to(device)
        util_coop_acc = torch.zeros(1).to(device)
        discriminator_loss = torch.zeros(1).to(device)
        discriminator_acc = torch.zeros(1).to(device)

        if train_emb:
            p = actor - 1
            p = torch.eye(privacy_classes)[p.long()].to(device)
            a = action - 1
            a = torch.eye(utility_classes)[a.long()].to(device)

            # privacy adversary
            priv_loss = self.adv_loss(self.priv_adv, self.dynamic_encoder(x_rot), p)
            priv_acc = self.adv_accuracy(self.priv_adv, self.dynamic_encoder(x_rot), p)

            # privacy cooperative
            priv_coop_loss = self.adv_loss(self.priv_coop, self.static_encoder(x_pos), p)
            priv_coop_acc = self.adv_accuracy(self.priv_coop, self.static_encoder(x_pos), p)
            
            # utility adversary
            util_loss = self.adv_loss(self.util_adv, self.static_encoder(x_pos), a)
            util_acc = self.adv_accuracy(self.util_adv, self.static_encoder(x_pos), a)

            # utility cooperative
            util_coop_loss = self.adv_loss(self.util_coop, self.dynamic_encoder(x_rot), a)
            util_coop_acc = self.adv_accuracy(self.util_coop, self.dynamic_encoder(x_rot), a)

        if train_discrim:
            # encode
            d = self.dynamic_encoder(x_rot)
            s = self.static_encoder(x_pos)

            # decode
            x_hat = self.decoder(torch.cat((d, s), dim=1))

            # train discriminator
            output_real = self.discriminator(x_pos.reshape(x_pos.size(0), T, -1))
            output_fake = self.discriminator(x_hat)
            discriminator_loss = self.bce_loss(output_real, torch.ones_like(output_real)) + self.bce_loss(output_fake, torch.zeros_like(output_fake))
            discriminator_acc = ((torch.sum(torch.round(output_fake) == 0).float() / batch_size) + (torch.sum(torch.round(output_real) == 1).float() / batch_size)) / 2

        # unfreeze encoders/decoder
        self.set_eval(False)

        return priv_loss.item(), priv_coop_loss.item(), util_loss.item(), util_coop_loss.item(), discriminator_loss.item(), priv_acc.item(), util_acc.item(), priv_coop_acc.item(), util_coop_acc.item(), discriminator_acc.item()

    def forward(self, x, x_rot):
        dyn = self.dynamic_encoder(x_rot)
        sta = self.static_encoder(x)
        x = self.decoder(torch.cat((dyn, sta), dim=1))
        return x
    
    def set_eval(self, eval=True):
        if eval:
            self.static_encoder.eval()
            self.dynamic_encoder.eval()
            self.decoder.eval()
            self.priv_adv.eval()
            self.priv_coop.eval()
            self.util_adv.eval()
            self.util_coop.eval()
            self.discriminator.eval()
        else:
            self.static_encoder.train()
            self.dynamic_encoder.train()
            self.decoder.train()


def load_sgn_models():
    """
    Example function if you want to load your SGN-based AR/RI models.
    This assumes you have SGN code in `SGN/model.py`.
    """
    from SGN.model import SGN
    # from SGN.data import ...
    sgn_ar = SGN(utility_classes, None, seg, batch_size, 0).to(device)
    sgn_priv = SGN(privacy_classes, None, seg, batch_size, 0).to(device)

    # Load from your pretrained weights
    # e.g.
    if only_use_pos:
        # Example logic
        sgn_ar.load_state_dict(torch.load('SGN/pretrained/action_60_sgnpt.pt')['state_dict'])
        sgn_priv.load_state_dict(torch.load('SGN/pretrained/privacy_60_sgnpt.pt')['state_dict'])
    else:
        sgn_ar.load_state_dict(torch.load('SGN/pretrained/action_60.pt')['state_dict'])
        sgn_priv.load_state_dict(torch.load('SGN/pretrained/privacy_60.pt')['state_dict'])

    sgn_ar.eval()
    sgn_priv.eval()
    return sgn_ar, sgn_priv