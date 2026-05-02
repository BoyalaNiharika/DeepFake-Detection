#Model with feature visualization
#with real dct, run this code

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import timm
import cv2
import numpy as np

class HybridDeepfakeModel(nn.Module):

    def __init__(
        self,
        num_classes=2,
        seq_len=10
    ):

        super().__init__()

        self.seq_len = seq_len

        # ===================================
        # Vision Transformer
        # ===================================

        self.vit = timm.create_model(

            'vit_small_patch16_224',

            pretrained=True,

            num_classes=0

        )

        self.vit_proj = nn.Linear(

            384,

            512

        )

        # ===================================
        # CNN Backbone
        # ===================================

        resnet = models.resnet50(

            weights=models.ResNet50_Weights.DEFAULT

        )

        self.cnn = nn.Sequential(

            *list(resnet.children())[:-2]

        )

        self.avgpool = nn.AdaptiveAvgPool2d(1)

        # ===================================
        # LSTM
        # ===================================

        self.lstm = nn.LSTM(

            input_size=2048,

            hidden_size=512,

            num_layers=1,

            batch_first=True

        )

        # ===================================
        # FAST FREQUENCY BRANCH (FFT)
        # ===================================

        self.freq_cnn = nn.Sequential(

            nn.Conv2d(1,32,3,padding=1),

            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Conv2d(32,64,3,padding=1),

            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Conv2d(64,128,3,padding=1),

            nn.ReLU(),

            nn.AdaptiveAvgPool2d(1)

        )

        self.freq_fc = nn.Linear(

            128,

            256

        )

        # ===================================
        # Fusion Layer
        # ===================================

        self.fc1 = nn.Linear(

            512 + 512 + 256,

            512

        )

        self.dropout = nn.Dropout(0.4)

        self.fc2 = nn.Linear(

            512,

            num_classes

        )

    # ===================================
    # FAST GPU FFT (Replaces Slow DCT)
    # ===================================

    def compute_frequency(self, x):

        # RGB → grayscale

        x = x.mean(

            dim=1,

            keepdim=True

        )

        # GPU FFT

        fft = torch.fft.fft2(x)

        fft = torch.abs(fft)

        return fft

    # ===================================
    # FORWARD
    # ===================================

    def forward(self, x):

        batch, seq, c, h, w = x.shape

        x_cnn = x.view(

            batch*seq,

            c,

            h,

            w

        )

        # -----------------------------------
        # CNN + LSTM
        # -----------------------------------

        feat = self.cnn(x_cnn)

        feat = self.avgpool(feat)

        feat = feat.view(

            batch,

            seq,

            2048

        )

        lstm_out,_ = self.lstm(feat)

        temporal_feat = torch.mean(

            lstm_out,

            dim=1

        )

        # -----------------------------------
        # ViT
        # -----------------------------------

        vit_in = F.interpolate(

            x_cnn,

            size=(224,224)

        )

        vit_feat = self.vit(vit_in)

        vit_feat = self.vit_proj(vit_feat)

        vit_feat = vit_feat.view(

            batch,

            seq,

            512

        )

        vit_feat = torch.mean(

            vit_feat,

            dim=1

        )

        # -----------------------------------
        # Frequency Branch
        # -----------------------------------

        freq_input = self.compute_frequency(x_cnn)

        freq_feat = self.freq_cnn(freq_input)

        freq_feat = freq_feat.view(

            batch,

            seq,

            128

        )

        freq_feat = torch.mean(

            freq_feat,

            dim=1

        )

        freq_feat = self.freq_fc(freq_feat)

        # -----------------------------------
        # Fusion
        # -----------------------------------

        fusion = torch.cat(

            (

                vit_feat,

                temporal_feat,

                freq_feat

            ),

            dim=1

        )

        out = self.fc1(fusion)

        out = self.dropout(out)

        out = self.fc2(out)

        return out