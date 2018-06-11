# -*- coding: utf-8 -*-
# From https://github.com/MorvanZhou/PyTorch-Tutorial/blob/master/tutorial-contents/404_autoencoder.py
import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self, img_size):
        super(Autoencoder, self).__init__()
        
        img_pixels = img_size[0] * img_size[1]
        
        self.encoder = nn.Sequential(
            nn.Linear(img_pixels, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 12),
            nn.Tanh(),
            nn.Linear(12, 3),
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(3, 12),
            nn.Tanh(),
            nn.Linear(12, 64),
            nn.Tanh(),
            nn.Linear(64, 128),
            nn.Tanh(),
            nn.Linear(128, img_pixels),
            nn.Sigmoid() # compress to a range (0, 1)
        )
        
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded