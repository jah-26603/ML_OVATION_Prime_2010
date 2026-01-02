# -*- coding: utf-8 -*-
"""
Created on Sun Dec 14 10:43:33 2025

@author: JDawg
"""

import os
from tqdm import tqdm
import glob
import numpy as np
import requests
import re
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torch
from scipy.ndimage import median_filter

import torch
import torch.nn as nn

class OP_dataset(Dataset):
    '''Ovation prime dataset'''
    def __init__(self, spwd_dir =r'E:\ml_aurora\paired_data\swdata', image_dir = r'E:\ml_aurora\paired_data\images' , transform=None):

        self.spwd_dir = spwd_dir
        self.image_dir = image_dir
        self.transform = transform
        self.titles = os.listdir(spwd_dir)
        
        #values determined across entire dataset
        self.input_mean = np.load(r'input_mean.npy')
        self.input_std = np.load(r'input_std.npy')
        self.image_mean = np.load(r'image_mean.npy')
        self.image_std = np.load(r'image_std.npy')

        #remove entirely 0 datapoints during loading since this case doesn't matter, or I could calculate it at the end with the loss
        # self.valid_titles = []
        # for t in titles:
        #     inputs = np.load(os.path.join(spwd_dir, t))
        #     mask = np.isnan(inputs)
        #     if mask.mean() <= 0.5:
        #         self.valid_titles.append(t)
                
                
    def __len__(self):
        return len(os.listdir(self.spwd_dir))

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        #here idx is the title name
        img = np.load(os.path.join(self.image_dir, self.titles[idx]))
        inputs = np.load(os.path.join(self.spwd_dir, self.titles[idx]))
        
        mask = np.isnan(inputs)
        
        if len(mask[mask == True])/len(inputs) > .5:
            inputs = np.zeros_like(inputs)
            img = np.zeros_like(img)
            
        
        # # img = median_filter(img,size = 3) #smooth
        # for c in range(img.shape[0]):
        #     img[c] = median_filter(img[c], size=3)
        
        inputs[:,:-1] = (inputs[:,:-1] - self.input_mean)/self.input_std
        inputs[:,-1] = inputs[:,-1]/365 #see if there is maybe an issue here...
        img = (img - self.image_mean[:,None,None])/self.image_std[:,None,None]
        sample = {'image': img, 'inputs': inputs}

        inputs[mask] = 0
        return sample

    

class FC_to_Conv(nn.Module):
    def __init__(self):
        super().__init__()
        p = 0.2
        
        # Less aggressive compression: 240×5 → 20×24
        self.fc = nn.Sequential(
            nn.Linear(240 * 5, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p),
            nn.Linear(1024, 128 * 20 * 24),  # Larger spatial resolution
            nn.BatchNorm1d(128 * 20 * 24),
            nn.ReLU(inplace=True),
        )
        
        # Simpler architecture with less downsampling
        self.enc1 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.pool1 = nn.MaxPool2d(2)  # 20×24 → 10×12
        
        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        
        # Decoder
        self.up1 = nn.ConvTranspose2d(64, 64, 4, 2, 1)  # 10×12 → 20×24
        self.dec1 = nn.Sequential(
            nn.Conv2d(64 + 64, 64, 3, padding=1),  # skip connection
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        
        self.up2 = nn.ConvTranspose2d(64, 32, 4, 2, 1)  # 20×24 → 40×48
        self.dec2 = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        
        self.up3 = nn.ConvTranspose2d(32, 16, 4, 2, 1)  # 40×48 → 80×96
        self.final = nn.Conv2d(16, 16, 3, padding=1)
        
    def forward(self, x):
        B = x.size(0)
        x = x.view(B, -1)
        x = self.fc(x)
        x = x.view(B, 128, 20, 24)
        
        # Encoder
        e1 = self.enc1(x)  # 64 @ 20×24
        p1 = self.pool1(e1)  # 64 @ 10×12
        e2 = self.enc2(p1)  # 64 @ 10×12
        
        # Decoder
        d1 = self.up1(e2)  # 64 @ 20×24
        d1 = torch.cat([d1, e1], dim=1)  # 128 @ 20×24
        d1 = self.dec1(d1)  # 64 @ 20×24
        
        d2 = self.up2(d1)  # 32 @ 40×48
        d2 = self.dec2(d2)  # 32 @ 40×48
        
        d3 = self.up3(d2)  # 16 @ 80×96
        d3 = self.final(d3)
        
        return d3