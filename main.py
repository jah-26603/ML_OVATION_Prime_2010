# -*- coding: utf-8 -*-
"""
Created on Sat Dec 13 23:23:43 2025

@author: JDawg
"""

from download_training_data import collate_solar_wind, download_solar_wind_data,download_OP_runs
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset, random_split
from dataloader import FC_to_Conv, OP_dataset
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from pytorch_msssim import ssim

# os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
#do this for the first run
if False:                  
    download_solar_wind_data(out_dir = r'E:\ml_aurora\solar_wind') #downloads files
    collate_solar_wind(fp = r'D:\ml_aurora\solar_wind', out = r'D:\ml_aurora\organized_solar_wind.csv') #collates the data into a single .csv file   


df = pd.read_csv(r'D:\ml_aurora\organized_solar_wind.csv')
df['datetime'] = pd.to_datetime(
    dict(
        year=df.year,
        month=df.month,
        day=df.day,
        hour=df.hhmm // 100,
        minute=df.hhmm % 100
    )
)



# #%% gather unique OP measurements -> might be unnecessary with the ace df to ovationpyme pipeline
# op_fp = r'D:\ml_aurora\ovation_prime'
# ndf = pd.DataFrame(os.listdir(op_fp), columns=['filepath'])

# ndf['year']  = ndf['filepath'].str[0:4]
# ndf['month'] = ndf['filepath'].str[4:6]
# ndf['day']   = ndf['filepath'].str[6:8]
# ndf['hhmm']  = ndf['filepath'].str[9:13]   # optional

# ndf['datetime'] = pd.to_datetime(
#     dict(
#         year=ndf.year,
#         month=ndf.month,
#         day=ndf.day,
#         hour=ndf.hhmm.astype(int) // 100,
#         minute=ndf.hhmm.astype(int) % 100
#     ))

# df = pd.merge(df, ndf[['filepath', 'datetime']], on = 'datetime', how = 'left')

# # pair_data(df, out = r'D:\ml_aurora\paired_data', op_fp = r'D:\ml_aurora\ovation_prime')


    
#%%


from dataloader import FC_to_Conv, OP_dataset
from tqdm import tqdm


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = FC_to_Conv().to(device)
    
    # dataset = OP_dataset(spwd_dir =r'D:\ml_aurora\paired_data\swdata', image_dir = r'D:\ml_aurora\paired_data\images')
    dataset = OP_dataset(spwd_dir =r'C:\Users\dogbl\Downloads\ml_aurora\paired_data\swdata', image_dir = r'C:\Users\dogbl\Downloads\ml_aurora\paired_data\images')

    subset_indices = list(range(128 *200)) # first 12800 samples
    dataset = Subset(dataset, subset_indices)
    
    train_size = int(0.75 * len(dataset))
    val_size = len(dataset) - train_size
    
    train_loss_hist, val_loss_hist= [],[]
    
    train_set, val_set = random_split(dataset, [train_size, val_size])
    train_dataloader = DataLoader(train_set,
                                batch_size=32,
                                shuffle=True,
                                num_workers=4,
                                persistent_workers=True,
                                prefetch_factor=2
                            )
    val_dataloader = DataLoader(val_set, 
                                batch_size=32,
                                shuffle=True,
                                num_workers=4,
                                persistent_workers=True,
                                prefetch_factor=2
                            )
    
    
    # Better optimizer settings
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay = 1e-3)
    criterion = nn.HuberLoss()
    num_epochs = 5
    best_val = 100
    
    for epoch in range(num_epochs):
        # ---------- Training ----------
        model.train()
        train_running_loss = 0.0
        val_running_loss = 0
        for i, batch in enumerate(tqdm(train_dataloader), 1):
            x = batch["inputs"].unsqueeze(1).to(device, dtype=torch.float32)
            y = batch["image"].unsqueeze(1).to(device, dtype=torch.float32)
    
            
            optimizer.zero_grad()
            y_hat = model(x)
            y_hat = (y_hat).unsqueeze(1)
            
            loss = criterion(y_hat, y)
            loss.backward()
            train_running_loss += loss.item()
            optimizer.step()
            
        train_loss = train_running_loss / i 
        print(f'Train Loss epoch {epoch +1}: {train_loss:.4f}')
    
        
        model.eval()
        with torch.no_grad():
            for i, batch in enumerate(tqdm(val_dataloader), 1):
                x = batch["inputs"].unsqueeze(1).to(device, dtype=torch.float32)
                y = batch["image"].unsqueeze(1).to(device, dtype=torch.float32)
    
                
                y_hat = model(x)
                y_hat = (y_hat).unsqueeze(1)
                
                loss = criterion(y_hat, y)
                val_running_loss += loss.item()
                
            
            val_loss = val_running_loss / i 
            

            print(f'Val Loss epoch {epoch +1}: {val_loss:.4f}')
            if val_loss < best_val:
                best_val = val_loss
                torch.save(model.state_dict(), f'OP_weight.pth')
                print('saving new weights...')
        train_loss_hist.append(train_loss)
        val_loss_hist.append(val_loss)
            
        
    
    for i in range(8): #unique images
        for c in range(16): 
            # order goes as: North_energy_diff, North_energy_mono, North_energy_wave, North_energy_ions
            #                North_number_diff, North_number_mono, North_number_wave, North_number_ions
            #                South_energy_diff, South_energy_mono, South_energy_wave, South_energy_ions
            #                South_number_diff, South_number_mono, South_number_wave, South_number_ions
    
            plt.figure()
            gt = y[i,0,c].detach().cpu().numpy()
            pr = y_hat[i,0,c].detach().cpu().numpy()
    
    
            vmax = max(np.max(gt), np.max(pr))
            vmin = min(np.min(gt), np.min(pr))
    
            plt.subplot(1,2,1)
            plt.imshow(gt.T, vmin = vmin, vmax = vmax)
            plt.title('Model truth')
            plt.colorbar(orientation = 'horizontal')
    
    
            plt.subplot(1,2,2)
            plt.imshow(pr.T, vmin = vmin, vmax = vmax)
            plt.colorbar(orientation = 'horizontal')
            plt.title('Model prediction')
    
            plt.show()  
            
    
    return train_loss_hist, val_loss_hist
if __name__ == "__main__":
    train_loss, val_loss = main()
    
    
    plt.figure()
    plt.plot(train_loss, label = 'train loss')
    plt.plot(val_loss, label = 'val loss')
    plt.legend()
    plt.show()
