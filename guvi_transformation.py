# -*- coding: utf-8 -*-
"""
Created on Sat Dec 27 00:06:06 2025

@author: dogbl
"""

import matplotlib.pyplot as plt
import os 
import glob
import numpy as np
import netCDF4 as nc
import pandas as pd
from tqdm import tqdm
from datetime import datetime, timedelta

wd = r'D:\guvi_aurora'
files = glob.glob(os.path.join(wd, '*.ncdf'))
#each scan is ~24 minutes from start to stop
for file in tqdm(files):
    
    
    ds = nc.Dataset(file, 'r')

    year = ds.variables['Year'][:][0]
    month = ds.variables['Month'][:][0]
    day = ds.variables['Day'][:][0]
    
    orbits = ds.variables['Orbit Number'][:]
    sorted_idx = orbits.argsort()
    #catches fill value
    if ((orbits.mask == True).any()) | (type(orbits) != np.ma.MaskedArray) | (year == 0):
        print('bad sample')
        continue
    




    north_fl   = ds.variables['Magnetic North Flux'][:][sorted_idx[:,None]].squeeze()
    north_ut   = ds.variables['Magnetic North UT second'][:][sorted_idx[:,None]].squeeze() #seconds since day start
    north_mlat = ds.variables['Magnetic North latitude'][:][sorted_idx[:,None]].squeeze()
    north_mlt  = ds.variables['Magnetic North Local Time'][:][sorted_idx[:,None]].squeeze()
    
    
    south_fl   = ds.variables['Magnetic South Flux'][:][sorted_idx[:,None]].squeeze()
    south_ut   = ds.variables['Magnetic South UT second'][:][sorted_idx[:,None]].squeeze() #seconds since day start
    south_mlat = ds.variables['Magnetic South latitude'][:][sorted_idx[:,None]].squeeze()
    south_mlt  = ds.variables['Magnetic South Local Time'][:][sorted_idx[:,None]].squeeze()

    
    north_mask = np.ones_like(north_ut)
    south_mask = np.ones_like(south_ut)
    
    north_mask[(north_ut == 0)] = np.nan
    south_mask[(south_ut == 0)] = np.nan
    

    
    north_mlon = (15.0 * north_mlt) % 360.0
    south_mlon = (15.0 * south_mlt) % 360.0
    
    

    
    
    north_fl *= north_mask
    north_ut *= north_mask
    north_mlat *= north_mask
    north_mlt *= north_mask
    
    south_fl *= south_mask
    south_ut *= south_mask
    south_mlat *= south_mask
    south_mlt *= south_mask
    
    base_date = datetime(year, month, day)

    north_datetime = [base_date + timedelta(seconds = int(np.nanmedian(t))) if ~np.isnan(np.nanmedian(t)) else np.nan for t in north_ut]
    south_datetime = [base_date + timedelta(seconds = int(np.nanmedian(t))) if ~np.isnan(np.nanmedian(t)) else np.nan for t in south_ut]
    
    nm = np.nanmedian(north_ut, axis = -1)
    sm = np.nanmedian(south_ut, axis = -1)
    
    if (nm[0] + nm[1] > 1.4e5) | (sm[0] + sm[1] > 1.4e5): # this check might eliminate <20 days, but its worth it for accurate pairing
        continue
    
    
    #this also assumes that the orbit number is consistent
    if not pd.isna(north_datetime[0]) and not pd.isna(north_datetime[1]):
        if (north_datetime[0] - north_datetime[1]) > timedelta(seconds=6000):
            north_datetime[0] -= timedelta(days=1)  # consistent day of year
    
    if not pd.isna(south_datetime[0]) and not pd.isna(south_datetime[1]):
        if (south_datetime[0] - south_datetime[1]) > timedelta(seconds=6000):
            south_datetime[0] -= timedelta(days=1)  # consistent day of year
        
    
    for i in range(len(north_datetime)):
        if pd.isna(north_datetime[i]): #skip bad times
            continue
        
        title = 'north_'+north_datetime[i+1].strftime('%Y%m%d_%H%M%S')
        
        flux = north_fl[i]
        mlat = north_mlat[i]
        mlt = north_mlt[i]
        
        
        
        breakpoint()
        
    
    # for i in range(north_mlon.shape[0]):   # loop over MLAT index
    
        # theta = np.deg2rad(north_mlon[i])          # angle
        # r     = 90.0 - north_mlat[i]               # radius
        # c     = north_fl[i]
    
        # fig = plt.figure(figsize=(6, 6))
        # ax = fig.add_subplot(111, projection='polar')
    
        # sc = ax.scatter(
        #     theta,
        #     r,
        #     c=c,
        #     s=5,
        #     cmap='inferno',
        #     alpha=0.8
        # )
    
        # ax.set_theta_zero_location('S')   # midnight at bottom (optional)
        # ax.set_theta_direction(1)          # CCW = dawn → noon → dusk
        # ax.set_rlim(0, 40)                 # MLAT ≥ 50°
    
        # ax.set_title(f'North Mean Energy')
        # plt.colorbar(sc, ax=ax, label='Energy flux (ergs/cm2)')
    
        # plt.tight_layout()
        # plt.show()


    # for i in range(south_mlon.shape[0]):   # loop over MLAT index
    
    #     theta = np.deg2rad(south_mlon[i])          # angle
    #     r     = 90.0 + south_mlat[i]               # radius
    #     c     = south_fl[i]
    
    #     fig = plt.figure(figsize=(6, 6))
    #     ax = fig.add_subplot(111, projection='polar')
    
    #     sc = ax.scatter(
    #         theta,
    #         r,
    #         c=c,
    #         s=5,
    #         cmap='inferno',
    #         alpha=0.8
    #     )
    
    #     ax.set_theta_zero_location('S')   # midnight at bottom (optional)
    #     ax.set_theta_direction(1)          # CCW = dawn → noon → dusk
    #     ax.set_rlim(0, 40)                 # MLAT ≥ 50°
    
    #     ax.set_title(f'South Mean Energy')
    #     plt.colorbar(sc, ax=ax, label='Energy flux (ergs/cm2)')
    
    #     plt.tight_layout()
    #     plt.show()
        
        
        
    
    #need a method for cross-calibration