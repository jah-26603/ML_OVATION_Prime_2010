# -*- coding: utf-8 -*-
"""
Created on Sat Dec 13 19:07:56 2025

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
from datetime import timedelta

def download_solar_wind_data(out_dir = r'E:\solar_wind'):
    
    #downloads 1m resolution solar wind data 
    url = r'https://sohoftp.nascom.nasa.gov/sdb/goes/ace/daily/'
    page = requests.get(url).text
    
    
    file_types = ['swepam', 'mag']
    pattern = re.compile(r'href="([^"]*(?:swepam|mag).*?\.txt)"')
    files = [url + m for m in pattern.findall(page)]
    
    os.makedirs(out_dir, exist_ok=True)
    session = requests.Session()

    for f in tqdm(files, desc = 'Downloading solar wind data...'):
        fname = os.path.join(out_dir, os.path.basename(f))
    
        if os.path.exists(fname):   # skip existing
            continue
    
        with session.get(f, stream=True) as r:
            r.raise_for_status()
            with open(fname, 'wb') as fp:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        fp.write(chunk)
    print(f'Solar Wind Data: Complete Download from: {files[0][-24:-15]} to {files[-4][-24:-15]}')
      


#transform solar wind data
def collate_solar_wind(fp = r'E:\solar_wind', out = r'E:\organized_solar_wind.csv'):

    mag_data = []
    vel_data = []
    
    for f in tqdm(glob.glob(os.path.join(fp, '*.txt')), desc = 'Loading in Solar Wind data...'):
        
    
        if 'mag' in f:
            with open(f, 'r') as fh:
                for i,line in enumerate(fh):
                    if i < 20:
                        continue
                    mag_data.append(line[:-2]) #ignore new line character
        else:
            
            with open(f, 'r') as fh:
                for i,line in enumerate(fh):
                    if i < 18:
                        continue
                    vel_data.append(line[:-2]) #ignore new line character
                    
    
    #mag file structure          
    ##                 Modified Seconds
    # UT Date   Time  Julian   of the   ----------------  GSM Coordinates ---------------
    # YR MO DA  HHMM    Day      Day    S     Bx      By      Bz      Bt     Lat.   Long.              
    
    #vel file structure
    ##                Modified Seconds   -------------  Solar Wind  -----------
    # UT Date   Time  Julian  of the          Proton      Bulk         Ion
    # YR MO DA  HHMM    Day     Day     S    Density     Speed     Temperature
    #-------------------------------------------------------------------------
    
    cols = ['year', 'month', 'day', 'hhmm','mjd', 'sec', 'mag_dqi', 'Bx', 'By', 'Bz','Bt', 'lat', 'lon']
    df = pd.DataFrame(mag_data)[0].str.split(expand=True).set_axis(cols, axis = 1)
    df = df[['year','month','day','hhmm','mag_dqi','Bx', 'By', 'Bz','Bt']]
    
    cols = ['year', 'month', 'day', 'hhmm','mjd', 'sec', 'vel_dqi', 'prot_dens', 'vel', 'T_ion']
    vel_df = pd.DataFrame(vel_data)[0].str.split(expand=True).set_axis(cols, axis = 1)
    vel_df = vel_df[['year','month','day','hhmm','vel_dqi','vel']]
    
    
    df = pd.merge(df, vel_df, on = ['year', 'month', 'day', 'hhmm'])
    df = df.astype(float)
    mask = ((df.vel_dqi == 9) 
              |(df.mag_dqi == 9) 
              |(df.vel_dqi == 9)
              |(df.vel <= -9999)
              |(df.Bz <= -999))

    cols = ['Bx', 'By', 'Bz', 'Bt', 'vel']
    df.loc[mask, cols] = np.nan
    df['clock_ang'] = np.arctan2(df.By, df.Bz) #radians
    
    df.to_csv(out)
    
    

def download_OP_runs(out_dir = r'E:\ml_aurora\ovation_prime' ):

    base_url = r"https://iswa.gsfc.nasa.gov/iswa_data_tree/model/ionosphere/ovation_prime/e+i_data/"

    current_files = os.listdir(out_dir)
    for year in tqdm(np.arange(2012, 2026), desc = 'Downloading OP data...'):
        for month in tqdm(np.arange(1, 13)):
            # build folder path
            folder = f"{year}/{str(month).zfill(2)}/"
            url = base_url + folder
    
            try:
                response = requests.get(url)
                response.raise_for_status()
                text = response.text
            except requests.HTTPError:
                continue
    
            # extract all txt files using regex
            files = re.findall(r'href="([^"]*(?:00|15|30|45)UT[^"]*\.txt)"',text)
            if not files:
                continue
    
            os.makedirs(out_dir, exist_ok=True)
    
            for fname in files:
                if fname in current_files:
                    print('Already downloaded: ', fname)
                    continue
                file_url = url + fname
                local_path = os.path.join(out_dir, fname)
                with requests.get(file_url, stream=True) as r:
                    r.raise_for_status()
                    with open(local_path, "wb") as f:
                        for chunk in r.iter_content(8192):
                            f.write(chunk)
                            
                            
                            
from datetime import timedelta
                
def pair_data(df, out = r'E:\paired_data', op_fp = r'E:\ovation_prime'):


    os.makedirs(os.path.join(out, 'images'), exist_ok=True)
    os.makedirs(os.path.join(out, 'swdata'), exist_ok=True)

    for f in tqdm(os.listdir(op_fp)):
        dd = []
        #if statement to skip already paired data, super easy
        with open(os.path.join(op_fp, f), 'r') as fh:
            for i, line in enumerate(fh):
                if i < 1 or i > 7680:
                    continue
                dd.append(line[-5:-1])

        timestamp = df.loc[df.filepath == f, 'datetime'].iloc[0]
        ts = timestamp.strftime('%Y%m%d_%H%M%S')


        #each continuous runtime is based on the omni data which is ~60 minutes BEHIND ace data
        # since I'm using OP ~= Ace + 1 hour -> ace_dt ~= OP - 1
        time_series = df[
            (df['datetime'] >= timestamp - timedelta(hours=5)) &
            (df['datetime']  < timestamp - timedelta(hours =1))
        ]
        
        #a preprocessing step is just to throw out measurements w/o 3 hours
        if len(time_series)< 180:
            continue
        
        input_data = time_series[['Bt', 'vel', 'clock_ang']].to_numpy()
        img = np.array(dd, dtype=float).reshape(96, 80)
        
        np.save(os.path.join(out, 'images', f'{ts}.npy'), img)
        np.save(os.path.join(out, 'swdata', f'{ts}.npy'), input_data)
