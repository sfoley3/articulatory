#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 13:35:28 2023

@author: seanfoley
"""

'load both the EMA data and MRI tracks from the USC TIMIT corpus (for a single speaker)'

import os
import numpy as np
from scipy.io import loadmat,whosmat

from tqdm import tqdm

#set speaker
spk = 'F1'

#directory of both MRI and EMA data
#assumes structure: 
    #TIMIT --EMA --Data --Speaker --mat
     #     |    
     #     --MRI --Data --Speaker --tracks
parentd = '/data'
emad = os.path.join(parentd, 'EMA/Data')
mrid = os.path.join(parentd,'MRI/Data')
spk_emad = os.path.join(emad, spk)
spk_mrid = os.path.join(mrid, spk)


#output directories for the final arrays
outd = os.path.join(parentd, 'ouput')
if not os.path.exists(outd):
    os.makedirs(outd)
ema_outd = os.path.join(outd, 'EMA')
if not os.path.exists(ema_outd):
    os.makedirs(ema_outd)
mri_outd = os.path.join(outd, 'MRI')
if not os.path.exists(mri_outd):
    os.makedirs(mri_outd)
    

#load EMA data and store as arrays of shape (# of frames, # of channels x 3) (3D datapoints)
#each array is saved in the EMA output directory    
ema_matd = os.path.join(spk_emad,'mat')
out_path = os.path.join(ema_outd, 'output_paths.txt')
with open(out_path, 'w+') as ouf:
    for f in tqdm(os.listdir(ema_matd)):
        if not f.endswith('.mat'):
            continue
        emap = os.path.join(ema_matd, f)
        mat = loadmat(emap)
        mat_data = mat[f[:-4]]
        channels = [l for l in mat_data[0]][1:]
    
        combined_points = []

        for chan in channels:
            points = chan[2]
       #xyz_points.extend(list(points))    
            combined_points.append(points)
            arr = np.concatenate(combined_points,axis=1)
            npp = os.path.join(ema_outd,f[:-4]+'.npy')
            np.save(npp,arr)
            ouf.write('%s\n' % npp)

#load MRI data and store as arrays of shape (# of frames, # of datapoints)
#each array is saved in the MRI output directory
mri_tracksd = os.path.join(spk_mrid, 'tracks')
out_path = os.path.join(mri_outd, 'output_paths.txt')
with open(out_path, 'w+') as ouf:
    for f in tqdm(os.listdir(mri_tracksd)):
        if not f.endswith('.mat'):
            continue
        mrip = os.path.join(mri_tracksd,f)
        var = whosmat(mrip)
        #if 'trackdata'not in var:
         #   print(f"Skipping '{f}' as it does not have 'trackdata'")
        
        #else:
        mri = loadmat(mrip,variable_names=['trackdata'])
        
        tracks = mri['trackdata'][0]
        comb_dat = []

       # Iterate through each cell in trackdata
        for cell_data in tracks:
           # Extract 'contours' struct
           contours_struct = cell_data[0][0]['contours'][0][0]

           # Extract all 'segment' cells
           segment_cells = contours_struct['segment'][0]

           # Initialize a list to store x, y points for this cell
           x_y_points = []

           # Iterate through each 'segment' cell in contours struct
           for segment_cell in segment_cells[:-1]:
               # Extract 'v' element from segment
               v_data = segment_cell['v'][0][0].flatten()

               x_y_points.append(v_data)

           # Concatenate all x, y points for this cell into a single array
           array = np.concatenate(x_y_points, axis=0)

        # Append the concatenated array to the main list
           comb_dat.append(array)
           
        arr = np.vstack(comb_dat)
        npp = os.path.join(mri_outd,f[:-4]+'.npy')
        np.save(npp,arr)
        ouf.write('%s\n' % npp)
    
