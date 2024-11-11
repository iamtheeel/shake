#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 12 22:21:21 2024

@author: zsjiang
"""

import h5py, csv
import numpy as np

from genPlots import *

testNumber = 2
trialNumber = 1
#subjectNumbers = ['001']
subjectNumbers = ['001', '002', '003']

dataDir = 'TestData'

for subjectNumber in subjectNumbers:
    # Path th the .csv label file
    csv_path = f'{dataDir}/Test_{testNumber}/APDM_data_fixed_step/MLK Walk_trials_{subjectNumber}_fixedstep.csv'
    with open(csv_path, mode='r') as labelFile:
        labelFile_csv = csv.DictReader(labelFile)
        #labelFile_csv = csv.reader(labelFile)
        labelList = [] 
        for line_number, row in enumerate(labelFile_csv, start=1):
            speed_L = float(row['Gait - Lower Limb - Gait Speed L (m/s) [mean]'])
            speed_R = float(row['Gait - Lower Limb - Gait Speed R (m/s) [mean]'])
            #print(f"Line {line_number}: mean: L={speed_L}, R={speed_R}(m/s) ")
            #thisLabel = (speed_L, speed_R, (speed_L+speed_R)/2)
            labelList.append((speed_R+speed_L)/2)

    # Path to your HDF5 file (data)
    file_path = f'{dataDir}/Test_{testNumber}/data/walking_hallway_single_person_APDM_{subjectNumber}.hdf5'
    # Load the HDF5 file and extract data for the second accelerometer in the first trial
    with h5py.File(file_path, 'r') as file:
        # Get perameters from the data file
        general_parameters = file['experiment/general_parameters'][:]
        general_parameters_list = []
        for param in general_parameters:
            param_info = {
                'ID': param['id'],
                'Parameter': param['parameter'].decode('utf-8'),
                'Units': param['units'].decode('utf-8'),
                'Value': param['value'].decode('utf-8')
            }
            general_parameters_list.append(param_info)
        #sampling_frequency = 1652  # in Hz
        sampling_frequency = int(general_parameters_list[0]['Value']) #Hz
        print(f"srate: {sampling_frequency}Hz")
    
        # Get the data from the datafile
        dataBlockSize =file['experiment/data'].shape 
        print(f"File Size: {dataBlockSize}")
        #nSensors = 19 # There are 20 acceleromiters
        nSensors = dataBlockSize[1]
        nTrials = dataBlockSize[0]
    
        #for trial in range(0,1):
        for trial in range(0,nTrials):
            plotTitle_str = f"Accelerometer Data: Test {testNumber}, subject: {subjectNumber}, trial: {trial+1}, speed: {labelList[trial]}"
            print(f"{plotTitle_str}")
            acclData = [] #zero out the datablock
            print(f"trials: {nTrials}, sensors: {nSensors}")
            for i in range(0,nSensors):
                accelerometer_data = file['experiment/data'][trial, i, :]  # Nth trial, Mth accelerometer
                acclData.append(accelerometer_data)
    
            print(f"accel data shape: {accelerometer_data.shape}")
            maxData = np.max(acclData)
            print(f"Max accl: {maxData}")
            
            # Calculate the time vector
            total_points = accelerometer_data.shape[0]
            time = np.linspace(0, total_points / sampling_frequency, total_points)
    
            # for filename
            runStr = f"test_{testNumber}-subject_{subjectNumber}-trial_{trial+1}"
    
            #plotOverlay(nSensors, time, acclData, runStr, plotTitle_str)
            #plotCombined(time, acclData, runStr, plotTitle_str)
            plotInLine(time, acclData, runStr, plotTitle_str)