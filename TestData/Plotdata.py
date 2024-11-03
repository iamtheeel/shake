#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 12 22:21:21 2024

@author: zsjiang
"""

import h5py, csv
import matplotlib.pyplot as plt
import numpy as np

testNumber = 2
trialNumber = 1
subjectNumber = '003'

dataDir = 'TestData'
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

#print(labelList)
#exit()

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
            
        # Plotting the data with time on the x-axis
        fig, axs = plt.subplots(4, 4, figsize=(12,12)) #figsize in inches?
        fig.suptitle(plotTitle_str)
        fig.supxlabel(f"Time (s)")
        fig.supylabel(f"Acceleration (g?)")
        #ylim = (-maxData, maxData)
        ylim = (-0.02, 0.02)
        row = 0
        sensor = 0
        axs[row, 0].plot(time, acclData[sensor])
        axs[row, 0].set_title(f'Sensor {sensor+1}')
        axs[row, 0].set_ylim(ylim)
        sensor += 1
        axs[row, 1].plot(time, acclData[sensor])
        axs[row, 1].set_title(f'Sensor {sensor+1}')
        axs[row, 1].set_ylim(ylim)
        sensor += 1
        axs[row, 2].plot(time, acclData[sensor])
        axs[row, 2].set_title(f'Sensor {sensor+1}')
        axs[row, 2].set_ylim(ylim)
        sensor += 1
        axs[row, 3].plot(time, acclData[sensor])
        axs[row, 3].set_title(f'Sensor {sensor+1}')
        axs[row, 3].set_ylim(ylim)
            
        row +=1
        sensor += 1
        axs[row, 0].plot(time, acclData[sensor])
        axs[row, 0].set_title(f'Sensor {sensor+1}')
        axs[row, 0].set_ylim(ylim)
        sensor += 1
        axs[row, 1].plot(time, acclData[sensor])
        axs[row, 1].set_title(f'Sensor {sensor+1}')
        axs[row, 1].set_ylim(ylim)
        sensor += 1
        axs[row, 2].plot(time, acclData[sensor])
        axs[row, 2].set_title(f'Sensor {sensor+1}')
        axs[row, 2].set_ylim(ylim)
        sensor += 1
        # Sensor 8 is x, y, z
        axs[row, 3].plot(time, acclData[sensor])
        axs[row, 3].set_title(f'Sensor {sensor+1}, x,y,z')
        axs[row, 3].set_ylim(ylim)
        sensor += 1
        axs[row, 3].plot(time, acclData[sensor])
        sensor += 1
        axs[row, 3].plot(time, acclData[sensor])
            
        row +=1
        sensor += 1
        # Sensor 9 is x, y, z
        axs[row, 0].plot(time, acclData[sensor])
        axs[row, 0].set_title(f'Sensor {sensor+1}, x,y,z')
        axs[row, 0].set_ylim(ylim)
        sensor += 1
        axs[row, 0].plot(time, acclData[sensor])
        sensor += 1
        axs[row, 0].plot(time, acclData[sensor])
            
        sensor += 1
        axs[row, 1].plot(time, acclData[sensor])
        axs[row, 1].set_title(f'Sensor {sensor+1}')
        axs[row, 1].set_ylim(ylim)
        sensor += 1
        axs[row, 2].plot(time, acclData[sensor])
        axs[row, 2].set_title(f'Sensor {sensor+1}')
        axs[row, 2].set_ylim(ylim)
        sensor += 1
        axs[row, 3].plot(time, acclData[sensor])
        axs[row, 3].set_title(f'Sensor {sensor+1}')
        axs[row, 3].set_ylim(ylim)
            
        row +=1
        sensor += 1
        axs[row, 0].plot(time, acclData[sensor])
        axs[row, 0].set_title(f'Sensor {sensor+1}')
        axs[row, 0].set_ylim(ylim)
        sensor += 1
        axs[row, 1].plot(time, acclData[sensor])
        axs[row, 1].set_title(f'Sensor {sensor+1}')
        axs[row, 1].set_ylim(ylim)
        sensor += 1
        axs[row, 2].plot(time, acclData[sensor])
        axs[row, 2].set_title(f'Sensor {sensor+1}')
        axs[row, 2].set_ylim(ylim)
        sensor += 1
        axs[row, 3].plot(time, acclData[sensor])
        axs[row, 3].set_title(f'Sensor {sensor+1}')
        axs[row, 3].set_ylim(ylim)
            
        runStr = f"test_{testNumber}-subject_{subjectNumber}-trial_{trial+1}"
        print(f"Saving plot {runStr}")
        fig.savefig(f"plots/{runStr}_combined.png")
            
            
        plt.figure(figsize=(10, 4))
            
        for i in range(0,nSensors-1):
            plt.plot(time, acclData[i])
            #plt.plot(time, accelerometer_data)
            plt.title(f"{plotTitle_str}\nSensors 1-20 ")
            #plt.title(f"Accelerometer Data: Trial {trialNum+1}, Sensor {accelNum+1}")
            plt.xlabel("Time (seconds)")
            plt.ylabel("Acceleration")
            plt.ylim(-.02, 0.02)
            #plt.xlim(6.6,7)
            plt.grid(True)
            plt.savefig(f"plots/{runStr}_overlayed.png")

            #plt.show()
