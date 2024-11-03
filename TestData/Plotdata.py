#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 12 22:21:21 2024

@author: zsjiang
"""

import h5py
import matplotlib.pyplot as plt
import numpy as np

trialNum = 0
accelNum = 2
nAccels = 19 # There are 20 acceleromiters
# Path to your HDF5 file
file_path = 'TestData/Test_2/data/walking_hallway_single_person_APDM_003.hdf5'
#file_path = '/Users/zsjiang/Desktop/TestData/Test_2/data/walking_hallway_single_person_APDM_001.hdf5'

# Load the HDF5 file and extract data for the second accelerometer in the first trial
acclData = []
with h5py.File(file_path, 'r') as file:
    for i in range(0,nAccels):
        accelerometer_data = file['experiment/data'][trialNum, i, :]  # Nth trial, Mth accelerometer
        acclData.append(accelerometer_data)
    #accelerometer_data = file['experiment/data'][trialNum, 1, :]  # Nth trial, Mth accelerometer
    #acclData.append(accelerometer_data)
    #acclData[0] = file['experiment/data'][trialNum, 0, :]  # Nth trial, 20 accelerometers
    #accelerometer_data = file['experiment/data'][trialNum, accelNum, :]  # Nth trial, Mth accelerometer
    #accelerometer_data = file['experiment/data'][0, 3, :]  # First trial, second accelerometer

print(f"accel data shape: {accelerometer_data.shape}")
# Calculate the time vector
sampling_frequency = 1652  # in Hz
total_points = accelerometer_data.shape[0]
time = np.linspace(0, total_points / sampling_frequency, total_points)

# Plotting the data with time on the x-axis
plt.figure(figsize=(10, 4))
#plt.plot(time, acclData[0])
#plt.plot(time, acclData[1])
for i in range(0,nAccels):
    plt.plot(time, acclData[i])
#plt.plot(time, accelerometer_data)
plt.title(f"Accelerometer Data: Trial {trialNum+1}, Sensors 1-20")
#plt.title(f"Accelerometer Data: Trial {trialNum+1}, Sensor {accelNum+1}")
plt.xlabel("Time (seconds)")
plt.ylabel("Acceleration")
plt.xlim(24,26)
plt.grid(True)
plt.show()
