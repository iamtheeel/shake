#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 09:27:03 2024

@author: zsjiang
"""

import h5py
import pandas as pd

# Path to the HDF5 file
file_path = 'TestData/Test_2/data/walking_hallway_single_person_APDM_001.hdf5'
#file_path = 'TestData/Test_2/data/walking_hallway_single_person_APDM_002.hdf5'
#file_path = 'TestData/Test_2/data/walking_hallway_single_person_APDM_003.hdf5'

# Load the general parameters dataset from the HDF5 file
with h5py.File(file_path, 'r') as hdf_file:
    general_parameters = hdf_file['experiment/general_parameters'][:]

#print(f"Sample Rate: {general_parameters[0]}Hz")
# Convert the structured array into a more readable format
general_parameters_list = []
for param in general_parameters:
    param_info = {
        'ID': param['id'],
        'Parameter': param['parameter'].decode('utf-8'),
        'Units': param['units'].decode('utf-8'),
        'Value': param['value'].decode('utf-8')
    }
    general_parameters_list.append(param_info)


print(f"{general_parameters_list[0]['Value']}")
# Convert the list to a DataFrame for better readability
general_parameters_df = pd.DataFrame(general_parameters_list)

# Display the DataFrame
print(general_parameters_df)
