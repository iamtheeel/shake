###
# main.py
# Joshua Mehlman
# MIC Lab
# Summer, 2025
###
# Parse the training files and extrace the expoch values
###
import pandas as pd


#dir = '/Volumes/Data/thesis/exp_Track_HPC/timeD/20250709-174319_ds4_lrWD-3/exp-4' # TimeDomain
#dir = '/Volumes/Data/thesis/exp_Track_HPC/sGram/20250710-074410_ds4_lrWD-1/exp-12' # sGram
#dir ='/Volumes/Data/thesis/exp_Track_HPC/downSample/20250712-153913_rickFStep_ds4/exp-1' #Ricker
#dir = '/Volumes/Data/thesis/exp_Track_HPC/morl/20250711-142208_ds4-1/exp-1' #morlet
#dir = '/Volumes/Data/thesis/exp_Track_HPC/cMorel/20250706-091032_ds-4_lr-wd_2/exp-3' #Complex morelet
dir = '/Volumes/Data/thesis/exp_Track_HPC/downSample/20250712-165823_fStep_ds4/exp-1' #Fstep

inFile = 'trainResults_byBatch.csv'
outFile = 'trainResults_epochOnly.csv'

dirFile_in = f'{dir}/{inFile}'
dirFile_out = f'{dir}/{outFile}'

# Load the CSV file
df = pd.read_csv(dirFile_in)  # adjust delimiter if needed

# Filter rows where 'batch' is NaN and 'lr' is not NaN
epoch_rows = df[df['batch'].isna() & df['lr'].notna()]

# Save to new CSV
epoch_rows.to_csv(dirFile_out, index=False)