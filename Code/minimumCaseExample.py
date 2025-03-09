###
# main.py
# Joshua Mehlman
# MIC Lab
# Spring, 2025
###
# Minimum Case DataLoad, fft, CWT Example
###

# Settings
dataFile = "../TestData/WalkingTest_Sensor8/walking_hallway_classroom_single_person.hdf5"


# Librarys needed
import h5py                      # For loading the data : pip install h5py
import matplotlib.pyplot as plt  # For plotting the data: pip install matplotlib
import numpy as np               # cool datatype, fun matix stuff and lots of math (we use the fft)    : pip install numpy==1.26.4
import pywt                      # The CWT              : pip install pywavelets

from foot_step_wavelet import FootStepWavelet, foot_step_cwt  # The custum footstep wavelet, in foot_step_wavelet.py

####
# Load the data 
print(f"Loading: {dataFile}")

with h5py.File(dataFile, 'r') as h5file:
    filePerams = h5file['experiment/general_parameters'][:]
    dataBlock_numpy = h5file['experiment/data'][:] #Load all the rows of data to the block, will not work without the [:]

print(f"{filePerams}")          #Show the peramiters
print(filePerams.dtype.names)   # Show the peramiter field names

#Extract the data capture rate from the file
dataCapRate_hz =filePerams[0]['value']#.decode('utf-8') # Data cap rate is the first entery (number 0)
dataCapUnits = filePerams[0]['units'].decode('utf-8')
print(f"Data Cap Rate ({filePerams[0]['parameter'].decode('utf-8')}): {dataCapRate_hz} {dataCapUnits}")

# Look at the shape of the data
print(f"Data type: {type(dataBlock_numpy)}, shape: {dataBlock_numpy.shape}")
# We happen to know that:
numTrials = dataBlock_numpy.shape[0]
numSensors = dataBlock_numpy.shape[1]
numTimePts = dataBlock_numpy.shape[2]
timeLen_s   = (numTimePts-1)/dataCapRate_hz # How far apart is each time point
print(f"The dataset has: {numTrials} trials, {numSensors} sensors, {numTimePts} timepoints")
print(f"The data was taken at {dataCapRate_hz} {dataCapUnits}, and is {timeLen_s} seconds long")


# Do some plotting
#Aka how to subplot
trial = 0  #Indexed from 0
#chToPlot = [5, 6, 7, 8, 9, 10]
chToPlot = [8, 9, 10]
#chToPlot = list(range(1,20+1)) # make an array of all the ch
plotTimeRange_s = [20, 25]

# First the time domain:
timePoints = np.linspace(0, timeLen_s, numTimePts) #start, stop, number of points
fig, axs = plt.subplots(len(chToPlot)) #Make the subplots for how many ch you want
fig.suptitle(f"Time Domain plot of trial: {trial} ch: {chToPlot}")

# Make room for the title, axis lables, and squish the plots up against eachother
fig.subplots_adjust(top = 0.95, bottom = 0.05, hspace=0, left = 0.15, right=0.99) # Mess with the padding (in percent)

for i, thisCh in enumerate(chToPlot):  # Enumerate will turbo charge the forloop, give the value and the idex
    # Plot the ch data
    thisDataRow = thisCh-1 #Indexed from 0
    timeData = dataBlock_numpy[trial,thisDataRow,:]  #Note: Numpy will alow negitive indexing (-1 = the last row)
    axs[i].plot(timePoints, timeData)

    # Set the ylimit
    axs[i].set_ylim([-0.01, 0.01])
    axs[i].set_xlim(plotTimeRange_s) # set the time range

    # Lable the axis
    axs[i].set_ylabel(f'Ch {chToPlot[i]}', fontsize=8)
#Only show the x-axis on the last plot
axs[i-1].get_xaxis().set_visible(True)
axs[i-1].set_xlabel(f"Time (s)")

plt.show()
#plt.savefig(f"overlayed_time.jpg")

#Now the frequency domain
# We would normaly run the plots as a function to not repeat code, but simple case and all
freqs = np.fft.rfftfreq(numTimePts, d=1.0/dataCapRate_hz)
fig, axs = plt.subplots(len(chToPlot)) #Make the subplots for how many ch you want
fig.suptitle(f"Frequency Domain plot of trial: {trial} ch: {chToPlot}")

# Make room for the title, axis lables, and squish the plots up against eachother
fig.subplots_adjust(top = 0.95, bottom = 0.05, hspace=0, left = 0.10, right=0.99) # Mess with the padding (in percent)
for i, thisCh in enumerate(chToPlot):  # Enumerate will turbo charge the forloop, give the value and the idex
    # Plot the ch data
    thisDataRow = thisCh-1 #Indexed from 0
    timeData = dataBlock_numpy[trial,thisDataRow,:]  #Note: Numpy will alow negitive indexing (-1 = the last row)

    # Calculate the fft
    # Apply a hanning window to minimize spectral leakate
    window = np.hanning(len(timeData))
    timeData = window*timeData
    freqData = np.fft.rfft(timeData) # Real value fft returns only below the nyquist
                                     # The data is returned as a complex value
    freqData_mag = np.abs(freqData) # Will only plot the magnitude

    axs[i].plot(freqs, freqData_mag)

    # Set the limits
    axs[i].set_ylim([0, .2])
    axs[i].set_xlim([0, 10]) # Only plot to 100Hz

    # Lable the axis
    axs[i].set_ylabel(f'Ch {chToPlot[i]}', fontsize=8)
#Only show the x-axis on the last plot
axs[i-1].get_xaxis().set_visible(True)
axs[i-1].set_xlabel(f"Frequency (Hz)")

plt.show()



# Cool, now look at cwt
cwtChList = [8, 9, 10]

# The range you want to see
min_freq = 1
max_freq = 100
numScales = 64 # How many frequencies to look at

#The wavelet peramiters
f0 = 1 # For cmorl, and footstep
bw = 0.8 # only footstep

# The frequencies wew want to look at
frequencies = np.logspace(np.log10(max_freq), np.log10(min_freq), numScales)

# Some calculateds
cwtChList_zeroIndex = [ch - 1 for ch in cwtChList]  # Convert to 0-based indexing
samplePeriod = 1/dataCapRate_hz

# The axis are rather nasty
def getYAxis(data_frequencies, yTicks):
    # Get the y-axis ticks and labels
    valid_ticks = yTicks.astype(int)[(yTicks >= 0) & (yTicks < len(data_frequencies))]
    freq_labels = data_frequencies[valid_ticks]
    return valid_ticks, freq_labels

def getXAxis(data_coefficients, xTicks):
    # Get the x-axis ticks and labels
    # Scale x-axis by sample rate to show time in seconds
    valid_ticks = xTicks.astype(int)[(xTicks >= 0) & (xTicks < data_coefficients.shape[1])]  # Only positive indices within data width
    time_labels = valid_ticks / dataCapRate_hz
    return valid_ticks, time_labels

#Complex wavelet
waveletName = f"cmorl{f0}-{bw}"
wavelet = pywt.ContinuousWavelet(waveletName)
center_freq = pywt.central_frequency(wavelet)
scales = center_freq / (frequencies * samplePeriod) 
# Cut the data to just the chs we want, and a smaller time frame (will take a year to calculate the whole thing)
dataPoint_from = int(plotTimeRange_s[0]*dataCapRate_hz)
dataPoint_to = int(plotTimeRange_s[1]*dataCapRate_hz)
data = dataBlock_numpy[:,cwtChList_zeroIndex,dataPoint_from:dataPoint_to]

# Calculate the CWT
print(f"Complex morlet input data shape: {data.shape}")
[data_coefficients, data_frequencies] = pywt.cwt(data, scales, wavelet=wavelet, sampling_period=samplePeriod)
cwtData_mag = np.abs(data_coefficients)

# We have cwtlines, trial, ch, timepoints
# Im show wants height, width, ch
print(f"Complex morelet Data shape: {cwtData_mag.shape}")
cwtData_mag = np.squeeze(cwtData_mag) # Remove the trial dimention
cwtData_mag = np.transpose(cwtData_mag, (0, 2, 1))
print(f"complex morelet cwtData_mag shape: {cwtData_mag.shape}")  # Should be (64, time, 3)

# Normalize
dataMax = np.max(cwtData_mag)
cwtData_mag = cwtData_mag/dataMax

plt.title(f"Complex Morelet, Center Frequency: {f0}, bandwidth: {bw}")
plt.imshow(cwtData_mag, aspect='auto')#, origin='lower')
validY_ticks, freq_labels = getYAxis(frequencies, plt.gca().get_yticks())
validX_ticks, time_labels = getXAxis(cwtData_mag[0], plt.gca().get_xticks())
plt.yticks(validY_ticks)
plt.xticks(validX_ticks)
plt.gca().set_yticklabels([f"{f:.1f}" for f in freq_labels])
plt.gca().set_xticklabels([f"{t:.1f}" for t in time_labels])

plt.xlabel("Time (s)")
plt.ylabel("Frequency (Hz)")

plt.show() # Open the plot

print(f"Done with CWT complex morelet")


#Footsetp
wavelet = FootStepWavelet(central_frequency=f0)
center_freq = wavelet.central_frequency
scales = center_freq / (frequencies * 1/dataCapRate_hz) 

# Calculate the cwt
print(f"footstep input Data shape: {data.shape}")
[data_coefficients, data_frequencies] = foot_step_cwt(data=data, scales=scales, 
                                                                sampling_period=samplePeriod, f_0=f0)
cwtData_mag = np.abs(data_coefficients) # It is not complex, but does go neg, so we still want the abs
#Note, this is where the "strips" come from

# We have cwtlines, trial, ch, timepoints
# Im show wants height, width, ch
print(f"Footstep: cwtData_mag shape: {data_coefficients.shape}")  # Should be (64, time, 3)
cwtData_mag = np.squeeze(cwtData_mag) # Remove the trial dimention
cwtData_mag = np.transpose(cwtData_mag, (0, 2, 1))
print(f"Footstep: cwtData_mag shape: {cwtData_mag.shape}")  # Should be (64, time, 3)

# Normalize
dataMax = np.max(cwtData_mag)
cwtData_mag = cwtData_mag/dataMax

plt.title(f"FootStep Wavelet, Center Frequency: {f0} ")
plt.imshow(cwtData_mag, aspect='auto')#, origin='lower')
validY_ticks, freq_labels = getYAxis(frequencies, plt.gca().get_yticks())
validX_ticks, time_labels = getXAxis(cwtData_mag[0], plt.gca().get_xticks())
plt.yticks(validY_ticks)
plt.xticks(validX_ticks)
plt.gca().set_yticklabels([f"{f:.1f}" for f in freq_labels])
plt.gca().set_xticklabels([f"{t:.1f}" for t in time_labels])

plt.xlabel("Time (s)")
plt.ylabel("Frequency (Hz)")

plt.show() # Open the plot
