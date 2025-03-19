###
# main.py
# Joshua Mehlman
# MIC Lab
# Spring, 2025
###
# Minimum Case DataLoad, fft, CWT Example
###

### Settings
# Data
dataFile = "../TestData/WalkingTest_Sensor8/walking_hallway_classroom_single_person.hdf5"

# What data are we interested in
plotTrial = 0  #Indexed from 0
dataTimeRange_s = [20, 30] # [0 0] for full dataset
dataFreqRange_hz = [1, 5]
# What data are we interested in
#chToPlot = [5, 6, 7, 8, 9, 10]
#chToPlot = list(range(1,20+1)) # all the chns
chToPlot = [6, 7, 10]
#chToPlot = [8, 9, 10]
#Ranges for the plotting
timeYRange = .02
freqYRange = 1

# CWT for only 3 ch (rgb)
cwtChList = [8, 9, 10]
#The wavelet peramiters
#waveletBase = 'mexh' # Shows low freq well
#waveletBase = 'morl'
#waveletBase = 'cmorl'
waveletBase = 'fstep'
f0 = 0.1 #2.14 # For cmorl, and footstep
bw = 0.8 # only cmorl
numScales = 64 # How many frequencies to look at



# Librarys needed
import h5py                      # For loading the data : pip install h5py
import matplotlib.pyplot as plt  # For plotting the data: pip install matplotlib
import numpy as np               # cool datatype, fun matix stuff and lots of math (we use the fft)    : pip install numpy==1.26.4
import pywt                      # The CWT              : pip install pywavelets

from foot_step_wavelet import FootStepWavelet, foot_step_cwt  # The custum footstep wavelet, in foot_step_wavelet.py


### 
# Functions
###

## Data Loaders
def loadData(dataFile):
    """
    Loads the data form an hdf version 5 file

    Args:
        dataFile: String of the data file name and location

    Returns:
        numpy: data 
        int: Data Capture Rate

    """
    print(f"Loading: {dataFile}")

    with h5py.File(dataFile, 'r') as h5file:
        filePerams = h5file['experiment/general_parameters'][:]
        dataFromFile = h5file['experiment/data'][:] #Load all the rows of data to the block, will not work without the [:]

    print(f"{filePerams}")          #Show the peramiters
    print(filePerams.dtype.names)   # Show the peramiter field names

    #Extract the data capture rate from the file
    dataCapRate_hz =filePerams[0]['value']#.decode('utf-8') # Data cap rate is the first entery (number 0)
    dataCapUnits = filePerams[0]['units'].decode('utf-8')
    print(f"Data Cap Rate ({filePerams[0]['parameter'].decode('utf-8')}): {dataCapRate_hz} {dataCapUnits}")
    
    # Look at the shape of the data
    print(f"Data type: {type(dataFromFile)}, shape: {dataFromFile.shape}")
    # We happen to know that:
    numTrials = dataFromFile.shape[0]
    numSensors = dataFromFile.shape[1]
    numTimePts = dataFromFile.shape[2]
    timeLen_s   = (numTimePts-1)/dataCapRate_hz # How far apart is each time point
    print(f"The data is {type(dataFromFile)}")
    print(f"The dataset has: {numTrials} trials, {numSensors} sensors, {numTimePts} timepoints")
    print(f"The data was taken at {dataCapRate_hz} {dataCapUnits}, and is {timeLen_s} seconds long")
    if dataTimeRange_s[1] == 0: dataTimeRange_s[1] = int(timeLen_s)

    return dataFromFile, dataCapRate_hz

## Data slicers
def sliceTheData(dataBlock:np, trial, chList, timeRange_sec):
    """
    Cuts the data by:
        ch
    
    Args:
        dataBlock: the raw data [Trial, ch, timePoints]

    Returns:
        numpy: the cut data
    """

    # The ch list
    chList_zeroIndexed = [ch - 1 for ch in chList]  # Convert to 0-based indexing

    # The time range
    dataPoint_from = int(timeRange_sec[0]*dataCapRate_hz)
    dataPoint_to = int(timeRange_sec[1]*dataCapRate_hz)

    # Ruturn the cut up data
    return dataBlock[trial, chList_zeroIndexed, dataPoint_from:dataPoint_to]

## Generate CWT
def generateCWT(data:np, freqRange, dataRate, waveletBase:str, f0=0, bw=0):
    # The frequencies wew want to look at
    if waveletBase == "cmorl":
        if f0 == 0 or bw ==0:
            print(f" ERROR!!, cmor must have f0 and bw != 0")
            exit()
        waveletName = f"{waveletBase}{f0}-{bw}"
    elif waveletBase == "fstep" and f0 != 0:
        waveletName = f"{waveletBase}{f0}"
    else: 
        waveletName = waveletBase

    frequencies = np.logspace(np.log10(freqRange[1]), np.log10(freqRange[0]), numScales)

    # Some calculateds
    #cwtChList_zeroIndex = [ch - 1 for ch in cwtChList]  # Convert to 0-based indexing
    samplePeriod = 1/dataRate

    # Get the ceter freq for calculating the scales to send to the CWT
    if waveletBase == 'fstep':
        wavelet = FootStepWavelet(central_frequency=f0)
        center_freq = wavelet.central_frequency
    else:
        wavelet = pywt.ContinuousWavelet(waveletName)
        center_freq = pywt.central_frequency(wavelet)

    scales = center_freq / (frequencies * samplePeriod) 

    # Calculate the CWT
    if waveletBase == 'fstep':
        [data_coefficients, data_frequencies] = foot_step_cwt(data=data, scales=scales, 
                                                                sampling_period=samplePeriod, f_0=f0)
    else:
        [data_coefficients, data_frequencies] = pywt.cwt(data, scales, wavelet=wavelet, sampling_period=samplePeriod)
    
    return data_coefficients, data_frequencies, waveletName

## Data Plottters
def dataPlot_2Axis(dataBlockToPlot:np, plotChList, trial:int, xAxisRange, yAxisRange, dataRate:int=0, domainToPlot:str="time"):
    """
    Plots the data in 2 axis (time or frequency domain)

    Args:
        dataBlockToPlot (Numpy): The data to be plotted [ch, timepoints]

    Returns:
        Null
    """
    numTimePts = dataBlockToPlot.shape[1]
    if domainToPlot == "time":
        xAxis_data = np.linspace(xAxisRange[0], xAxisRange[1], numTimePts) #start, stop, number of points
        xAxis_str = f"Time"
        xAxisUnits_str = "(s)"

    if domainToPlot == "freq":
        xAxis_data = np.fft.rfftfreq(numTimePts, d=1.0/dataRate)
        xAxis_str = f"Frequency"
        xAxisUnits_str = "(Hz)"
    title_str = f"{xAxis_str} Domain plot of trial: {trial} ch: {plotChList}"

    fig, axs = plt.subplots(len(plotChList)) #Make the subplots for how many ch you want
    fig.suptitle(title_str)

    # Make room for the title, axis lables, and squish the plots up against eachother
    fig.subplots_adjust(top = 0.95, bottom = 0.1, hspace=0, left = 0.1, right=0.99) # Mess with the padding (in percent)

    for i, thisCh in enumerate(plotChList):  # Enumerate will turbo charge the forloop, give the value and the idex
        # Plot the ch data
        timeD_data = dataBlockToPlot[i,:]  #Note: Numpy will alow negitive indexing (-1 = the last row)
        if domainToPlot == "time":
            yAxis_data = timeD_data
        if domainToPlot == "freq":
            # Calculate the fft
            # Apply a hanning window to minimize spectral leakage
            window = np.hanning(len(timeD_data))
            timeD_data = timeD_data - np.mean(timeD_data)  # Center the signal before FFT
            timeD_data_windowed = window*timeD_data
            timeD_data_windowed /= np.sum(window) / len(window)  # Normalize
            freqD_data = np.fft.rfft(timeD_data_windowed) # Real value fft returns only below the nyquist
                                                          # The data is returned as a complex value
            freqD_mag = np.abs(freqD_data)                  # Will only plot the magnitude
            yAxis_data = freqD_mag

        print(f"Ch {thisCh} Min: {np.min(yAxis_data)}, Max: {np.max(yAxis_data)}, Mean: {np.mean(yAxis_data)}")
        axs[i].plot(xAxis_data, yAxis_data)
    
        # Set the ylimit
        axs[i].set_ylim(yAxisRange)
        axs[i].set_xlim(xAxisRange) 

        # Label the axis
        axs[i].set_ylabel(f'Ch {plotChList[i]}', fontsize=8)
        if i < len(plotChList) - 1:
            axs[i].set_xticklabels([]) # Hide the xTicks from all but the last
    #Only show the x-axis on the last plot
    axs[-1].get_xaxis().set_visible(True)
    axs[-1].set_xlabel(f"{xAxis_str} {xAxisUnits_str}")

    #plt.savefig(f"overlayed_time.jpg")

## 3 Axis Plotters (AKA "Image" Generaters)
# The axis are rather nasty
def getYAxis(data_frequencies, yTicks):
    # Get the y-axis ticks and labels
    valid_ticks = yTicks.astype(int)[(yTicks >= 0) & (yTicks < len(data_frequencies))]
    freq_labels = data_frequencies[valid_ticks]
    return valid_ticks, freq_labels

def getXAxis(data_len, xTicks):
    # Get the x-axis ticks and labels
    # Scale x-axis by sample rate to show time in seconds
    valid_ticks = xTicks.astype(int)[(xTicks >= 0) & (xTicks < data_len)]  # Only positive indices within data width

    #if len(valid_ticks) == 0:  # If filtering removed all values, generate custom ticks
    #    valid_ticks = np.linspace(0, data_len - 1, num=10, dtype=int)

    time_labels = valid_ticks / dataCapRate_hz
    return valid_ticks, time_labels

def plotCWT(data, freqs, timeStart_s, waveletName):
    # Cmor data is real imaginary
    # Plot the magnitude, even for non-complex as that data is positive and negitive
    data_mag = np.abs(data) 
    # Imshow wants height, width, ch, but it comes in as lines, ch, timepoints
    imageData = np.transpose(data_mag, (0, 2, 1))
    print(f"CWT tranfomred shape after rearange: {imageData.shape}")  # Should be (64, time, 3)

    # Normalize
    dataMax = np.max(imageData)
    dataNorm = imageData/dataMax # we want to plot from 0 to 1

    plt.figure() # Make a new figure
    plt.title(f"Wavelet: {waveletName}")
    plt.imshow(dataNorm, aspect='auto')#, origin='lower')

    validY_ticks, freq_labels = getYAxis(freqs, plt.gca().get_yticks())
    validX_ticks, time_labels = getXAxis(dataNorm.shape[1], plt.gca().get_xticks())
    time_labels = time_labels + timeStart_s

    plt.yticks(validY_ticks)
    plt.xticks(validX_ticks)
    plt.gca().set_yticklabels([f"{f:.1f}" for f in freq_labels])
    plt.gca().set_xticklabels([f"{t:.1f}" for t in time_labels])

    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")

    #plt.show() # Open the plot

#### Do the stuff
# Load the data 
dataBlock_numpy, dataCapRate_hz = loadData(dataFile=dataFile)

# Get the parts of the data we are interested in:
print(f"Data len pre-cut: {dataBlock_numpy.shape}")
dataBlock_sliced = sliceTheData(dataBlock=dataBlock_numpy, trial=0, chList=chToPlot, timeRange_sec=dataTimeRange_s)
print(f"Data len: {dataBlock_sliced.shape}")

# Plot the data in the time domain
#timeYRange = np.max(np.abs(dataBlock_sliced))
dataPlot_2Axis(dataBlockToPlot=dataBlock_sliced, plotChList=chToPlot, trial=plotTrial, xAxisRange=dataTimeRange_s, yAxisRange=[-1*timeYRange, timeYRange], domainToPlot="time")

# Plot the data in the frequency domain
dataPlot_2Axis(dataBlockToPlot=dataBlock_sliced, plotChList=chToPlot, trial=plotTrial, xAxisRange=dataFreqRange_hz, yAxisRange=[0, freqYRange], dataRate=dataCapRate_hz, domainToPlot="freq")

# Generate CTW
dataBlock_forCWT = sliceTheData(dataBlock=dataBlock_numpy, trial=0, chList=cwtChList, timeRange_sec=dataTimeRange_s)
cwtData, cwtFreqs, waveletName  = generateCWT(data=dataBlock_forCWT, dataRate=dataCapRate_hz, waveletBase=waveletBase, freqRange=dataFreqRange_hz, f0=f0, bw=bw)

# Plot the Spectrogram
plotCWT(data=cwtData, freqs=cwtFreqs, timeStart_s=dataTimeRange_s[0], waveletName=waveletName)

plt.show() # SHow all the plots