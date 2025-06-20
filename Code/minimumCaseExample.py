###
# main.py
# Joshua Mehlman
# MIC Lab
# Spring, 2025
###
# Minimum Case DataLoad, time domain, fft, Spectrogram, CWT Example
###

### Settings
# Data/_h
#dataFile = "../dataOnFastDrive/data_acquisition_p6.hdf5" # Data from paper
#chToPlot = [1, 2, 3, 4]
#cwtChList = chToPlot #[1, 2, 3]
#dataTimeRange_s = [10.75, 15.75] # [0 0] for full dataset
dataTimeRange_s = [0, 0] # [0 0] for full dataset

#dataFreqRange_hz = [0.5, 10] # If the second argument is 0, use the nyquist
#dataFreqRange_hz = [0.5, 2.5] # If the second argument is 0, use the nyquist

#dataFile = "../TestData/WalkingTest_Sensor8/walking_hallway_classroom_single_person.hdf5" # Ch 10 test
#dataFile = "../TestData/Test_2/data/walking_hallway_single_person_APDM_002.hdf5"
#dataFile = "/home/josh/winShare/joshTest_1652.hdf5"
#dataFile = "/home/josh/winShare/joshTest_1652_2.hdf5"
#dataFile = "/home/josh/winShare/joshTest_413.hdf5"
#dataFile = "/home/josh/winShare/joshTest_413_413.hdf5"
#dataFile = "/home/josh/winShare/joshTest_413_1652.hdf5"
dataFile = "/Users/theeel/schoolDocs/MIC/NSF_Floor_Vib_Camera-Labeling/StudentData/25_06_18/Yoko_s3_Run1.hdf5" #Has trigger time

# What data are we interested in
#dataTimeRange_s = [15, 55] # [0 0] for full dataset
dataFreqRange_hz = [1, 0] # If the second argument is 0, use the nyquist
#dataFreqRange_hz = [1, 100] # If the second argument is 0, use the nyquist
#dataFreqRange_hz = [0.5, 10] # If the second argument is 0, use the nyquist
logFreq = False
# What data are we interested in
chToPlot = [6, 5, 4]
cwtChList = chToPlot 
#chToPlot = list(range(1,20+1)) # all the chns
#chToPlot = [6, 7, 10]
#chToPlot = [8, 9, 10]
#cwtChList = [6, 7, 10] # CWT for only 3 ch (rgb)
#Ranges for the plotting
#timeYRange = .02
#freqYRange = .2

pltXRange = dataFreqRange_hz #[10, 45]

#The wavelet peramiters
#waveletBase = 'mexh' # Shows low freq well
#waveletBase = 'morl'
waveletBase = 'cmorl'
#waveletBase = 'fstep'
#f0 = 2.14 #10 # For footstep
f0 = 10 # For cmorl
bw = 0.8 # only cmorl
numScales = 256 # How many frequencies to look at


# Librarys needed
from datetime import datetime           # Built in
import h5py                             # For loading the data : pip install h5py
import matplotlib.pyplot as plt         # For plotting the data: pip install matplotlib
import numpy as np                      # cool datatype, fun matix stuff and lots of math (we use the fft)    : pip install numpy==1.26.4
from scipy.signal import spectrogram    # For spectrogram
import pywt                             # The CWT              : pip install pywavelets
#scikit-learn
from foot_step_wavelet import FootStepWavelet, foot_step_cwt  # The custum footstep wavelet, in foot_step_wavelet.py

### 
# Functions
###

## Data Loaders
def print_attrs(name, obj): #From Chatbot
        print(f"\n📂 Path: {name}")
        for key, val in obj.attrs.items():
            print(f"  🔧 Attribute - {key}: {val}")
        if isinstance(obj, h5py.Dataset):
            print(f"  📊 Dataset - Shape: {obj.shape}, Dtype: {obj.dtype}")

def get_peram(perams, peramName:str, asStr=False):
    mask = perams['parameter'] == peramName.encode()
    matches = perams[mask]
    if len(matches) > 0:
        if asStr:
            peram_value = matches['value'][0].decode('utf8')
            #peram_value = perams[perams['parameter'] == peramName.encode()]['value'][0].decode('utf8')
        else:
            peram_value= perams[perams['parameter'] == peramName.encode()]['value'][0] 
        units_value = perams[perams['parameter'] == peramName.encode()]['units'][0].decode('utf-8')
    else: 
        peram_value = None
        units_value = None
    #print(f"{peramName}: {peram_value} {units_value}")

    return peram_value, units_value

def get_perams(perams, peramName:str, asType='dateTime'):
    values = [
        #row['value'].decode()
        datetime.fromtimestamp(float(row['value'].decode()))
        for row in perams
            if row['parameter'] == peramName.encode()
    ]
    return values 

def loadPeramiters(dataFile):
    with h5py.File(dataFile, 'r') as h5file:
        #h5file.visititems(print_attrs)
        # Move this to a saved peramiter
        nTrials = h5file['experiment/data'][:].shape[0] #Load all the rows of data to the block, will not work without the [:]
        filePerams = h5file['experiment/general_parameters'][:]

    #Extract the data capture info from the file
    dataCapRate_hz, dataCapUnits = get_peram(filePerams, 'fs')
    recordLen_s, _ = get_peram(filePerams, 'record_length')
    preTrigger_s, _ = get_peram(filePerams, 'pre_trigger')

    #print(filePerams.dtype.names)   # Show the peramiter field names
    #print(f"experiment/general_parameters: {filePerams}")          #Show the peramiters

    # Now that we know which is the timepoints
    #print(f"The data was taken at {dataCapRate_hz} {dataCapUnits}, and is {recordLen_s} seconds long")

    if dataFreqRange_hz[1] == 0: dataFreqRange_hz[1] = dataCapRate_hz/2
    if dataTimeRange_s[1] == 0: dataTimeRange_s[1] = int(recordLen_s)


    return dataCapRate_hz, recordLen_s, preTrigger_s, nTrials

def loadData(dataFile, trial=-1):
    """
    Loads the data form an hdf version 5 file

    Args:
        dataFile: String of the data file name and location

    Returns:
        numpy: data 
        int: Data Capture Rate

    """
    print(f"Loading file: {dataFile}")
    with h5py.File(dataFile, 'r') as h5file:
        if trial >= 0:
            dataFromFile = h5file['experiment/data'][trial,:,:] #Load trial in question
            runPerams = h5file['experiment/specific_parameters']#Load all the rows of data to the block, will not work without the [:]
            triggerTimes, _ = get_peram(runPerams, 'triggerTime', asStr=False)
            if triggerTimes != None:
                triggerTimes = next(
                                row['value'] for row in runPerams
                                if row['parameter'] == b'triggerTime' and row['id'] == trial
                                ).decode() #Get from string
                triggerTimes = datetime.fromtimestamp(float(triggerTimes))
            print(f"Loaded trial: {trial}")
        elif trial == -1: # Load the whole thing
            dataFromFile = h5file['experiment/data'][:] #Load all the rows of data to the block, will not work without the [:]
            runPerams = h5file['experiment/specific_parameters']#Load all the rows of data to the block, will not work without the [:]
            triggerTimes = get_perams(runPerams, 'triggerTime', asType='dateTime')
        # Otherwize, we are just after the peramiters

    if trial <=0:
        #print(filePerams.dtype.names)   # Show the peramiter field names
        #print(f"experiment/general_parameters: {filePerams}")          #Show the peramiters
        # Look at the shape of the data
        print(f"Data type: {type(dataFromFile)}, shape: {dataFromFile.shape}")

    # We happen to know that:
    if trial < 0:
        numTrials = dataFromFile.shape[0]
        numSensors = dataFromFile.shape[1]
        numTimePts = dataFromFile.shape[2]
        print(f"The dataset has: {numTrials} trials, {numSensors} sensors, {numTimePts} timepoints")

    else:
        numSensors = dataFromFile.shape[0]
        numTimePts = dataFromFile.shape[1]
        if trial == 0:
            print(f"The dataset has: {numSensors} sensors, {numTimePts} timepoints")

    return dataFromFile, triggerTimes
    """

        filePerams = h5file['experiment/general_parameters'][:]
        if trial >= 0:
            #print(f"Loading trial: {trial}")
            dataFromFile = h5file['experiment/data'][trial,:,:] #Load trial in question
        elif trial == -1: # Load the whole thing
            print(f"Loading the full dataset")
            dataFromFile = h5file['experiment/data'][:] #Load all the rows of data to the block, will not work without the [:]
        # Otherwize, we are just after the peramiters

    #Extract the data capture rate from the file
    # Data cap rate is the first entery (number 0)
    dataCapRate_hz =filePerams[0]['value']  # Some files needs decode, others can't have it
    #dataCapRate_hz =int(filePerams[0]['value'].decode('utf-8'))  # Some files needs decode, others can't have it

    dataCapUnits = filePerams[0]['units'].decode('utf-8')
    if trial <=0:
        print(f"experiment/general_parameters: {filePerams}")          #Show the peramiters
        print(filePerams.dtype.names)   # Show the peramiter field names
        print(f"Data Cap Rate ({filePerams[0]['parameter'].decode('utf-8')}): {dataCapRate_hz} {dataCapUnits}")
    
        # Look at the shape of the data
        print(f"Data type: {type(dataFromFile)}, shape: {dataFromFile.shape}")

    # We happen to know that:
    if trial < 0:
        numTrials = dataFromFile.shape[0]
        numSensors = dataFromFile.shape[1]
        numTimePts = dataFromFile.shape[2]
        print(f"The dataset has: {numTrials} trials, {numSensors} sensors, {numTimePts} timepoints")

    else:
        numSensors = dataFromFile.shape[0]
        numTimePts = dataFromFile.shape[1]
        if trial == 0:
            print(f"The dataset has: {numSensors} sensors, {numTimePts} timepoints")

    if trial <=0:
        # Now that we know which is the timepoints
        timeLen_s   = (numTimePts-1)/dataCapRate_hz # How far apart is each time point
        if dataTimeRange_s[1] == 0: dataTimeRange_s[1] = int(timeLen_s)
        if dataFreqRange_hz[1] == 0: dataFreqRange_hz[1] = dataCapRate_hz/2
        print(f"The data was taken at {dataCapRate_hz} {dataCapUnits}, and is {timeLen_s} seconds long")

    return dataFromFile, dataCapRate_hz
    """

def downSampleData(self, data, downSample):
        from scipy.signal import decimate

        #logger.info(f" dataLen from file: {self.dataConfigs.dataLen_pts}")
        #logger.info(f"Before downsample shape: {np.shape(data)} ")
        nTrials, nCh, timePoints = data.shape
        downSampled_data = np.empty((nTrials, nCh, timePoints // downSample))  
        for trial in range(nTrials):
            for ch in range(nCh):
                downSampled_data[trial, ch] = decimate(data[trial, ch], 
                                                       downSample, 
                                                       ftype='iir', 
                                                       zero_phase=True)

        return downSampled_data, dataCapRate_hz/downSample


## Data slicers
def sliceTheData(dataBlock:np, chList, timeRange_sec, trial=-1):
    """
    Cuts the data by:
        ch
    
    Args:
        dataBlock: the raw data [Trial, ch, timePoints]
        trial: if -1, then the data is already pre-cut for trial

    Returns:
        numpy: the cut data
    """

    # The ch list
    chList_zeroIndexed = [ch - 1 for ch in chList]  # Convert to 0-based indexing
    print(f"ChList index: {chList_zeroIndexed}")

    # The time range
    dataPoint_from = int(timeRange_sec[0]*dataCapRate_hz)
    dataPoint_to = int(timeRange_sec[1]*dataCapRate_hz)

    # Ruturn the cut up data
    if trial > 0:
        return dataBlock[trial, chList_zeroIndexed, dataPoint_from:dataPoint_to]
    else:
        return dataBlock[chList_zeroIndexed, dataPoint_from:dataPoint_to]

def generateSpectragram(data:np, chList, dataRate, timeRes= 1, overlap=0.5):
    # Compute spectrograms for each channel
    Sxx_list = []
    for i, ch in enumerate(chList):
        print(f" Data Block[i]: {data[i].shape}")
        # nDataPoints in each window: Larger = better freq res, lower time res
        nperseg = int(timeRes * dataRate) 
        noverlap = int(nperseg)*overlap # Overlap processing % overlap
        freqs, times, Sxx = spectrogram(data[i], fs=dataRate, nperseg=nperseg, noverlap=noverlap)
        Sxx_list.append(Sxx)

    Sxx_np = np.stack(Sxx_list, axis=0)  # Shape: (n_channels, n_frequencies, n_time)
    #Sxx_np = np.log10(Sxx_np)

    return Sxx_np, freqs


## Generate CWT
def generateCWT(data:np, freqRange, dataRate, waveletBase:str, f0=0, bw=0, log=True):
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

    if log:
        frequencies = np.logspace(np.log10(freqRange[0]), np.log10(freqRange[1]), numScales)
    else:
        frequencies = np.linspace(freqRange[0], freqRange[1], numScales)

    # Some calculateds
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

    #print(data_frequencies)
    #print(frequencies)
    
    return data_coefficients, data_frequencies, waveletName

def applyButterWorth(data, fs):
    from scipy.signal import butter, filtfilt
    # 0.1 makes the dip slower and higher freq
    #b, a = butter(N=2, Wn=np.array([0.5, 100])/(0.5*fs), btype='bandpass', analog=False)  # Filter coefficients
    b, a = butter(N=2, Wn=np.array([0.5, 4])/(0.5*fs), btype='bandpass', analog=False)  # Filter coefficients

    filtData = []
    print(f"Data type before butterworth: {type(data)}, {data.shape}")
    for ch in range(data.shape[0]):
        filtData.append(filtfilt(b, a, data[ch, :])) 
        #filtData.append(filtfilt(b, a, data[ch, :], padlen=1000))  Makes worse
        #filtData.append(filtfilt(b, a, data[ch, :], method='gust'))  Makes edge effects worse
    filtData = np.array(filtData) # Put the data back into numpy
    print(f"Data type: {type(filtData)}, ")
    return  filtData

## Data Plottters
def dataPlot_2Axis(dataBlockToPlot:np, plotChList, trial:int, xAxisRange, yAxisRange, dataRate:int=0, 
                   domainToPlot:str="time", logX=False, logY=False, title="", save=""):
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
    title_str = f"{xAxis_str} Domain plot of trial: {trial} ch: {plotChList}{title}"

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
    
        # Set the Axis limits and scale
        axs[i].set_xlim(xAxisRange) 
        axs[i].set_ylim(yAxisRange)
        if logX: axs[i].set_xscale('log')  # Set log scale
        if logY: axs[i].set_yscale('log')  # Set log scale

        # Label the axis
        axs[i].set_ylabel(f'Ch {plotChList[i]}', fontsize=8)
        if i < len(plotChList) - 1:
            axs[i].set_xticklabels([]) # Hide the xTicks from all but the last

    #Only show the x-axis on the last plot
    axs[-1].get_xaxis().set_visible(True)
    axs[-1].set_xlabel(f"{xAxis_str} {xAxisUnits_str}")

    #plt.savefig(f"images/{save}_{domainToPlot}_trial-{trial}.jpg")
    #plt.close()
    return xAxis_data # Save for later use

## 3 Axis Plotters (AKA "Image" Generaters)
def plot_3D(data, freqs, title, extraBump = 1, log=False, freqScale=None, save="", ch=""):
    # Cmor data is real imaginary
    # Plot the magnitude, even for non-complex as that data is positive and negitive
    data_mag = np.abs(data) 
    imageData = data_mag
    #print(f"CWT tranfomred shape after rearange: {imageData.shape}")  # Should be (64, time, 3)

    # Normalize
    dataMax = np.max(imageData)
    dataNorm = extraBump*imageData/dataMax # we want to plot from 0 to 1

    plt.figure() # Make a new figure
    plt.title(f"{title}, trial:{trial}, ch: {ch}")
    #plt.imshow(dataNorm, aspect='auto')#, origin='lower')
    if ch == "":
        plt.imshow(dataNorm, aspect='auto', origin='lower', 
               extent=[dataTimeRange_s[0], dataTimeRange_s[1], min(freqs), max(freqs)])
    else: # Single ch
        plt.imshow(dataNorm, aspect='auto', origin='lower', cmap='viridis',
               extent=[dataTimeRange_s[0], dataTimeRange_s[1], min(freqs), max(freqs)])
    
    if log: plt.yscale('log')
    if freqScale != None:
        plt.ylim(freqScale)

    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")

    '''
    fileName = f"images/{save}_trial-{trial}_ch{ch}.jpg"
    print(f"Saving file: {fileName}")
    plt.savefig(fileName)
    plt.close()
    ''' 

#### Do the stuff
# get the peramiters
dataCapRate_hz, recordLen_s, preTrigger_s, nTrials = loadPeramiters(dataFile=dataFile) 
print(f"Data cap rate: {dataCapRate_hz} Hz, Record Length: {recordLen_s} sec, pretrigger len: {preTrigger_s}sec, Trials: {nTrials}")

#timeYRange = np.max(np.abs(dataBlock_numpy))

# 2-22.21-APDM-data.xlsx has 27 enterys, so this is probably the data
# Is this even in the right order???
trialList = [21, 34, 35, 36, 37, 39, 42, 45, 46,
                     22, 23, 24, 25, 26, 27, 28, 30,
                     50, 51, 53, 54, 57, 58, 60, 61, 62, 64]
trialList = [0]

#for trial in range(20): # Cycle through the trials
#for trial in range(dataBlock_numpy.shape[0]): # Cycle through the trials
for i, trial in enumerate(trialList): # Cycle through the trials
    print(f"Running Trial: {trial}")
    dataBlock_numpy, triggerTime = loadData(dataFile=dataFile, trial=trial)
    if triggerTime != None:
        print(f"Trigger Time: {triggerTime.strftime("%Y-%m-%d %H:%M:%S.%f")}")

    # Get the parts of the data we are interested in:
    #downSampledData, dataCapRate_hz = downSampleData(dataBlock_numpy, 4) #4x downsample... may need fudging, have not tryed in minCaseEx

    print(f"Data len pre-cut: {dataBlock_numpy.shape}")
    dataBlock_sliced = sliceTheData(dataBlock=dataBlock_numpy, trial=-1, chList=chToPlot, timeRange_sec=dataTimeRange_s) # -1 if the data is already with the trial
    #dataBlock_sliced = sliceTheData(dataBlock=dataBlock_numpy, trial=trial, chList=chToPlot, timeRange_sec=dataTimeRange_s)
    print(f"Data len: {dataBlock_sliced.shape}")
    
    # Plot the data in the time domain
    timeYRange = 0.01
    #timeYRange = np.max(np.abs(dataBlock_sliced))
    timeSpan = dataPlot_2Axis(dataBlockToPlot=dataBlock_sliced, plotChList=chToPlot, trial=trial, 
                              xAxisRange=dataTimeRange_s, yAxisRange=[-1*timeYRange, timeYRange], domainToPlot="time", save="original")
    #freqYRange = [0.001, 0.1]
    freqYRange = [0.01, 10]
    freqSpan = dataPlot_2Axis(dataBlockToPlot=dataBlock_sliced, plotChList=chToPlot, trial=trial, 
                              xAxisRange=dataFreqRange_hz, yAxisRange=freqYRange, 
                              dataRate=dataCapRate_hz, domainToPlot="freq", logX=logFreq, logY=True, save="original")

    #plt.show() # Open the plots
    #exit()

    # Generate and plot the CWT
    cwtData, cwtFreqs, waveletName  = generateCWT(data=dataBlock_sliced, freqRange=dataFreqRange_hz, dataRate=dataCapRate_hz, waveletBase=waveletBase, f0=f0, bw=bw, log=False)
    cwtData = np.transpose(cwtData, (0, 2, 1)) #Needs to be [h, w, ch] for plot
    plot_3D(data=cwtData, freqs=cwtFreqs, title=f"Wavelet: {waveletName}", extraBump=1, save=f"{waveletName}", log=logFreq)

    ## The spectrogram
    # TODO: Only for the freq range in quesiton
    #TODO: Cut the data not display the range
    # TODO: add arguments for timeRes and overlap
    spectraGramData, spectraFreqs = generateSpectragram(dataBlock_sliced, cwtChList, dataCapRate_hz, timeRes=1, overlap=0.99)
    spectraGramData = np.transpose(spectraGramData, (1, 2, 0)) #Needs to be [h, w, ch] for plot
    plot_3D(spectraGramData, freqs=spectraFreqs, title="Spectragram", extraBump=10, freqScale=dataFreqRange_hz, log=False)

    
    #Pre Filter the data
    print(f"Apply Filters and norms")
    from scipy.signal import wiener
    dataBlock_sliced = wiener(dataBlock_sliced)
#
    # Time Domain Norm to max, do by trial, but will need to ba across the dataset
    # Note: small changeup, the paper is norm to ch, we are norm to datablock
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    # The scaler scales across colls, we want rows: So transpose the input, and output
    dataBlock_sliced = scaler.fit_transform(dataBlock_sliced.T).T # Fit and transform the data
    dataBlock_sliced = dataBlock_sliced - np.mean(dataBlock_sliced) # Remove the offset

    # Now the butterworth:
    dataBlock_sliced = applyButterWorth(dataBlock_sliced, fs=dataCapRate_hz)

    # Plot the post Transfromed Data
    timeYRange = 1
    timeSpan = dataPlot_2Axis(dataBlockToPlot=dataBlock_sliced, plotChList=chToPlot, trial=trial, 
                              xAxisRange=dataTimeRange_s, yAxisRange=[-1*timeYRange, timeYRange], domainToPlot="time",
                              title=": After TimeD Data Mods", save="postMod")
    
    # Plot the data in the frequency domain
    # TODO: Move the fft outside of the plot
    #freqYRange = [0.1, 20]
    freqSpan = dataPlot_2Axis(dataBlockToPlot=dataBlock_sliced, plotChList=chToPlot, trial=trial, 
                              xAxisRange=dataFreqRange_hz, yAxisRange=freqYRange, 
                              dataRate=dataCapRate_hz, domainToPlot="freq", logX=logFreq, logY=True,# save="postMod")
                              title=": After TimeD Data Mods", save="postMod")

    # Combined plot
    print(f"Generate and plot the CWT Data")
    cwtData, cwtFreqs, waveletName  = generateCWT(data=dataBlock_sliced, freqRange=dataFreqRange_hz, dataRate=dataCapRate_hz, waveletBase=waveletBase, f0=f0, bw=bw, log=False)
    cwtData = np.transpose(cwtData, (0, 2, 1)) #Needs to be [h, w, ch] for plot
    plot_3D(data=cwtData, freqs=cwtFreqs, title=f"Wavelet: {waveletName}", extraBump=1, save=f"{waveletName}", log=logFreq)
    
    plt.show() # SHow all the plots