###
# Footfall
# Joshua Mehlman
# MIC Lab
# Spring, 2025
###
# CWT Transform
###
import numpy as np
import pywt
import matplotlib.pyplot as plt
import logging
import os
import time
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class cwt:
    def __init__(self, configs, dataConfigs):
        #Data information
        self.samplePeriod = 1/dataConfigs.sampleRate_hz
        self.sampleRate_hz = dataConfigs.sampleRate_hz
        self.configs = configs

        self.saveDir = f"{configs['plts']['pltDir']}/cwt"
        if not os.path.exists(self.saveDir): os.makedirs(self.saveDir)

    def setupWavelet(self, wavelet_name ):
        # The wavelet
        self.wavelet_name = wavelet_name

        self.numScales = 240#480 #This is the height of the image

        self.wavelet = pywt.ContinuousWavelet(self.wavelet_name)
        self.level = 8 #Number of iterations, for descrete wavelets
        self.length = 256

        # Our wave function
        self.wavelet_fun, self.wavelet_Time = self.wavelet.wavefun(length=self.length)#, level=self.level) 
        #logger.info(f"Wavelet time from: {self.wavelet_Time[0]} to {self.wavelet_Time[-1]}")
    def setScale(self, logScale=True):
        #scales = np.arange(1, self.numScales)  # N+1 number of scales (frequencies)
        min_freq = 5 #Hz
        max_freq = self.sampleRate_hz/2 #Nyquist
        if logScale:
            central_freq = pywt.central_frequency(self.wavelet)
            min_scale = central_freq / (max_freq * self.samplePeriod)
            max_scale = central_freq / (min_freq * self.samplePeriod)
            self.scales = np.logspace(np.log10(min_scale), np.log10(max_scale), self.numScales)
            #self.scales = np.logspace(0, np.log10(self.numScales/2), self.numScales) # Aprox exponential scale
        else:
            # Use the wavelet's central frequency to create a linear scale
            frequencies = np.linspace(max_freq, min_freq, self.numScales) # No 0 frequency
            #frequencies = np.linspace(0, self.sampleRate_hz/2, self.numScales)
            center_freq = pywt.central_frequency(self.wavelet)
            self.scales = center_freq / (frequencies * self.samplePeriod)
        #logger.info(f"Scales: {self.scales}")
        #logger.info(f"Scales: {self.scales.shape}")

    def cwtTransform(self, data, logScale=True):
        # Perform continuous wavelet transform using the defined wavelet

        start_time = time.time()
        [transformedData, data_frequencies] = pywt.cwt(data, self.scales, wavelet=self.wavelet, sampling_period=self.samplePeriod)
        #logger.info(f"Frequencies: {self.data_frequencies.shape}")
        #logger.info(f"Frequencies: {self.data_frequencies}")
        end_time = time.time()
        logger.info(f"CWT Calculation time: {end_time - start_time} seconds")

        # Keep only every nth column (time point) from the results
        data_coefficients = transformedData
        #step_size = 5  # Adjust this to control output resolution: TODO: make this a config
        #self.data_coefficients = data[:, ::step_size]
        #logger.info(f"Coefficients: type: {type(transformedData[0][0])}, shape: {transformedData.shape},  shape: {data_coefficients.shape}") #each dataum is a numpy.complex128

        return data_coefficients, data_frequencies
        
        # Plot the CWT coefficients
    def plotCWTransformed_data(self, data_coefficients, data_frequencies, run, timeWindow, subjectLabel, ch, logScale, save=False, display=True):
        plt.figure(figsize=(12, 8))
        colorMap = 'gray'
        #colorMap = 'jet'
        plt.imshow(np.abs(data_coefficients), aspect='auto', cmap=colorMap)
        plt.colorbar(label='Magnitude')

        #plt.ylabel('Scale')
        plt.ylabel('Frequency (Hz)')
        # Get y-axis ticks and convert frequencies to Hz
        yticks = plt.gca().get_yticks()
        valid_ticks = yticks.astype(int)[(yticks >= 0) & (yticks < len(data_frequencies))]
        freq_labels = data_frequencies[valid_ticks]
        plt.gca().set_yticks(valid_ticks)
        plt.gca().set_yticklabels([f"{f:.1f}" for f in freq_labels])  # Then set the labels



        plt.xlabel('Time (s)')
        # Scale x-axis by sample rate to show time in seconds
        #xticks = plt.gca().get_xticks()
        xticks = plt.gca().get_xticks()
        valid_ticks = xticks.astype(int)[(xticks >= 0) & (xticks < data_coefficients.shape[1])]  # Only positive indices within data width
        time_labels = valid_ticks * self.samplePeriod
        plt.gca().set_xticks(valid_ticks)
        plt.gca().set_xticklabels([f"{t:.1f}" for t in time_labels])
        plt.title(f'CWT using {self.wavelet_name} wavelet (scales={self.numScales})\n'
                  f'run: {run}, timeWindow: {timeWindow}, subjectLabel: {subjectLabel}, channel: {ch}, logScale: {logScale}')
        if save:
            fileName = f"{self.saveDir}/{self.wavelet_name}_run{run}_timeWindow{timeWindow}_subjectLabel{subjectLabel}_channel{ch}_logScale{logScale}.jpg"
            plt.savefig(fileName)
            logger.info(f"Saved: {fileName}")
        if display:
            plt.show()

    
    def plotWavelet(self ):
        # Get the wavelet function values

        plt.figure(figsize=(10, 8))
        plt.title(f'{self.wavelet_name} wavelet')
        plt.plot(self.wavelet_Time, np.real(self.wavelet_fun), label='Real')
        plt.plot(self.wavelet_Time, np.imag(self.wavelet_fun), label='Imaginary')
        plt.xlabel('Time')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.grid(True)
        plt.show()


    def trackWavelet(self, dataPrep, dataumNumber, ch):
        logger.info(f"Tracking wavelet transform")
        #Get a data for the wavelet transform
        data, run, timeWindow, subjectLabel = self.getDataForWaveletTracking(dataPrep, dataumNumber, ch)
        logScales = [True]
        #logScales = [True, False]

        #Setup the wavelet
        for wavelet_name_base in self.configs['cwt']['wavelet']:
            if wavelet_name_base == 'cmor': 
                f0_list = self.configs['cwt']['waveLet_center_freq']
                bw_list = self.configs['cwt']['waveLet_bandwidth']
            else:
                f0_list = [0.0]
                bw_list = [0.0]

            for center_freq in f0_list:
                for bandwidth in bw_list:
                    if wavelet_name_base == 'cmor': 
                        wavelet_name = f"{wavelet_name_base}-{center_freq}-{bandwidth}"
                    else:
                        wavelet_name = f"{wavelet_name_base}"

                    for logScale in logScales:
                        logger.info(f"Wavelet name: {wavelet_name}")
                        self.setupWavelet(wavelet_name)
                        self.setScale(logScale=logScale)
                        cwtData, cwtFrequencies = self.cwtTransform(data) 
                        self.plotCWTransformed_data(cwtData, cwtFrequencies, run, timeWindow, subjectLabel, ch, logScale=logScale, save=True, display=True)


    def getDataForWaveletTracking(self, dataPrep, dataumNumber, ch ):
        logger.info(f"Getting data for wavelet tracking")
        dataPrep.resetData() # The data is windows, channels, dataPoints

        logger.info(f"dataPrep.dataConfigs.chList: {dataPrep.dataConfigs.chList}, ch: {ch}")
        chNumInList = dataPrep.dataConfigs.chList.index(ch)
        thisData = dataPrep.data[dataumNumber][chNumInList]
        run = dataPrep.runList_raw[dataumNumber]
        timeWindow = dataPrep.startTimes_raw[dataumNumber]
        subjectLabel = dataPrep.subjectList_raw[dataumNumber]
        ch = dataPrep.dataConfigs.chList[chNumInList]
        logger.info(f"subjectLabel: {subjectLabel}, run: {run}, timeWindow: {timeWindow}, channel: {ch}")

        return thisData, run, timeWindow, subjectLabel