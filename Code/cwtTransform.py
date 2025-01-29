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
from jFFT import jFFT_cl
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

        self.minData = 1000000.0
        self.maxData = 0.0

    def setupWavelet(self, wavelet_name, useLogForFreq ):
        # The wavelet
        self.wavelet_name = wavelet_name

        self.numScales = 240#480 #This is the height of the image

        self.wavelet = pywt.ContinuousWavelet(self.wavelet_name)
        self.level = 8 #Number of iterations, for descrete wavelets
        self.length = 512 #at 256 with f_0 = 10, it looks a little ragged

        # Our wave function
        self.wavelet_fun, self.wavelet_Time = self.wavelet.wavefun(length=self.length)#, level=self.level) 

        self.setFreqScale(freqLogScale=useLogForFreq)
        #logger.info(f"Wavelet time from: {self.wavelet_Time[0]} to {self.wavelet_Time[-1]}")
    def setFreqScale(self, freqLogScale=True):
        #scales = np.arange(1, self.numScales)  # N+1 number of scales (frequencies)
        min_freq = 5 #Hz
        max_freq = self.sampleRate_hz/2 #Nyquist
        if freqLogScale:
            central_freq = pywt.central_frequency(self.wavelet)
            min_scale = central_freq / (max_freq * self.samplePeriod)
            max_scale = central_freq / (min_freq * self.samplePeriod)
            self.scales = np.logspace(np.log10(min_scale), np.log10(max_scale), self.numScales)
            #self.scales = np.logspace(0, np.log10(self.numScales/2), self.numScales) # Aprox exponential scale
            logger.info(f"Scales: {self.scales.shape}")
        else:
            # Use the wavelet's central frequency to create a linear scale
            frequencies = np.linspace(max_freq, min_freq, self.numScales) # No 0 frequency
            #frequencies = np.linspace(0, self.sampleRate_hz/2, self.numScales)
            center_freq = pywt.central_frequency(self.wavelet)
            self.scales = center_freq / (frequencies * self.samplePeriod)
        #logger.info(f"Scales: {self.scales}")
        #logger.info(f"Scales: {self.scales.shape}")

    def cwtTransform(self, data):
        # Perform continuous wavelet transform using the defined wavelet
        #logger.info(f"Transforming data: {type(data)}")

        start_time = time.time()
        [transformedData, data_frequencies] = pywt.cwt(data, self.scales, wavelet=self.wavelet, sampling_period=self.samplePeriod)
        #logger.info(f"Frequencies: {self.data_frequencies}")
        end_time = time.time()
        logger.info(f"CWT output datashapes | transformedData: {transformedData.shape}, data_frequencies: {data_frequencies.shape}, time: {end_time - start_time}s")

        # Keep only every nth column (time point) from the results
        data_coefficients = transformedData
        #step_size = 5  # Adjust this to control output resolution: TODO: make this a config
        #self.data_coefficients = data[:, ::step_size]
        #logger.info(f"Coefficients: type: {type(transformedData[0][0])}, shape: {transformedData.shape},  shape: {data_coefficients.shape}") #each dataum is a numpy.complex128

        return data_coefficients, data_frequencies
        
        # Plot the CWT coefficients
    def plotCWTransformed_data_1ch(self, data_coefficients, data_frequencies, run, timeWindow, subjectLabel, ch, freqLogScale, save=False, display=True):
        plt.figure(figsize=(12, 8))
        colorMap = 'gray'
        #colorMap = 'jet'
        plt.imshow(np.abs(data_coefficients), aspect='auto', cmap=colorMap)
        plt.colorbar(label='Magnitude')

        #plt.ylabel('Scale')
        plt.ylabel('Frequency (Hz)')
        valid_ticks, freq_labels = self.getYAxis(data_frequencies, plt.gca().get_yticks())
        plt.gca().set_yticks(valid_ticks)
        plt.gca().set_yticklabels([f"{f:.1f}" for f in freq_labels])  # Then set the labels

        plt.xlabel('Time (s)')
        valid_ticks, time_labels = self.getXAxis(data_coefficients, plt.gca().get_xticks())   
        plt.gca().set_xticks(valid_ticks)
        plt.gca().set_xticklabels([f"{t:.1f}" for t in time_labels])

        plt.title(f'CWT using {self.wavelet_name} wavelet (scales={self.numScales})\n'
                  f'run: {run}, timeWindow: {timeWindow}, subjectLabel: {subjectLabel}, channel: {ch}, freqLogScale: {freqLogScale}')
        if save:
            fileName = f"{self.saveDir}/{self.wavelet_name}_run{run}_timeWindow{timeWindow}_subjectLabel{subjectLabel}_channel{ch}_freqLogScale{freqLogScale}.jpg"
            plt.savefig(fileName)
            logger.info(f"Saved: {fileName}")
        if display:
            plt.show()

    def get3ChData(self, plotChList, data_coefficients, dataChList, normTo_max = 0, normTo_min = 0):
        """
        Input is complex
        Returns as mag
        """
        nCh = len(plotChList) #Had better be 3
        #data comming in as ch, freq, time
        rgb_data = np.zeros((data_coefficients.shape[1], data_coefficients.shape[2], nCh)) #Height(freq), width(time), ch
        #logger.info(f"data: {data_coefficients.shape}, rgb_data: {rgb_data.shape}")

        # Normalize each channel's data to 0-1 range and assign to RGB channels
        logger.info(f"get3ChData | normTo_max: {normTo_max}, normTo_min: {normTo_min}")
        for i, thisCh in enumerate(plotChList):
            #Data comming in as (ch, freq, time)
            channel_data = np.abs(data_coefficients[dataChList.index(thisCh),:,:]) # Converts real/imag to mag
            if normTo_max == 0: 
                norm_max = np.max(channel_data)
                norm_min = np.min(channel_data)
                if norm_max > self.maxData: self.maxData = norm_max
                if norm_min < self.minData: self.minData = norm_min
            else:
                norm_min = normTo_min
                norm_max = normTo_max

            normalized_data = (channel_data - norm_min) / (norm_max - norm_min)
            #logger.info(f"channel {thisCh} channel_data: {channel_data.shape}, {channel_data.max()}, {channel_data.min()}")
            #logger.info(f"channel {thisCh} normalized_data: {normalized_data.shape}, {normalized_data.max()}, {normalized_data.min()}")

            #freqs, times, ch?
            rgb_data[:,:,i] = normalized_data
        if normTo_max == 0:
            logger.info(f"channel_max: {self.maxData}, channel_min: {self.minData}")

        return rgb_data

    def plotCWTransformed_data_3CH(self, data_coefficients, data_frequencies, run, timeWindow, subjectLabel, plotChList, dataChList, logScale, save=False, display=True):
        print(f"data_coefficients: {data_coefficients.shape}, data_frequencies: {data_frequencies.shape}")
        nCh = 3
        # Create a figure with 3 color channels overlaid
        plt.figure(figsize=(12, 8))

        # Create RGB array to hold the 3 channels
        rgb_data = self.get3ChData(plotChList, data_coefficients, dataChList)
        print(f"rgb_data: {rgb_data.shape}")

        # Display the combined RGB image
        plt.imshow(rgb_data, aspect='auto')
        plt.colorbar(label='Magnitude')

        plt.ylabel('Frequency (Hz)')
        valid_ticks, freq_labels = self.getYAxis(data_frequencies, plt.gca().get_yticks())
        plt.gca().set_yticks(valid_ticks)
        plt.gca().set_yticklabels([f"{f:.1f}" for f in freq_labels])

        plt.xlabel('Time (s)')
        valid_ticks, time_labels = self.getXAxis(data_coefficients[0], plt.gca().get_xticks())
        plt.gca().set_xticks(valid_ticks)
        plt.gca().set_xticklabels([f"{t:.1f}" for t in time_labels])

        plt.title(f'CWT using {self.wavelet_name} wavelet (scales={self.numScales})\n'
                  f'run: {run}, timeWindow: {timeWindow}, subjectLabel: {subjectLabel}\n'
                  f'Normalized to 1 | Red: Ch{plotChList[0]}, Green: Ch{plotChList[1]}, Blue: Ch{plotChList[2]}')

        if save:
            fileName = f"{self.saveDir}/{self.wavelet_name}_run{run}_timeWindow{timeWindow}_subjectLabel{subjectLabel}_3CH_logScale{logScale}.jpg"
            plt.savefig(fileName)
            logger.info(f"Saved: {fileName}")
        if display:
            plt.show()

    def getYAxis(self, data_frequencies, yTicks):
        # Get the y-axis ticks and labels
        valid_ticks = yTicks.astype(int)[(yTicks >= 0) & (yTicks < len(data_frequencies))]
        freq_labels = data_frequencies[valid_ticks]
        return valid_ticks, freq_labels

    def getXAxis(self, data_coefficients, xTicks):
        # Get the x-axis ticks and labels
        # Scale x-axis by sample rate to show time in seconds
        valid_ticks = xTicks.astype(int)[(xTicks >= 0) & (xTicks < data_coefficients.shape[1])]  # Only positive indices within data width
        time_labels = valid_ticks * self.samplePeriod
        return valid_ticks, time_labels

    def plotWavelet(self, sRate=0, saveDir="", show = False, save = True):
        # Get the wavelet function values

        # Plot the time Domain
        plt.figure(figsize=(10, 8))
        plt.title(f'{self.wavelet_name} wavelet')
        plt.plot(self.wavelet_Time, np.real(self.wavelet_fun), label='Real')
        plt.plot(self.wavelet_Time, np.imag(self.wavelet_fun), label='Imaginary')
        plt.xlabel('Time')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.grid(True)
        if save:
            timeDFileName = f"{self.wavelet_name}_timeD.jpg"
            timeDFileNamePath = f"{saveDir}/{timeDFileName}"
            logger.info(f"Saving Wavelet Plots: {timeDFileNamePath}")
            plt.savefig(timeDFileNamePath)

        # Bode plot
        fftClass = jFFT_cl()
        #ch, datapoint
        nSamp = len(self.wavelet_Time)
        logger.info(f"timeD | shape{type(self.wavelet_Time)}, len: {nSamp}")
        #logger.info(f"{self.wavelet_Time}")
        waveletDt = self.wavelet_Time[1] - self.wavelet_Time[0]
        if sRate == 0:
            sRate = 1/waveletDt
        freqList = fftClass.getFreqs(sRate=sRate, tBlockLen=nSamp)
        fftData = fftClass.calcFFT(self.wavelet_fun) #mag, phase
        logger.info(f"Freq D | shape : {fftData.shape}")

        fig, axs = plt.subplots(2, 1, figsize=(10,10)) #w, h figsize in inches?
        fig.subplots_adjust(top = 0.95, bottom = 0.05, hspace=0.10, left = 0.10, right=0.99) 
        #plt.figure(figsize=(10, 8))
        fig.suptitle(f'{self.wavelet_name} wavelet, Data Sample Rate: {sRate}Hz')
        axs[0].plot(freqList, fftData[0])
        axs[0].set_yscale('log')
        axs[0].set_ylabel(f"Magnigude (dB)")
        #axs[0].set_ylim(bottom=1e-7, top=None)
        #axs[0].set_xlim([0,15])
        #axs[0].grid(which="both")
        axs[0].minorticks_on()
        axs[0].grid(True, which='minor', linestyle=':', alpha=0.2)
        axs[0].grid(True, which='major', linestyle='-', alpha=0.6)

        axs[1].plot(freqList, np.degrees(fftData[1]))
        axs[1].set_ylabel(f"Phase (deg)")
        axs[1].set_xlabel(f"Frequency (Hz, 1/f0)")
        axs[1].minorticks_on()
        axs[1].grid(True, which='minor', linestyle=':', alpha=0.2)
        axs[1].grid(True, which='major', linestyle='-', alpha=0.6)
        axs[1].set_ylim([-180, 180])

        if save:
            timeDFileName = f"{self.wavelet_name}_freqD.jpg"
            timeDFileNamePath = f"{saveDir}/{timeDFileName}"
            logger.info(f"Saving Wavelet Plots: {timeDFileNamePath}")
            plt.savefig(timeDFileNamePath)

        if show:
            plt.show()
