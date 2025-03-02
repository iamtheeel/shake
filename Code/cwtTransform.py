###
# Footfall
# Joshua Mehlman
# MIC Lab
# Spring, 2025
###
# CWT Transform
###
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import torch

import pywt #pip install pywavelets
from foot_step_wavelet import FootStepWavelet, foot_step_cwt

#from dataLoader import normClass
import typing
if typing.TYPE_CHECKING: #Fix circular import
    from fileStructure import fileStruct

from jFFT import jFFT_cl

import logging
from pathlib import Path
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class cwt:
    def __init__(self, fileStructure:"fileStruct", configs, dataConfigs):
        print(f"\n")
        logger.info(f"----------------------      Get cwt peramiters   ----------------------")
        self.fileStructure = fileStructure
        #Data information
        self.samplePeriod = 1/dataConfigs.sampleRate_hz
        self.sampleRate_hz = dataConfigs.sampleRate_hz
        self.configs = configs

        #self.saveDir = f"{configs['plts']['pltDir']}/cwt"
        #if not os.path.exists(self.saveDir): os.makedirs(self.saveDir)

        self.minData = 1000000.0
        self.maxData = 0.0

        # List of wavelets
        #wavelets = pywt.wavelist()
        #print(wavelets)
        self.wavelet_name=None
        self.wavelet_base = None
        self.f0 = None
        self.bw = None

    def setupWavelet(self, wavelet_base, sampleRate_hz, f0=1.0, bw=1.0, useLogForFreq=False):
        self.useLogScaleFreq  = useLogForFreq

        self.min_freq = self.configs['cwt']['fMin'] #5 #Hz
        self.max_freq = self.configs['cwt']['fMax']

        self.wavelet_base = wavelet_base
        self.f0= f0
        self.bw = bw
        logger.info(f"Wavelet base: {self.wavelet_base}, f0: {self.f0}, bw: {self.bw}")
        if wavelet_base == "mexh" or  wavelet_base=="morl" or wavelet_base == 'None':
            self.wavelet_name = wavelet_base
        elif wavelet_base == "fstep": #No arguments, arguments handled seperately
            self.wavelet_name = f"{wavelet_base}-{f0}"
        else:  #cmorl
            self.wavelet_name = f"{wavelet_base}{f0}-{bw}"
        # The wavelet
        logger.info(f"Wavelet name: {self.wavelet_name}")


        self.fileStructure.setCWT_dir(self) 
        if self.wavelet_name == 'None':return #Now that we have the name, we can skip the rest on None

        self.numScales = self.configs['cwt']['numScales']  #480 #This is the height of the image

        if self.wavelet_base == 'fstep':
            self.wavelet = FootStepWavelet(central_frequency=f0)
        else:
            #Center freq and bw are imbedded in the name, or not used
            self.wavelet = pywt.ContinuousWavelet(self.wavelet_name)
        self.level = 8 #Number of iterations, for descrete wavelets
        self.length = 512 #at 256 with f_0 = 10, it looks a little ragged

        # Our wave function
        self.wavelet_fun, self.wavelet_Time = self.wavelet.wavefun(length=self.length)#, level=self.level) 
        logger.info(f"{self.wavelet_name}, Complex:{np.iscomplexobj(self.wavelet_fun)}")

        self.setFreqScale(freqLogScale=self.useLogScaleFreq)

        self.plotWavelet(sRate=sampleRate_hz, save=True, show=False )

    def setFreqScale(self, freqLogScale=True):
        #scales = np.arange(1, self.numScales)  # N+1 number of scales (frequencies)
        
        if self.max_freq == 0: self.max_freq = self.sampleRate_hz/2 #Nyquist

        if self.wavelet_base != 'fstep':
            center_freq = pywt.central_frequency(self.wavelet)
        else: 
            center_freq = self.wavelet.central_frequency

        #min_scale = center_freq / (self.max_freq * self.samplePeriod)
        #max_scale = center_freq / (self.min_freq * self.samplePeriod)

        if freqLogScale:
            self.frequencies = np.logspace(np.log10(self.max_freq), np.log10(self.min_freq), self.numScales)
            #self.scales = np.logspace(np.log10(min_scale), np.log10(max_scale), self.numScales)
        else:
            self.frequencies = np.linspace(self.max_freq, self.min_freq, self.numScales) 

        #self.frequencies = center_freq / (self.scales * self.samplePeriod) #not used
        self.scales = center_freq / (self.frequencies * self.samplePeriod)

        #TODO: write to log file
        #logger.info(f"Scales: {self.scales.shape}")
        #logger.info(f"Scales: \n{self.scales}")
        #logger.info(f"Frequencies: \n{self.frequencies}")

    def cwtTransform(self, data, debug=False):
        # Perform continuous wavelet transform using the defined wavelet
        if debug:
            logger.info(f"cwtTransform: wavelet: {self.wavelet_name}")
        #logger.info(f"Transforming data: {type(data)}")

        start_time = time.time()
        if self.wavelet_base == 'fstep':
            if debug: logger.info(f"Fstep")
            [data_coefficients, data_frequencies] = foot_step_cwt(data=data, scales=self.scales, 
                                                                sampling_period=self.samplePeriod, f_0=self.f0)
        else:
            if debug: logger.info(f"pywt.cwt")
            [data_coefficients, data_frequencies] = pywt.cwt(data, self.scales, wavelet=self.wavelet, sampling_period=self.samplePeriod)
        #logger.info(f"Frequencies: {self.data_frequencies}")
        end_time = time.time()
        if debug:
            logger.info(f"CWT output datashapes | transformedData: {data_coefficients.shape}, {data_coefficients.dtype}, data_frequencies: {data_frequencies.shape}, time: {end_time - start_time}s")

        # Keep only every nth column (time point) from the results?
        #step_size = 5  # Adjust this to control output resolution: TODO: make this a config
        #self.data_coefficients = data[:, ::step_size]
        #logger.info(f"Coefficients: type: {type(transformedData[0][0])}, shape: {transformedData.shape},  shape: {data_coefficients.shape}") #each dataum is a numpy.complex128

        return data_coefficients, data_frequencies
        

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

    def plotWavelet(self, sRate=0, show = False, save = True):
        # Get the wavelet function values
        complexInput = False
        if np.iscomplexobj(self.wavelet_fun): complexInput = True
        #if self.wavelet_name == "mexh" : complexInput = False

        # Plot the time Domain
        expStr = ""
        titleStr = f"{expStr}{self.wavelet_base}"
        if self.f0 != 0 or self.bw !=0:
            if self.wavelet_base != 'mexh' or self.wavelet_base != 'morl':
                if self.wavelet_base == 'fstep':
                    titleStr = f"{titleStr}, f0={self.f0}"
                else:
                    titleStr = f"{titleStr}, f0={self.f0}, bw={self.bw}"
        plt.figure(figsize=(10, 8))
        plt.title(f'{titleStr}')
        plt.plot(self.wavelet_Time, np.real(self.wavelet_fun), label='Real')
        if np.iscomplexobj(self.wavelet_fun):
            plt.plot(self.wavelet_Time, np.imag(self.wavelet_fun), label='Imaginary')
        plt.xlabel('Time')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.grid(True)
        if save:
            timeDFileName = f"{self.wavelet_name}_timeD.jpg"
            timeDFileNamePath = f"{self.fileStructure.dataDirFiles.saveDataDir.waveletDir.waveletDir_name}/{timeDFileName}"
            logger.info(f"Saving Wavelet Time Plot: {timeDFileNamePath}")
            plt.savefig(timeDFileNamePath)
        if show:
            plt.show()
        plt.close()

        # Bode plot
        fftClass = jFFT_cl()
        #ch, datapoint
        nSamp = len(self.wavelet_Time)
        #logger.info(f"timeD | shape{type(self.wavelet_Time)}, len: {nSamp}")
        #logger.info(f"{self.wavelet_Time}")
        waveletDt = self.wavelet_Time[1] - self.wavelet_Time[0]
        if sRate == 0:
            sRate = 1/waveletDt
        freqList = fftClass.getFreqs(sRate=sRate, tBlockLen=nSamp, complex=complexInput)
        #logger.info(f"Freq D | shape : {fftData.shape}")
        #logger.info(f"{freqList}")
        fftData = fftClass.calcFFT(self.wavelet_fun, complex=complexInput) #mag, phase

        if complexInput: #Re-center so neg is before positive
            freqList = np.fft.fftshift(freqList)


        fig, axs = plt.subplots(2, 1, figsize=(10,10)) #w, h figsize in inches?
        fig.subplots_adjust(top = 0.95, bottom = 0.05, hspace=0.10, left = 0.10, right=0.99) 
        #plt.figure(figsize=(10, 8))
        fig.suptitle(f'{titleStr}, Data Sample Rate: {sRate}Hz')
        axs[0].plot(freqList, fftData[0])
        axs[0].set_yscale('log')
        axs[0].set_ylabel(f"Magnigude (dB)")
        #axs[0].set_ylim(bottom=1e-7, top=None)
        #axs[0].set_xlim([0,15])
        #axs[0].grid(which="both")
        axs[0].minorticks_on()
        axs[0].grid(True, which='minor', linestyle=':', alpha=0.2)
        axs[0].grid(True, which='major', linestyle='-', alpha=0.6)

        phase = fftData[1]
        unwrapped_phase = np.unwrap(phase) 
        axs[1].plot(freqList, np.degrees(unwrapped_phase)) #Rad --> deg
        axs[1].set_ylabel(f"Phase (deg)")
        axs[1].set_xlabel(f"Frequency (Hz, for Sam Rate)")
        axs[1].minorticks_on()
        axs[1].grid(True, which='minor', linestyle=':', alpha=0.2)
        axs[1].grid(True, which='major', linestyle='-', alpha=0.6)
        #axs[1].set_ylim([-180, 180])

        if save:
            freqDFilename = f"{self.wavelet_name}_freqD.jpg"
            freqDFilePathName = f"{self.fileStructure.dataDirFiles.saveDataDir.waveletDir.waveletDir_name}/{freqDFilename}"
            logger.info(f"Saving Wavelet Plots: {freqDFilePathName}")
            plt.savefig(freqDFilePathName)

        if show:
            plt.show()
        plt.close()
    
    def cwtTransformBatch(self, data):
        #logger.info(f"before cwt: {data.shape}")
        data, freqs = self.cwtTransform(data=data.numpy())
        #logger.info(f"after cwt: {data.shape}")
        data = np.transpose(data, (1, 2, 0, 3))           # we want: windows, ch, freqs, timepoints
        #logger.info(f"after transpose: {data.shape}")
        data = torch.from_numpy(data)

        return data