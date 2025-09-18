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
        self.numScales = self.configs['cwt']['numScales']  #480 #This is the height of the image

        self.wavelet_base = wavelet_base
        self.f0= f0
        self.bw = bw
        logger.info(f"Wavelet base: {self.wavelet_base}, f0: {self.f0}, bw: {self.bw}")

        ## Note: some wavelets are complex, some are real
        asMagnitude = self.configs['cwt']['runAsMagnitude']

        #if wavelet_base == "ricker" or  wavelet_base=="morl" or wavelet_base== 'spectroGram' or wavelet_base == 'None' or wavelet_base == 'db4':
        if wavelet_base == 'cmorl' or wavelet_base == 'shan':
            # https://pywavelets.readthedocs.io/en/latest/ref/cwt.html
            # cmorB-C where B is the bandwidth parameter and C is the center frequency
            self.wavelet_name = f"{wavelet_base}{bw}-{f0}"
        elif wavelet_base == "fstep" or wavelet_base == 'cfstep': #No arguments, arguments handled seperately
            self.wavelet_name = f"{wavelet_base}-{f0}"
        else:  
            self.wavelet_name = wavelet_base
        # The wavelet
        logger.info(f"Wavelet name: {self.wavelet_name}")


        if self.wavelet_name == 'None' or self.wavelet_name == 'spectroGram':
            logger.info(f"wavelet_name: {self.wavelet_name}")
            self.fileStructure.setCWT_dir(self) 
            #  None wavelet is never complex
            #  Burn the spectrogram problem later
            return #Now that we have the name, we can skip the rest on None

        if self.wavelet_base == 'fstep' or self.wavelet_base == 'cfstep':
            self.wavelet = FootStepWavelet(central_frequency=f0, complex=complex)
        else:
            #Center freq and bw are imbedded in the name, or not used
            if self.wavelet_base == 'ricker':
                self.wavelet = pywt.ContinuousWavelet('mexh') # mexh is probimatic, lets re-name to ricker
            else:
                self.wavelet = pywt.ContinuousWavelet(self.wavelet_name)
        #level = 10 #Number of iterations, for descrete wavelets
        self.length = 512 #at 256 with f_0 = 10, it looks a little ragged

        # Our wave function
        #self.wavelet_fun, self.wavelet_Time = self.wavelet.wavefun(level=level)#, level=self.level) 
        self.wavelet_fun, self.wavelet_Time = self.wavelet.wavefun(length=self.length)#, level=self.level) 
        #if wavelet_base == 'cmor' or wavelet_base == 'cmorl' or wavelet_base == 'cfmorl' or wavelet_base == 'cfstep' or wavelet_base == 'shan': complex = True  
        #else: complex = False
        isComplex = np.iscomplexobj(self.wavelet_fun)
        if isComplex:
            if asMagnitude: self.wavelet_name = f"{self.wavelet_name}_mag"
            else:           self.wavelet_name = f"{self.wavelet_name}_complex"
        else:
            self.wavelet_name = f"{self.wavelet_name}_real"

        logger.info(f"{self.wavelet_name}, Complex:{isComplex}, len: {len(self.wavelet_fun)}, dt: {self.wavelet_Time[1]-self.wavelet_Time[0] }")

        self.fileStructure.setCWT_dir(self, isComplex=complex, asMagnitude=asMagnitude) 

        #f0, bw
        f0, bw = self.getF0_BW()
        logger.info(f"Calculated f0: {f0}, bw: {bw}")

        self.setFreqScale(freqLogScale=self.useLogScaleFreq)
        f0, bw = self.plotWavelet(sRate=0, save=True, show=False ) #Use 0 to use the wavelet default
        f0, bw = self.plotWavelet(sRate=sampleRate_hz, save=True, show=False )
        logger.info(f"Calculated f0: {f0}, bw: {bw}")

    def setFreqScale(self, freqLogScale=True):
        #scales = np.arange(1, self.numScales)  # N+1 number of scales (frequencies)
        
        if self.max_freq == 0: self.max_freq = self.sampleRate_hz/2 #Nyquist

        if self.wavelet_base == 'fstep' or self.wavelet_base == 'cfstep':
            center_freq = self.wavelet.central_frequency
        else: 
            center_freq = pywt.central_frequency(self.wavelet)

        #min_scale = center_freq / (self.max_freq * self.samplePeriod)
        #max_scale = center_freq / (self.min_freq * self.samplePeriod)

        if freqLogScale:
            self.frequencies = np.logspace(np.log10(self.min_freq), np.log10(self.max_freq), self.numScales)
            #self.scales = np.logspace(np.log10(min_scale), np.log10(max_scale), self.numScales)
        else:
            self.frequencies = np.linspace(self.min_freq, self.max_freq, self.numScales) 

        #self.frequencies = center_freq / (self.scales * self.samplePeriod) #not used
        self.scales = center_freq / (self.frequencies * self.samplePeriod)

        #TODO: write to log file
        #logger.info(f"Scales: {self.scales.shape}")
        #logger.info(f"Scales: \n{self.scales}")
        #logger.info(f"Frequencies: \n{self.frequencies}")

    def getF0_BW(self):
        # mexh does not have phi
        psi, x = self.wavelet.wavefun(length=512) 
        #phi, psi, x = self.wavelet.wavefun(level=10)  # psi is the wavelet

        # Get the frequencies, and do an fft on psi
        #dt = x[1] - x[0]
        dt = 1/(1706.666667/4)
        f = np.fft.fftfreq(len(psi), d=dt)
        #print(f"getF0_BW | len: {len(psi)}, dt: {dt}")
        Psi = np.abs(np.fft.fft(psi))

        # Keep positive frequencies
        mask = f > 0
        f = f[mask]
        Psi = Psi[mask]

        # Get approximate center frequency
        f0_index = np.argmax(Psi)
        f0 = f[f0_index]

        # Find Full Width at Half Maximum (FWHM)
        half_max = Psi[f0_index] / 2
        indices = np.where(Psi >= half_max)[0]
        bw = f[indices[-1]] - f[indices[0]]

        return f0, bw

    def cwtTransform(self, data, debug=False):
        # Perform continuous wavelet transform using the defined wavelet
        if debug:
            logger.info(f"cwtTransform: wavelet: {self.wavelet_name}")
            logger.info(f"Transforming data: {type(data)}")

        start_time = time.time()
        if self.wavelet_base == 'fstep' or self.wavelet_base == 'cfstep':
            if debug: logger.info(f"{self.wavelet_base}.foot_step_cwt, complex: {complex}")
            [data_coefficients, data_frequencies] = foot_step_cwt(data=data, scales=self.scales, 
                                                                  sampling_period=self.samplePeriod, f_0=self.f0, 
                                                                  complex=complex)
            if debug:
                logger.info(f"data_coefficients dtype: {data_coefficients.dtype}, complex: {data_coefficients.dtype.kind == 'c'}")
        else:
            if debug: logger.info(f"pywt.cwt")
            [data_coefficients, data_frequencies] = pywt.cwt(data, self.scales, wavelet=self.wavelet, sampling_period=self.samplePeriod)
        #logger.info(f"Frequencies: {self.data_frequencies}")
        end_time = time.time()
        if debug:
            logger.info(f"CWT output datashapes | transformedData: {data_coefficients.shape}, {data_coefficients.dtype}, data_frequencies: {data_frequencies.shape}, time: {end_time - start_time}s")

        data_coefficients = np.transpose(data_coefficients, (1, 0, 2))           # we want: ch, freqs, timepoints
        # Keep only every nth column (time point) from the results?
        #logger.info(f"Coefficients: type: {type(data_coefficients[0][0])},  shape: {data_coefficients.shape}") #each dataum is a numpy.complex128

        ## TODO: allow for complex data
        if self.configs['cwt']['runAsMagnitude'] and np.iscomplexobj(data_coefficients):
            #logger.info(f"Converting complex CWT output to magnitude")
            data_coefficients = np.abs(data_coefficients) #Magnitude
            #data_coefficients = np.angle(data_coefficients) #phase

        return data_coefficients, data_frequencies
        

    def getYAxis(self, data_frequencies, yTicks):
        # Get the y-axis ticks and labels
        valid_ticks = yTicks.astype(int)[(yTicks >= 0) & (yTicks < len(data_frequencies))]
        freq_labels = data_frequencies[valid_ticks]
        return valid_ticks, freq_labels

    def getXAxis(self, data_len, xTicks):
        # Get the x-axis ticks and labels
        # Scale x-axis by sample rate to show time in seconds
        valid_ticks = xTicks.astype(int)[(xTicks >= 0) & (xTicks < data_len)]  # Only positive indices within data width
        time_labels = valid_ticks * self.samplePeriod
        return valid_ticks, time_labels

    def plotWavelet(self, sRate=0, show = False, save = True):
        # Get the wavelet function values
        tFontSize = 28
        complexInput = False
        if np.iscomplexobj(self.wavelet_fun): complexInput = True

        # Plot the time Domain
        expStr = ""
        if self.wavelet_base == "fstep" or self.wavelet_base == 'cfstep':
            titleStr = f"Wavelet: {expStr}cust"
        else:
            titleStr = f"Wavelet: {expStr}{self.wavelet_base}"
        print(f"wavelet base: {self.wavelet_base}")

        if self.wavelet_base != 'ricker' and self.wavelet_base != 'morl':
            if self.wavelet_base == 'fstep' or self.wavelet_base == 'cfstep':
                titleStr = f"{titleStr}, f0={self.f0}"
            else:
                titleStr = f"{titleStr}, f0={self.f0}, bw={self.bw}"
        print(f"title str: {titleStr}")


        # Get the bandwidth and f0
        fftClass = jFFT_cl()
        nSamp = len(self.wavelet_Time)
        #logger.info(f"timeD | shape{type(self.wavelet_Time)}, len: {nSamp}")
        #logger.info(f"{self.wavelet_Time}")
        if sRate == 0: #If we call with no sample rate, use the wavelet default
            waveletDt = self.wavelet_Time[1] - self.wavelet_Time[0]
            logger.info(f"dt: {waveletDt}")
            sRate = 1/waveletDt
            atRateOrDefault = f"default-{sRate}Hz"
        else:
            atRateOrDefault = f"dataRate-{sRate}Hz"
        freqList = fftClass.getFreqs(sRate=sRate, tBlockLen=nSamp, complex=complexInput)
        #logger.info(f"Freq D | shape : {fftData.shape}")
        #logger.info(f"len: {nSamp}, sRate: {sRate}, dt: {1/sRate}")
        fftData = fftClass.calcFFT(self.wavelet_fun, complex=complexInput) #mag, phase

        if complexInput: #Re-center so neg is before positive
            freqList = np.fft.fftshift(freqList)

        #Get the f0 and bw from the wavelet
        # find the max value
        f0_index = np.argmax(fftData[0]) #fft data is mag/phase
        #print(f"plotWavelet | {f0_index}")
        f0 = freqList[f0_index] # the assosiated freq

        # Find Full Width at Half Maximum (FWHM)
        half_max = fftData[0, f0_index] / 2
        indices = np.where(fftData[0] >= half_max)[0]
        bw = freqList[indices[-1]] - freqList[indices[0]]
        if save:
            freqDFilename = f"{atRateOrDefault}_{self.wavelet_name}"
            freqDFilePathName = f"{self.fileStructure.dataDirFiles.saveDataDir.waveletDir.waveletDir_name}/{freqDFilename}.txt"
            with open(freqDFilePathName, "w") as file:
                file.write(f"Wavelet: {self.wavelet_name}\n")
                file.write(f"Sample Frequency: {sRate:.4f} Hz\n")
                file.write(f"Center frequency (f0): {f0:.4f} Hz\n")
                file.write(f"Bandwidth Frequency (bw): {bw:.4f} Hz\n")


        self.plotWaveletTime(titleStr, tFontSize, save, show)
        if np.iscomplexobj(self.wavelet_fun):
            self.plotWaveletTime_complex(titleStr, tFontSize, save, show)
        self.plotWaveletBode(titleStr, sRate, tFontSize, freqList, fftData, save, freqDFilename, show)     


        return f0, bw

    def plotWaveletTime_complex(self, titleStr, tFontSize, save, show):
        from mpl_toolkits.mplot3d import Axes3D  # Required for 3D plotting

        fig = plt.figure(figsize=(10, 8))

        ax = fig.add_subplot(111, projection='3d')
        ax.plot(self.wavelet_Time, np.real(self.wavelet_fun), np.imag(self.wavelet_fun) )

        azimRot_deg = -10 # 10 deg ccw about z
        ax.view_init(elev=30, azim=-60+azimRot_deg)

        ax.set_title(f'{titleStr}', fontsize=tFontSize)
        ax.set_xlabel('Time')
        ax.set_ylabel('Real Part')
        ax.set_zlabel('Imaginary Part')

        self.savePlot(plt, f"{self.wavelet_name}_timeD-complex.jpg", save, show)

    def plotWaveletTime(self, titleStr, tFontSize, save, show):
        plt.figure(figsize=(10, 8))
        plt.title(f'{titleStr}', fontsize=tFontSize)
        plt.plot(self.wavelet_Time, np.real(self.wavelet_fun), label='Real')
        if np.iscomplexobj(self.wavelet_fun):
            plt.plot(self.wavelet_Time, np.imag(self.wavelet_fun), label='Imaginary')
        plt.xlabel('Time')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.grid(True)

        self.savePlot(plt, f"{self.wavelet_name}_timeD.jpg", save, show)

    def plotWaveletBode(self, titleStr, sRate, tFontSize, freqList, fftData, save, fileName, show):
        fig, axs = plt.subplots(2, 1, figsize=(10,10)) #w, h figsize in inches?
        fig.subplots_adjust(top = 0.95, bottom = 0.05, hspace=0.10, left = 0.10, right=0.99) 
        #plt.figure(figsize=(10, 8))
        fig.suptitle(f'{titleStr}, Data Sample Rate: {sRate}Hz', fontsize=tFontSize)
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

        self.savePlot(plt, f"{fileName}_freqD.jpg", save, show)

    def savePlot(self, plot, fileName, save, show):
        if save:
            freqDFilePathName = f"{self.fileStructure.dataDirFiles.saveDataDir.waveletDir.waveletDir_name}/{fileName}"
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