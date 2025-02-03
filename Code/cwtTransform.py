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
from foot_step_wavelet import FootStepWavelet, foot_step_cwt


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

        # List of wavelets
        #wavelets = pywt.wavelist()
        #print(wavelets)
        self.wavelet_name=None

    def setupWavelet(self, wavelet_base, f0=1.0, bw=1.0, useLogForFreq=False):
        self.f0= f0
        self.bw = bw
        if wavelet_base == "mexh" or wavelet_base == "morl":
            wavelet_name = wavelet_base
        elif wavelet_base == "fstep": #No arguments, arguments handled seperately
            wavelet_name = f"{wavelet_base}-{f0}"
        else: 
            wavelet_name = f"{wavelet_base}-{f0}-{bw}"
        # The wavelet
        self.wavelet_base = wavelet_base
        self.wavelet_name = wavelet_name

        self.numScales = 240#480 #This is the height of the image

        if self.wavelet_base == 'fstep':
            self.wavelet = FootStepWavelet(central_frequency=f0)
        else:
            #Center freq and bw are imbedded in the name, or not used
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

        if self.wavelet_base != 'fstep':
            center_freq = pywt.central_frequency(self.wavelet)
        else: 
            center_freq = self.wavelet.central_frequency

        min_scale = center_freq / (max_freq * self.samplePeriod)
        max_scale = center_freq / (min_freq * self.samplePeriod)

        if freqLogScale:
            self.scales = np.logspace(np.log10(min_scale), np.log10(max_scale), self.numScales)
        else:
            self.scales = np.linspace(min_scale, max_scale, self.numScales) 

        self.frequencies = center_freq / (self.scales * self.samplePeriod) #not used
        logger.info(f"Scales: {self.scales.shape}")
        #logger.info(f"Scales: {self.scales}")

    def cwtTransform(self, data):
        # Perform continuous wavelet transform using the defined wavelet
        #logger.info(f"Transforming data: {type(data)}")

        start_time = time.time()
        if self.wavelet_base == 'fstep':
            [transformedData, data_frequencies] = foot_step_cwt(data=data, scales=self.scales, 
                                                                sampling_period=self.samplePeriod, f_0=self.f0)
        else:
            [transformedData, data_frequencies] = pywt.cwt(data, self.scales, wavelet=self.wavelet, sampling_period=self.samplePeriod)
        #logger.info(f"Frequencies: {self.data_frequencies}")
        end_time = time.time()
        logger.info(f"CWT output datashapes | transformedData: {transformedData.shape}, data_frequencies: {data_frequencies.shape}, time: {end_time - start_time}s")

        data_coefficients = transformedData

        # Keep only every nth column (time point) from the results?
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
        if normTo_max == 0: 
            normTo_min = np.min(np.abs(data_coefficients))
            normTo_max = np.max(np.abs(data_coefficients))
            # Normalize each channel's data to 0-1 range and assign to RGB channels
            logger.info(f"get3ChData | normTo_max: {normTo_max}, normTo_min: {normTo_min}")

        for i, thisCh in enumerate(plotChList):
            #Data comming in as (ch, freq, time)
            channel_data = np.abs(data_coefficients[dataChList.index(thisCh),:,:]) # Converts real/imag to mag

            norm_min = normTo_min
            norm_max = normTo_max

            normalized_data = (channel_data - norm_min) / (norm_max - norm_min)
            #normalized_data = channel_data
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

    def plotWavelet(self, expNum, sRate=0, saveDir="", show = False, save = True):
        # Get the wavelet function values
        complexInput = False
        if np.iscomplexobj(self.wavelet_fun): complexInput = True
        #if self.wavelet_name == "mexh" : complexInput = False

        # Plot the time Domain
        titleStr = f"Exp: {expNum}, {self.wavelet_base}"
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
            timeDFileNamePath = f"{saveDir}/{timeDFileName}"
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
        axs[1].plot(freqList, np.degrees(unwrapped_phase))
        axs[1].set_ylabel(f"Phase (deg)")
        axs[1].set_xlabel(f"Frequency (Hz, for Sam Rate)")
        axs[1].minorticks_on()
        axs[1].grid(True, which='minor', linestyle=':', alpha=0.2)
        axs[1].grid(True, which='major', linestyle='-', alpha=0.6)
        #axs[1].set_ylim([-180, 180])

        if save:
            timeDFileName = f"{self.wavelet_name}_freqD.jpg"
            timeDFileNamePath = f"{saveDir}/{timeDFileName}"
            logger.info(f"Saving Wavelet Plots: {timeDFileNamePath}")
            plt.savefig(timeDFileNamePath)

        if show:
            plt.show()
        plt.close()