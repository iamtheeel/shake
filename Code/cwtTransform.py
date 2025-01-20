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
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class cwt:
    def __init__(self, configs, wavelet_name, wavelet_params, dataConfigs):
        #Data information
        self.samplePeriod = 1/dataConfigs.sampleRate_hz

        # The wavelet
        self.wavelet_name = wavelet_name
        self.wavelet_params = wavelet_params

        self.numScales = 480 #This is the height of the image

        self.wavelet = pywt.ContinuousWavelet(self.wavelet_name)
        self.levels = 8 
        self.length = 256
        # Use a smaller length for wavefun to avoid memory error

        # Our wave function
        self.wavelet_fun, self.wavelet_Time = self.wavelet.wavefun(length=self.length, level=self.levels) # Default level is typically sufficient
        #self.wavelet_Time *= 0.25
        logger.info(f"Wavelet time from: {self.wavelet_Time[0]} to {self.wavelet_Time[-1]}")

    def cwtTransform(self, data):
        # Perform continuous wavelet transform using the defined wavelet
        #scales = np.arange(1, self.numScales)  # N+1 number of scales (frequencies)
        #Todo: chang the min/max
        scales = np.logspace(0, np.log10(self.numScales), self.numScales)
        #scales = np.linspace(1, 826, self.numScales)
        #scales = np.logspace(0, 826, self.numScales)

        [self.data_coefficients, self.data_frequencies] = pywt.cwt(data, scales, wavelet=self.wavelet, sampling_period=self.samplePeriod)
        #logger.info(f"Frequencies: {self.data_frequencies.shape}")
        logger.info(f"Frequencies: {self.data_frequencies}")
        logger.info(f"Coefficients: type: {type(self.data_coefficients[0][0])}, shape: {self.data_coefficients.shape}") #each dataum is a numpy.complex128
        
        # Plot the CWT coefficients
    def plotCWTransformed_data(self, run, timeWindow, subject, ch):
        plt.figure(figsize=(12, 8))
        colorMap = 'gray'
        #colorMap = 'jet'
        plt.imshow(np.abs(self.data_coefficients), aspect='auto', cmap=colorMap)
        plt.colorbar(label='Magnitude')

        #plt.ylabel('Scale')
        plt.ylabel('Frequency (Hz)')
        # Get y-axis ticks and convert frequencies to Hz
        yticks = plt.gca().get_yticks()
        freq_labels = self.data_frequencies[yticks.astype(int)[yticks < len(self.data_frequencies)]]
        plt.gca().set_yticklabels([f"{f:.1f}" for f in freq_labels])

        plt.xlabel('Time')
        # Scale x-axis by sample rate to show time in seconds
        xticks = plt.gca().get_xticks()
        time_labels = xticks * self.samplePeriod
        plt.gca().set_xticklabels([f"{t:.1f}" for t in time_labels])
        plt.title(f'CWT using {self.wavelet_name} wavelet (scales={self.numScales})\n'
                  f'run: {run}, timeWindow: {timeWindow}, subject: {subject}, channel: {ch}')
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
