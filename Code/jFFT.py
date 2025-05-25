####
# jFFT, wraper for running FFT using numpy.rFFT (real input only)
#
#
# THE BEER-WARE LICENSE" (Revision 42, Poul-Henning Kamp):
#   <iamtheeel> wrote this file.
#   As long as you retain this notice
#   you can do whatever you want with this stuff.
#   If we meet some day, and you think this stuff is worth it,
#   you can buy me a beer in return. - Joshua B. Mehlman
#
# Theeel December 2023
#####
version = 1.0

# Python Libs
import cmath

#Third Party
import numpy

class jFFT_cl:
    def __init__(self):
        #print("jFFT_cl.__init__: version = {}".format(version))
        self.deltaF = None
        self.fMax = None

    #def __del__(self):
        #print("jFFT_cl.__del__")

    def getFreqs(self, sRate, tBlockLen, complex = False):
        if complex:
            freqs = numpy.fft.fftfreq(tBlockLen, d=1.0/sRate)
        else:
            freqs = numpy.fft.rfftfreq(tBlockLen, d=1.0/sRate)
        self.deltaF = freqs[1]
        self.fMax =freqs[len(freqs)-1]
        #print("jFFT.getFreqs: deltaF = {}Hz, fMax = {}Hz".format(self.deltaF, self.fMax) )
        return freqs

    def calcFFT(self, timeData, complex= False, debug=False):
        # run FFT
        if complex == False:
            fftDataBlock_ri = numpy.fft.rfft(timeData)
        else: 
            fftDataBlock_ri = numpy.fft.fft(timeData)
            fftDataBlock_ri = numpy.fft.fftshift(fftDataBlock_ri) #Re-center so neg is before pos

        # for debuging
        #print("jFFT_cl.calcFFT: yDataFreq:{} = {}".format(fftDataBlock_ri.shape, fftDataBlock_ri)) 

        ##TODO## #return mag/phase or real imag
        return self.riToMF(fftDataBlock_ri)

    def riToMF(self, realImagData):
        import numpy as np
        data_mag = numpy.abs(realImagData) /np.sqrt(realImagData.size) #Parseval's theorem
        data_pha = numpy.angle(realImagData)
        data_mf = numpy.stack((data_mag, data_pha))

        #print("data mag = {} phase = {}".format(data_mag, data_pha))
        #print("data mag/phase {}{} = {} ".format(type(data_mf), data_mf.shape, data_mf))
        #print("data mag/phase {}{} ".format(type(data_mf), data_mf.shape))
        #thisVal = 1
        #print("riToMF: mag/phase mag = {} phase = {} ".format(data_mag[thisVal], data_pha[thisVal]))
        #print("riToMF: data mag/phase mag = {} phase = {} ".format(data_mf[0][thisVal], data_mf[1][thisVal]))

        return data_mf

    ## TODO ## averager

    def appWindow(self, timeData, window="None", debug=False):
        if window=="Hanning":
            window = numpy.hanning(len(timeData))
            windowedTimeData = window*timeData
        else:
            windowedTimeData = timeData.copy()

        return windowedTimeData
