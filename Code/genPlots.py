###
# genPlots.py
# Joshua Mehlman
# MIC Lab
# Fall, 2024
#
# Generate different plots for footfall dataset
#
###

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm  #progress bar

from jFFT import jFFT_cl
from utils import *

import typing
if typing.TYPE_CHECKING: #Fix circular import
    from dataLoader import dataLoader, normClass
    from cwtTransform import cwt

#ICE default IO error handler doing an exit(), pid = 12090, errno = 32
#import matplotlib
#matplotlib.use('qt5agg')

import os
from ConfigParser import ConfigParser
config = ConfigParser(os.path.join(os.getcwd(), 'config.yaml'))
configs = config.get_config()

## Logging
import logging
debug = configs['debugs']['debug']
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
if debug == False:
    logging.disable(level=logging.CRITICAL)
    logger.disabled = True


#plotDir = f"{configs['data']['dataOutDir']}/{configs['plts']['pltDir']}"
#yLim = configs['plts']['yLim']

#Each sensors ch list:Sensor 1: ch1, Sensor8: Ch11, 12, 13
#sensorChList = [[1], [2], [3], [4], [5], [6], [7], [8, 9, 10], [11, 12, 13], [14], [15], [16], [17], [18], [19], [20] ]
sensorChList = configs['data']['chList'] 
# Erorr on missing chList. Load from dataConfigs.chList
#chList = configs['data']['chList'] 
# x is vert
#sensorChList = [[1], [2], [3], [4], [5], [6], [7], [10], [11], [14], [15], [16], [17], [18], [19], [20] ]

def getTime(total_points, sampling_frequency):
    time = np.linspace(0, (total_points-1) / sampling_frequency, total_points) #start, stop, number of points

    return time

def plotLabels(labels):
    plt.figure(figsize=(8, 6))
    plt.title(f"Raw Labels")
    plt.plot(labels)


def plotRegreshDataSetLab(dataset, title):
    labels = []
    for _, label in dataset:
        labels.append(label.item())

    print(f"labels size: {len(labels)}")
    plt.figure(figsize=(8, 6))
    plt.title(f"DataSet: {title}")
    plt.plot(labels)
    #plt.show()



def plotRegreshDataLoader(dataLoader):
    labels = []
    for _, batchlabels in dataLoader:
        for label in batchlabels:
            labels.append(label.item())

    plt.figure(figsize=(8, 6))
    plt.title(f"Validation loader")
    plt.plot(labels)
    #plt.plot(range(len(labels)), labels)
    plt.xlabel('Validation Test #')
    plt.ylabel('Speed (m/s)')
    plt.show()

def plotOverlay(acclData, runStr, plotTitle_str, sFreq):
    #print(f"plotOver: data shape: {acclData.shape}")
    time = getTime(acclData.shape[1], sFreq)

    plt.figure(figsize=(15, 10))

    #for i in range(0,nSensors-1):
    #sens = 0
    markers = ['o', '^', 'v', 's', 'D', 'x', '+', '*']
    for sens, row in enumerate(acclData):
        thisMarker = markers[sens%len(markers)]
        #if sens == 0: thisSens = 'total'
        if sens >= len(chList): thisSens = 'all'
        else:         thisSens = chList[sens]
        plt.plot(time, row, label=f"ch {thisSens}")
        #plt.plot(time, row, label=f"ch {thisSens}", marker= thisMarker)

        #plt.plot(time, acclData[i])
        #plt.plot(time, accelerometer_data)
    plt.legend(loc="center left")
    plt.title(f"{plotTitle_str}\n")
    #plt.title(f"Accelerometer Data: Trial {trialNum+1}, Sensor {accelNum+1}")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Acceleration")
    plt.ylim(yLim)
    plt.grid(True)

    # Save the plots
    pltSaveDir = Path(f"{plotDir}/overLay")
    pltSaveDir.mkdir(parents=True, exist_ok=True)
    fileName = f"{pltSaveDir}/{runStr}_overlayed.jpg"
    print(f"Saving: {fileName}")
    plt.savefig(f"{pltSaveDir}/{runStr}_overlayed.jpg")
    #plt.show()

    plt.close

class dataPlotter_class():
    def __init__(self):
        self.fftClass = jFFT_cl()
        self.freqMax = 0
        pass

    def generalConfigs(self, samRate):
        self.samRate = samRate

    def configTimeD(self, imageDir, yLim, dataMax=0):
        self.timeDDir = imageDir
        checkFor_CreateDir(self.timeDDir, echo=False)

        self.yLim_time = yLim #configs['plts']['yLim']
        if self.yLim_time == 0: self.yLim_time = [-dataMax, dataMax]
    def configFreqD(self, yLim, dataMax=0):

        self.yLim_freq = yLim #configs['plts']['yLim']
        if self.yLim_freq == 0: self.yLim_freq = [-dataMax, dataMax]

    def plotOrShow(self, plt, fig, fileName, show=False):
        if show:
            plt.show()
        else:
            # Save the plots
            #print(f"FileName: {fileName}")
            fig.savefig(fileName)
        plt.close(fig)


    def plotInLineTime(self, acclData, saveStr, plotTitle_str, show=False):
        xData = getTime(acclData.shape[1], self.samRate)
        data = acclData

        xlabel = "Time (s)"
        fileName = f"{self.timeDDir}/{saveStr}_inLine.jpg"
        yLim = self.yLim_time
        self.plotInLine(data, plotTitle_str, xData, xlabel, yLim, fileName, show=False)

    def plotInLineFreq(self, imageDir, acclData, saveStr, plotTitle_str, xlim, yLim = None, xInLog = False, show=False):
        #logger.info(f"{imageDir}")
        xlimStr = f"fmin-{xlim[0]}_fmax-{xlim[1]}"
        self.freqDDir = f"{imageDir}_{xlimStr}"
        if xInLog: self.freqDDir = f"{self.freqDDir}_log"
        checkFor_CreateDir(self.freqDDir, echo=False)

        xData = self.fftClass.getFreqs(self.samRate, acclData.shape[1])
        data = acclData

        xlabel = "Frequency (Hz)"
        fileName = f"{self.freqDDir}/{saveStr}_{xlimStr}.jpg"
        if yLim == None: yLim = self.yLim_freq
        self.plotInLine(data, plotTitle_str, xData, xlabel, yLim, fileName, xLim=xlim, isFreq=True, xInLog=xInLog, show=False)


    def plotInLine(self, data, plotTitle_str, xData, xlabel, yLim, fileName, xLim=None, isFreq=False, xInLog=False, show=False):
        #print(f"data shape: {data.shape}")
        fig, axs = plt.subplots(data.shape[0], figsize=(12,12)) #figsize in inches?
        #Start and end the plot at x percent of the page, no space between each plot
        fig.subplots_adjust(top = 0.95, bottom = 0.05, hspace=0, left = 0.10, right=0.99) 
        fig.suptitle(plotTitle_str)
    
        thisRow = 0
        thisMax = None
        for chData in data:
            if isFreq:
                windowedData = self.fftClass.appWindow(chData, window="Hanning")
                freqData = self.fftClass.calcFFT(windowedData) #Mag, phase
                chData = freqData[0]

                thisMax = np.max(chData)
                if thisMax > self.freqMax: self.freqMax = thisMax

            #print(f"rowLen: {row.shape}, {time.shape}")
            axs[thisRow].plot(xData, chData)
            if xInLog: axs[thisRow].set_xscale('log')
            axs[thisRow].set_ylabel(f'Ch {sensorChList[thisRow]}', fontsize=8)
            #axs[thisRow].set_ylabel(f'S#{sensor}, Ch{sensorChList[sensor-1]}', fontsize=8)
    
            axs[thisRow].set_ylim(yLim)
            if xLim != None:
                axs[thisRow].set_xlim(xLim)
            axs[thisRow].get_xaxis().set_visible(False)
    
            thisRow +=1
        #Only show the x-axis on the last plot
        axs[thisRow-1].get_xaxis().set_visible(True)
        axs[thisRow-1].set_xlabel(xlabel)

        self.plotOrShow(plt, fig, fileName, show)
        #logger.info(f"thisMax: {thisMax}")
    
    
    def plotFFT(self,    data,    saveStr, plotTitle_str, show=False):
        xlim = [0, 10]
        #ch, datapoint
        nSamp = data.shape[1]
        freqList = self.fftClass.getFreqs(self.samRate, nSamp)
    
        #print(f"plotFFT: data: {data.shape}, nSamp: {nSamp}, sRate: {samRate}, subject: {subject}")
        #runStr = f"sub-{subject}_run-{runNum+1}_time-{timeStart}_{name}"
        #titleStr = f"{name} subject: {subject}, run: {runNum+1}, startTime: {timeStart}sec"
    
        plt.figure(figsize=(15, 10))
        #print(f"run {runNum}: {runData.shape}")
        for ch, chData in enumerate(data):
            windowedData = self.fftClass.appWindow(chData, window="Hanning")
            freqData = self.fftClass.calcFFT(windowedData) #Mag, phase
            #plotData = np.log10(freqData[0])
            plotData = freqData[0] #mag, phase
            #for i in range(0,3): plotData[i] = 0 #dont plot the dc
    
            plt.plot(freqList, plotData, label=f"ch {sensorChList[ch]}")
    
        plt.xlim(xlim)
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Magnitude")
        plt.legend()
        plt.title(plotTitle_str)
    
        fileName = f"{self.freqDDir}/{saveStr}_freq.jpg"
        #self.plotOrShow(plt, fig, f"{self.timeDDir}/{runStr}_inLine.jpg", show)
        print(f"Saving plot {fileName}")
        if show:
            plt.show()
        else:
            plt.savefig(fileName)
        plt.close()

def plotCombined(time, acclData, runStr, plotTitle_str):
    # Plotting the data with time on the x-axis
    fig, axs = plt.subplots(4, 4, figsize=(12,12)) #figsize in inches?
    fig.suptitle(plotTitle_str)
    fig.supxlabel(f"Time (s)")
    fig.supylabel(f"Acceleration (g?)")
    #ylim = (-maxData, maxData)
    row = 0
    sensor = 0
    axs[row, 0].plot(time, acclData[sensor])
    axs[row, 0].set_title(f'Sensor {sensor+1}')
    axs[row, 0].set_ylim(yLim)
    sensor += 1
    axs[row, 1].plot(time, acclData[sensor])
    axs[row, 1].set_title(f'Sensor {sensor+1}')
    axs[row, 1].set_ylim(yLim)
    sensor += 1
    axs[row, 2].plot(time, acclData[sensor])
    axs[row, 2].set_title(f'Sensor {sensor+1}')
    axs[row, 2].set_ylim(yLim)
    sensor += 1
    axs[row, 3].plot(time, acclData[sensor])
    axs[row, 3].set_title(f'Sensor {sensor+1}')
    axs[row, 3].set_ylim(yLim)

    row +=1
    sensor += 1
    axs[row, 0].plot(time, acclData[sensor])
    axs[row, 0].set_title(f'Sensor {sensor+1}')
    axs[row, 0].set_ylim(yLim)
    sensor += 1
    axs[row, 1].plot(time, acclData[sensor])
    axs[row, 1].set_title(f'Sensor {sensor+1}')
    axs[row, 1].set_ylim(yLim)
    sensor += 1
    axs[row, 2].plot(time, acclData[sensor])
    axs[row, 2].set_title(f'Sensor {sensor+1}')
    axs[row, 2].set_ylim(yLim)
    sensor += 1
    # Sensor 8 is x, y, z
    axs[row, 3].plot(time, acclData[sensor])
    axs[row, 3].set_title(f'Ch {sensor+1}, x,y,z')
    axs[row, 3].set_ylim(yLim)

    row +=1
    sensor += 1
    axs[row, 0].plot(time, acclData[sensor])
    axs[row, 0].set_title(f'Ch {sensor+1}, x,y,z')
    axs[row, 0].set_ylim(yLim)
    sensor += 1
    axs[row, 1].plot(time, acclData[sensor])
    axs[row, 1].set_title(f'Ch {sensor+1}, x,y,z')
    axs[row, 1].set_ylim(yLim)
            
    row +=1
    sensor += 1
    # Sensor 9 is x, y, z
    axs[row, 0].plot(time, acclData[sensor])
    axs[row, 0].set_title(f'Ch {sensor+1}, x,y,z')
    axs[row, 0].set_ylim(yLim)
    sensor += 1
    axs[row, 1].plot(time, acclData[sensor])
    axs[row, 1].set_title(f'Ch {sensor+1}, x,y,z')
    axs[row, 1].set_ylim(yLim)
    sensor += 1
    axs[row, 2].plot(time, acclData[sensor])
    axs[row, 2].set_title(f'Ch {sensor+1}, x,y,z')
    axs[row, 2].set_ylim(yLim)
            
    '''
    sensor += 1
    axs[row, 1].plot(time, acclData[sensor])
    axs[row, 1].set_title(f'Sensor {sensor+1}')
    axs[row, 1].set_ylim(yLim)
    sensor += 1
    axs[row, 2].plot(time, acclData[sensor])
    axs[row, 2].set_title(f'Sensor {sensor+1}')
    axs[row, 2].set_ylim(yLim)
    sensor += 1
    axs[row, 3].plot(time, acclData[sensor])
    axs[row, 3].set_title(f'Sensor {sensor+1}')
    axs[row, 3].set_ylim(yLim)

    row +=1
    sensor += 1
    axs[row, 0].plot(time, acclData[sensor])
    axs[row, 0].set_title(f'Sensor {sensor+1}')
    axs[row, 0].set_ylim(yLim)
    sensor += 1
    axs[row, 1].plot(time, acclData[sensor])
    axs[row, 1].set_title(f'Sensor {sensor+1}')
    axs[row, 1].set_ylim(yLim)
    sensor += 1
    axs[row, 2].plot(time, acclData[sensor])
    axs[row, 2].set_title(f'Sensor {sensor+1}')
    axs[row, 2].set_ylim(yLim)
    sensor += 1
    axs[row, 3].plot(time, acclData[sensor])
    axs[row, 3].set_title(f'Sensor {sensor+1}')
    axs[row, 3].set_ylim(yLim)
    '''
            
    # Save the plots
    pltSaveDir = Path(f"{plotDir}/combined")
    pltSaveDir.mkdir(parents=True, exist_ok=True)
    print(f"Saving plot {pltSaveDir}/{runStr}")
    fig.savefig(f"{pltSaveDir}/{runStr}_combined.jpg")
    plt.close

def plotRunFFT(data, samRate, subject, timeStart, name):
    #run, ch, datapoint
    for runNum, runData in enumerate(data):
        plotFFT(runData, samRate, subject, runNum, timeStart, name)


class saveCWT_Time_FFT_images():
    def __init__(self, data_preparation:"dataLoader", cwt_class:"cwt", expDir):
        print(f"\n")
        logger.info(f"----------     Generate Plots  ----------------")
        self.data_preparation = data_preparation
        self.showImageNoSave = configs['plts']['showFilesForAnimation']
        sensorChList = configs['data']['chList']
        self.chPlotList = configs['plts']['rgbPlotChList']
        if self.chPlotList == 0: self.chPlotList = sensorChList
        self.cwt_class = cwt_class

        # Create the save dir: Just created it so we can check during the no-save
        # Create animation directory if it doesn't exist
        #print(f"Exp Dir: {expDir}")
        self.animDir = expDir
        #self.animDir = os.path(expDir)
        #self.animDir = os.path.join(expDir, "time_fft_cwt_images")
        #os.makedirs(self.animDir, exist_ok=True)
        checkFor_CreateDir(self.animDir, echo=True)
        #logger.info(f"Saving plots in: {self.animDir}")

        self.complexInput = False
        if np.iscomplexobj(cwt_class.wavelet_fun): self.complexInput = True
        #Renorm the data
        self.normTo_max = configs['cwt']['normTo_max'] 
        self.normTo_min = configs['cwt']['normTo_min'] 
        # Find what the min and max will be after scaling
        if self.normTo_max == 0: 
            if data_preparation.dataNormConst.type == "std":
                fudge = 2 #1
            else: fudge = 4
            self.normTo_max = data_preparation.dataNormConst.max 
            if self.complexInput:
                self.normTo_max = np.max(np.abs(data_preparation.dataNormConst.max)) # We plot in mag
            # Not seting the datanormConst is somehow overwriting it?? Makes no sense
            if data_preparation.dataNormConst.type == None:
                self.normTo_max = data_preparation.dataNormConst.max
            else:
                self.normTo_max, data_preparation.dataNormConst = data_preparation.scale_data(data=self.normTo_max, norm=data_preparation.dataNormConst, debug=False) #scalers may be complex
            self.normTo_max = np.abs(self.normTo_max)
            self.normTo_max = self.normTo_max/fudge
        ### For CWT we plot the magnitude
        # For non complex tranforms this is still the abs
        # As there is a negitive component.
        # So our min to will be 0
        ###
        #if self.normTo_min == 0:
        #    self.normTo_min = data_preparation.dataNormConst.min 
        #    if self.complexInput:
        #        self.normTo_min = np.min(np.abs(data_preparation.dataNormConst.min))#*fudge
        #    self.normTo_min, data_preparation.dataNormConst = data_preparation.scale_data(data=self.normTo_min, norm=data_preparation.dataNormConst, debug=False)
        ###
        self.normTo_min = 0 
        logger.info(f"plot cwt data | normTo_max: {self.normTo_max}, normTo_min: {self.normTo_min}")
        

    def setupFigure(self):
        # Set plot configureations
        self.colorList = ['r', 'g', 'b', 'y', 'm', 'c', 'k']

        # Set up the plot axis
        fig, axs = plt.subplots(2, 2, figsize=(16,12)) #w, h figsize in inches?
        fig.subplots_adjust(top = 0.95, bottom = 0.05, hspace=0.05, left = 0.10, right=0.99) 

        # Adjust subplot sizes - make left plots smaller
        # Make right plots wider and bottom plots taller
        gs = fig.add_gridspec(2, 2, width_ratios=[1, 3], height_ratios=[1, 3])
        # Set positions for all subplots based on the gridspec
        for i in range(2):
            for j in range(2):
                axs[i,j].set_position(gs[i,j].get_position(fig))
        
        axs[0, 0].get_xaxis().set_visible(False)
        axs[0, 0].get_yaxis().set_visible(False)

        return fig, axs
    
    def setUpInfoBox(self, axs, run, timeWindow, subjectLabel):
        # Add text box with run info
        normStr = f'norm: {self.data_preparation.dataNormConst.type}'
        if self.data_preparation.dataNormConst.scale != 1:
            normStr = f"{normStr}, {self.data_preparation.dataNormConst.scale}"
        self.regClasStr = "Classification"
        if configs['model']['regression']:
            self.regClasStr = "Regression"

        textstr = f'Run: {run}\n' \
                  f'Time Window: {timeWindow}\n' \
                  f'Subject: {subjectLabel}\n' \
                  f'cwt: {self.cwt_class.wavelet_name}\n' \
                  f'{normStr}\n' \
                  f'Solution: {self.regClasStr}'
        

        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        axs[0, 0].text(0.05, 0.95, textstr, transform=axs[0, 0].transAxes, fontsize=10, verticalalignment='top', bbox=props)

        ''' Add a note: Example
        textstr = f'Note:\n' \
                  f'The stomp has very high frequency\n' \
                  f'but steps are lower\n' \
                  f'CWT Normalized to 1' 
        props = dict(boxstyle='round', facecolor='white', alpha=0.5)
        axs[0, 0].text(0.05, 0.50, textstr, transform=axs[0, 0].transAxes, fontsize=10, verticalalignment='top', bbox=props)
        '''

    def plotTimeD(self, chData, time, axs, thisColor):
        # Flip x-axis direction
        axs[0, 1].set_xlim(0, time[-1] + (time[1] - time[0]))
        axs[0, 1].set_ylim([-0.015, 0.015])
        axs[0, 1].plot(time, chData, color=thisColor) #Col, row
        axs[0, 1].set_ylabel(f'Amplitude (accl)', fontsize=8)

    def plotFreqD(self, fftData, freqList, axs, thisColor, asLogScale):
        # Plot the Frequency domain data
            if asLogScale: #Is the mag log scale?
                # Set x-axis to log scale for frequency plot
                axs[1, 0].set_xscale('log')
                axs[1, 0].set_xlim([0.01, 1])
            else:
                axs[1, 0].set_xlim([0, 0.5])
            axs[1, 0].invert_xaxis()

            # Set minimum frequency to 10 Hz
            bottom = self.cwt_class.min_freq
            top = self.cwt_class.max_freq
            if self.cwt_class.useLogScaleFreq:
                axs[1, 0].set_yscale('log')
            axs[1, 0].set_ylim(bottom=bottom, top=top)
            axs[1, 0].plot(fftData, freqList, color=thisColor)
            axs[1, 0].set_ylabel('Frequency (Hz)')

            # Add minor grid lines
            axs[1, 0].grid(True, which='minor', linestyle=':', alpha=0.2)
            axs[1, 0].grid(True, which='major', linestyle='-', alpha=0.4)

            '''
            #Set the position of the frequency plot to match the CWT plot
            shift = 0.0 # 0.033  # Amount to shift up
            pos = axs[1,0].get_position()
            new_height = pos.height - shift  # Reduce height by shift amount
            axs[1,0].set_position([pos.x0, pos.y0 + shift, pos.width, new_height])
            #textstr = f'Bottom of plot shifted up by {shift*100:.1f}% to match-ish CWT plot'
            props = dict(boxstyle='square', alpha=0.5, facecolor='white')
            axs[1, 0].text(0.05, -0.15, textstr, transform=axs[1, 0].transAxes, fontsize=10, verticalalignment='top', bbox=props)
            '''

    def getCWTData(self, cwtDataSet):
        # Get the list of ch indexes for the data we want
        indices = [sensorChList.index(ch) for ch in self.chPlotList if ch in sensorChList]

        cwtData = cwtDataSet[0].permute(1, 2, 0) #coming in as ch, freqs, timepoints
        cwtData = cwtData[:,:,indices] # Only use the data from the plot list

        # scale as we would for processing
        #logger.info(f"CWT Before scaling: min: {np.min(cwtData)}, max: {np.max(cwtData)}")
        # Not seting the datanormConst is somehow overwriting it?? Makes no sense
        #cwtData, _ = self.data_preparation.scale_data(data=cwtData, norm=self.data_preparation.dataNormConst, debug=False)
        if self.data_preparation.dataNormConst.type != None:
            if self.data_preparation.dataNormConst.type != "none":
                cwtData, self.data_preparation.dataNormConst = self.data_preparation.scale_data(data=cwtData, norm=self.data_preparation.dataNormConst, debug=False)

        cwtData = np.abs(cwtData) # non complex goes negitive

        ### Normalize for plotting
        #logger.info(f"norm min: {self.normTo_min}, max: {self.normTo_max}")
        #logger.info(f"CWT Before Plot Norm: min: {np.min(cwtData)}, max: {np.max(cwtData)}")
        cwtData = (cwtData - self.normTo_min) / (self.normTo_max - self.normTo_min)
        #logger.info(f"CWT After Plot Norm: min: {np.min(cwtData)}, max: {np.max(cwtData)}")
        
        return cwtData 
    
    def plotCWT(self, axs, cwtData):
        # We only plot the data from the sensor list4
        #indices = [sensorChList.index(ch) for ch in self.chPlotList if ch in sensorChList]
        #cwtData = cwtData[:,:, indices]

        axs[1, 1].imshow(cwtData, aspect='auto')
        #TODO: Put in function
        #logger.info(f"Freqs: {data_preparation.cwtFrequencies}")
        valid_ticks, freq_labels = self.cwt_class.getYAxis(self.cwt_class.frequencies, plt.gca().get_yticks())
        #logger.info(f"freq_labels: {freq_labels}")
        plt.gca().set_yticks(valid_ticks)
        plt.gca().set_yticklabels([f"{f:.1f}" for f in freq_labels])

        plt.xlabel('Time (s)')
        valid_ticks, time_labels = self.cwt_class.getXAxis(cwtData.shape[1], plt.gca().get_xticks())
        plt.gca().set_xticks(valid_ticks)
        plt.gca().set_xticklabels([f"{t:.1f}" for t in time_labels])

    def generateAndSaveImages(self, logScaleData):
        #dataEnd = self.data_preparation.data_raw.shape[0]
        #for dataumNumber in tqdm(range(0, dataEnd), desc="Generating CWT, FFT Plots For Movie", unit="Plot"):
        for thisWindowNum, (data_torch, label_speed, subjectLabel, subject, run, timeWindow) in  \
                  tqdm(enumerate(self.data_preparation.timeDDataSet), total= len(self.data_preparation.timeDDataSet), desc="Generateing CWT Data", unit="Window" ):

            data = data_torch.numpy()
            #data, run, timeWindow, subjectLabel = self.data_preparation.getThisWindowData(dataumNumber, ch=0) #If 0, get all channels
            # Get the fft data and labels
            time = getTime(data.shape[1], self.data_preparation.dataConfigs.sampleRate_hz)
            freqList, fftData = self.data_preparation.getFFTData(data)

            fig, axs = self.setupFigure()
            self.setUpInfoBox(axs, run, timeWindow, subjectLabel)

            #print(f"data: {data.shape}, time: {time.shape}, run: {run}, timeWindow: {timeWindow}, subjectLabel: {subjectLabel}")
            #print(f"fftData: {fftData.shape}, freqList: {freqList.shape}")
            for thisChNum, chData in enumerate(data):
                thisCh = sensorChList[thisChNum]
                if thisCh in self.chPlotList:

                    thisColor = self.colorList[self.chPlotList.index(thisCh)%len(self.colorList)]
                    #The upper left is information about the data
                    # It has the ch list and legend
                    axs[0, 0].plot(0, label=f"ch {thisCh}", color=thisColor) #Col, row

                    self.plotTimeD(chData=chData, time=time, axs=axs, thisColor=thisColor)
                    self.plotFreqD(fftData=fftData[thisChNum], freqList=freqList, axs=axs, thisColor=thisColor, asLogScale=logScaleData)
            #End Ch Data

            axs[0, 0].legend() #loc="lower center") #Display the ch list on our info window

            # Plot the CWT Data
            #cwtData, cwtFreqList = self.calcCWTData(data)
            cwt_Data = self.data_preparation.CWTDataSet.__getitem__(thisWindowNum) # data, label_speed, label_subject, subject, run, sTime
            cwtFreqList = self.cwt_class.frequencies
            cwtData = self.getCWTData(cwt_Data)
            self.plotCWT(axs, cwtData) #h, w, 3


            pltCh_Str = "_".join(map(str, self.chPlotList))
            fileName = f"{thisWindowNum:04d}_ch-{pltCh_Str}_subject-{subjectLabel}_run-{run}_timeStart-{timeWindow}.png"
            #filePath = os.path.join(self.animDir, fileName)
            filePath = f"{self.animDir}/{fileName}"
            if self.showImageNoSave:
                logger.info(f"Image File: {filePath}")
                plt.show()
            else:
                plt.savefig(filePath, dpi=250, bbox_inches=None)
            plt.close(fig)

            #procTime.endTime(echo=True, echoStr=f"Finished with dataNum: {dataumNumber}")