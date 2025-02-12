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
from utils import timeTaken

from cwtTransform import cwt
import typing
if typing.TYPE_CHECKING: #Fix circular import
    from dataLoader import dataLoader, normClass

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

plotDir = configs['plts']['pltDir']
yLim = configs['plts']['yLim']

#Each sensors ch list:Sensor 1: ch1, Sensor8: Ch11, 12, 13
#sensorChList = [[1], [2], [3], [4], [5], [6], [7], [8, 9, 10], [11, 12, 13], [14], [15], [16], [17], [18], [19], [20] ]
#sensorList = configs['data']['sensorList'] 
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

def plotInLine(acclData, runStr, plotTitle_str, sFreq, show=False):
    time = getTime(acclData.shape[1], sFreq)

    fig, axs = plt.subplots(acclData.shape[0], figsize=(12,12)) #figsize in inches?
    #Start and end the plot at x percent of the page, no space between each plot
    fig.subplots_adjust(top = 0.95, bottom = 0.05, hspace=0, left = 0.10, right=0.99) 
    fig.suptitle(plotTitle_str)

    thisRow = 0
    for chData in acclData:
        #print(f"rowLen: {row.shape}, {time.shape}")
        axs[thisRow].plot(time, chData)
        axs[thisRow].set_ylabel(f'Ch {chList[thisRow]}', fontsize=8)
        #axs[thisRow].set_ylabel(f'S#{sensor}, Ch{sensorChList[sensor-1]}', fontsize=8)

        axs[thisRow].set_ylim(yLim)
        #axs[thisRow].set_xlim([20, 21])
        axs[thisRow].get_xaxis().set_visible(False)

        thisRow +=1
    #Only show the x-axis on the last plot
    axs[thisRow-1].get_xaxis().set_visible(True)
    axs[thisRow-1].set_xlabel("Time (s)")

    # Save the plots
    pltSaveDir = Path(f"{plotDir}/inLine")
    pltSaveDir.mkdir(parents=True, exist_ok=True)
    fileName = f"{pltSaveDir}/{runStr}_inLine.jpg"
    print(f"FileName: {fileName}")

    if show:
        plt.show()
    else:
        fig.savefig(fileName)

    plt.close

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

def plotFFT(data, samRate, subject, runNum, timeStart, name, show=False):
    xlim = [0, 65]
    fftClass = jFFT_cl()
    #ch, datapoint
    nSamp = data.shape[1]
    freqList = fftClass.getFreqs(samRate,nSamp)

    pltSaveDir = Path(f"{plotDir}/freq_df-{fftClass.deltaF:.3f}_fMax-{fftClass.fMax}_plotXMax-{xlim[1]}hz")
    pltSaveDir.mkdir(parents=True, exist_ok=True)
    print(f"plotFFT: data: {data.shape}, nSamp: {nSamp}, sRate: {samRate}, subject: {subject}")

    runStr = f"sub-{subject}_run-{runNum+1}_time-{timeStart}_{name}"
    titleStr = f"{name} subject: {subject}, run: {runNum+1}, startTime: {timeStart}sec"

    plt.figure(figsize=(15, 10))
    #print(f"run {runNum}: {runData.shape}")
    for ch, chData in enumerate(data):
        windowedData = fftClass.appWindow(chData, window="Hanning")
        freqData = fftClass.calcFFT(windowedData) #Mag, phase
        #plotData = np.log10(freqData[0])
        plotData = freqData[0]
        #plotData[0] = 0 #dont plot the dc
        #plotData[1] = 0 #dont plot the dc
        #plotData[2] = 0 #dont plot the dc
        #plotData[3] = 0 #dont plot the dc

        plt.plot(freqList, plotData, label=f"ch {chList[ch]}")

    plt.xlim(xlim)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.legend()
    plt.title(titleStr)

    fileName = f"{pltSaveDir}/{runStr}_freq.jpg"
    print(f"Saving plot {fileName}")
    if show:
        plt.show()
    else:
        plt.savefig(fileName)

    plt.close()

class saveCWT_Time_FFT_images():
    def __init__(self, data_preparation:"dataLoader", cwt_class:"cwt", expDir):
        logger.info(f"----------     Generate Plots  ----------------")
        self.data_preparation = data_preparation
        self.showImageNoSave = configs['plts']['showFilesForAnimation']
        self.sensorList = configs['data']['chList']
        self.chPlotList = configs['plts']['rgbPlotChList']
        if self.chPlotList == 0: self.chPlotList = self.sensorList
        self.cwt_class = cwt_class

        # Create the save dir: Just created it so we can check during the no-save
        # Create animation directory if it doesn't exist
        self.animDir = os.path.join(configs['plts']['animDir'], expDir)
        os.makedirs(self.animDir, exist_ok=True)
        logger.info(f"Saving plots in: {self.animDir}")

        self.complexInput = False
        if np.iscomplexobj(cwt_class.wavelet_fun): self.complexInput = True
        #Renorm the data
        self.normTo_max = configs['cwt']['normTo_max'] 
        self.normTo_min = configs['cwt']['normTo_min'] 
        # Find what the min and max will be after scaling
        if self.normTo_max == 0: 
            fudge = 4
            self.normTo_max = data_preparation.dataNormConst.max 
            if self.complexInput:
                self.normTo_max = np.max(np.abs(data_preparation.dataNormConst.max)) # We plot in mag
            self.normTo_max, _ = data_preparation.scale_data(data=self.normTo_max, norm=data_preparation.dataNormConst, debug=False)
            self.normTo_max = self.normTo_max/fudge
        if self.normTo_min == 0:
            self.normTo_min = data_preparation.dataNormConst.min 
            if self.complexInput:
                self.normTo_min = np.min(np.abs(data_preparation.dataNormConst.min))#*fudge
            self.normTo_min, _ = data_preparation.scale_data(data=self.normTo_min, norm=data_preparation.dataNormConst, debug=False)
        self.normTo_min = 0 #TODO: Look at this
        logger.info(f"plot cwt data | normTo_max: {self.normTo_max}, normTo_min: {self.normTo_min}")

        
        # Set plot configureations
        self.colorList = ['r', 'g', 'b', 'y', 'm', 'c', 'k']

    def setupFigure(self):
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

    def calcCWTData(self, timeDData):
        # Get the list of ch indexes for the data we want
        indices = [self.sensorList.index(ch) for ch in self.chPlotList if ch in self.sensorList]

        #logger.info(f"Data shape: {timeDData.shape}") #ch, timepoints

        # Filter foo along the channel dimension
        cwtData, cwtFrequencies = self.cwt_class.cwtTransform(timeDData[indices, :])
        #height, ch, width (240, 3, 3304)
        cwtData = np.transpose(cwtData, (0, 2, 1))
        #if self.complexInput:
        cwtData = np.abs(cwtData)
        #logger.info(f"CWT Data shape: {cwtData.shape}") 

        # scale as we would for processing
        cwtData, _ = self.data_preparation.scale_data(data=cwtData, norm=self.data_preparation.dataNormConst, debug=False)

        #Normalize for plotting
        #self.normTo_max = abs(self.data_preparation.dataNormConst.max )
        #self.normTo_min = 0 #abs(self.data_preparation.dataNormConst.min )
        #logger.info(f"CWT Before Norm: min: {np.min(cwtData)}, max: {np.max(cwtData)}")
        cwtData = (cwtData - self.normTo_min) / (self.normTo_max - self.normTo_min)
        #logger.info(f"CWT After Norm: min: {np.min(cwtData)}, max: {np.max(cwtData)}")
        
        return cwtData, cwtFrequencies

    def generateAndSaveImages(self, logScaleData):
        #procTime = timeTaken(2) 
        dataEnd = self.data_preparation.data_raw.shape[0]
        #procTime.endTime(echo=True, echoStr=f"Preliminary Done, datasize: {dataEnd}")

        for dataumNumber in tqdm(range(0, dataEnd), desc="Generating CWT, FFT Data For Movie", unit="Plot"):
            #procTime.startTime()
            #The time domain data and labels
            #Data is ch, timepoint
            data, run, timeWindow, subjectLabel = self.data_preparation.getThisWindowData(dataumNumber, ch=0) #If 0, get all channels
            # Get the fft data and labels
            time = getTime(data.shape[1], self.data_preparation.dataConfigs.sampleRate_hz)
            freqList, fftData = self.data_preparation.getFFTData(data)

            fig, axs = self.setupFigure()
            self.setUpInfoBox(axs, run, timeWindow, subjectLabel)

            #print(f"data: {data.shape}, time: {time.shape}, run: {run}, timeWindow: {timeWindow}, subjectLabel: {subjectLabel}")
            #print(f"fftData: {fftData.shape}, freqList: {freqList.shape}")
            for i, chData in enumerate(data):
                thisCh = self.sensorList[i]
                if thisCh in self.chPlotList:

                    thisColor = self.colorList[self.chPlotList.index(thisCh)%len(self.colorList)]
                    #The upper left is information about the data
                    # It has the ch list and legend
                    axs[0, 0].plot(0, label=f"ch {thisCh}", color=thisColor) #Col, row

                    self.plotTimeD(chData=chData, time=time, axs=axs, thisColor=thisColor)
                    self.plotFreqD(fftData=fftData[i], freqList=freqList, axs=axs, thisColor=thisColor, asLogScale=logScaleData)
            #End Ch Data

            axs[0, 0].legend() #loc="lower center") #Display the ch list on our info window

            # Plot the CWT Data
            cwtData, cwtFreqList = self.calcCWTData(data)
            axs[1, 1].imshow(cwtData, aspect='auto')


            pltCh_Str = "_".join(map(str, self.chPlotList))
            fileName = f"{dataumNumber:04d}_ch-{pltCh_Str}_subject-{subjectLabel}_run-{run}_timeStart-{timeWindow}.png"
            filePath = os.path.join(self.animDir, fileName)
            if self.showImageNoSave:
                logger.info(f"Image File: {filePath}")
                plt.show()
            else:
                plt.savefig(filePath, dpi=250, bbox_inches=None)

            #don't forget to add the plt ch when we save
            #and regresh or not
            #procTime.endTime(echo=True, echoStr=f"Finished with dataNum: {dataumNumber}")

    




def saveMovieFrames(data_preparation:"dataLoader", cwt_class:"cwt", asLogScale, showImageNoSave, expDir):
    procTime = timeTaken(2) 
    logger.info(f"saveMovieFrames: data: {data_preparation.data.shape} ")
    colorList = ['r', 'g', 'b', 'y', 'm', 'c', 'k']

    procTime.endTime(echo=True, echoStr=f"Finished color list")
    if not showImageNoSave:
        # Create animation directory if it doesn't exist
        animDir = os.path.join(configs['plts']['animDir'], expDir)
        os.makedirs(animDir, exist_ok=True)
    procTime.endTime(echo=True, echoStr=f"Finished Dir setup")

    #For the cwt data
    normTo_max = configs['cwt']['normTo_max'] 
    normTo_min = configs['cwt']['normTo_min'] 
    procTime.endTime(echo=True, echoStr=f"Finished min/max config read")
    if normTo_max == 0: 
        fudge = 4
        normTo_max = np.max(np.abs(data_preparation.data))/fudge
        procTime.endTime(echo=True, echoStr=f"finished max, shape={data_preparation.data.shape}")
    if normTo_min == 0:
        normTo_min = np.min(np.abs(data_preparation.data))#*fudge
        procTime.endTime(echo=True, echoStr=f"finished min, shape={data_preparation.data.shape}")
    logger.info(f"plot cwt data | normTo_max: {normTo_max}, normTo_min: {normTo_min}")
    procTime.endTime(echo=True, echoStr=f"Finished Get Norm data")

    dataEnd = data_preparation.data.shape[0]
    procTime.endTime(echo=True, echoStr=f"Preliminary Done")
    for dataumNumber in range(0, dataEnd):
        procTime.startTime()

        #This will get moved out of the loop when we animate
        fig, axs = plt.subplots(2, 2, figsize=(16,12)) #w, h figsize in inches?
        fig.subplots_adjust(top = 0.95, bottom = 0.05, hspace=0.05, left = 0.10, right=0.99) 

        # Adjust subplot sizes - make left plots smaller
        # Make right plots wider and bottom plots taller
        gs = fig.add_gridspec(2, 2, width_ratios=[1, 3], height_ratios=[1, 3])
        # Set positions for all subplots based on the gridspec
        for i in range(2):
            for j in range(2):
                axs[i,j].set_position(gs[i,j].get_position(fig))
        

        #The time domain data and labels
        data, run, timeWindow, subjectLabel = data_preparation.getThisWindowData(dataumNumber, ch=0) #If 0, get all channels
        #Data is ch, timepoint
        time = getTime(data.shape[1], data_preparation.dataConfigs.sampleRate_hz)
        #print(f"data: {data.shape}, time: {time.shape}, run: {run}, timeWindow: {timeWindow}, subjectLabel: {subjectLabel}")
        freqList, fftData = data_preparation.getFFTData(data)
        #print(f"fftData: {fftData.shape}, freqList: {freqList.shape}")

        procTime.endTime(echo=True, echoStr=f"Done getting data and lables")
        for i, chData in enumerate(data):
            if configs['cwt']['rgbPlotChList'] == 0:
                chList = data_preparation.dataConfigs.chList
            else:
                chList = configs['cwt']['rgbPlotChList']
            thisColor = colorList[i%len(colorList)]
            #The upper left is information about the data
            axs[0, 0].plot(0, label=f"ch {chList[i]}", color=thisColor) #Col, row
            axs[0, 0].get_xaxis().set_visible(False)
            axs[0, 0].get_yaxis().set_visible(False)
            # Add text box with run info
            textstr = f'Run: {run}\n' \
                      f'Time Window: {timeWindow}\n' \
                      f'Subject: {subjectLabel}\n' \
                      f'cwt: {cwt_class.wavelet_name}\n' \
                      f'norm: {data_preparation.dataNormConst.type}, {data_preparation.dataNormConst.scale}'
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            axs[0, 0].text(0.05, 0.95, textstr, transform=axs[0, 0].transAxes, fontsize=10, verticalalignment='top', bbox=props)

            '''
            textstr = f'Note:\n' \
                      f'The stomp has very high frequency\n' \
                      f'but steps are lower\n' \
                      f'CWT Normalized to 1' 
            props = dict(boxstyle='round', facecolor='white', alpha=0.5)
            axs[0, 0].text(0.05, 0.50, textstr, transform=axs[0, 0].transAxes, fontsize=10, verticalalignment='top', bbox=props)
            '''

            #Plot the time domain data
            # Flip x-axis direction
            axs[0, 1].set_xlim(0, time[-1] + (time[1] - time[0]))
            axs[0, 1].set_ylim([-0.015, 0.015])
            axs[0, 1].plot(time, chData, color=thisColor) #Col, row
            axs[0, 1].set_ylabel(f'Amplitude (accl)', fontsize=8)


            # Plot the Frequency domain data
            if asLogScale: #Is the mag log scale?
                # Set x-axis to log scale for frequency plot
                axs[1, 0].set_xscale('log')
                axs[1, 0].set_xlim([0.01, 1])
            else:
                axs[1, 0].set_xlim([0, 0.5])
            axs[1, 0].invert_xaxis()

            # Set minimum frequency to 10 Hz
            bottom = cwt_class.min_freq
            top = cwt_class.max_freq
            if cwt_class.useLogScaleFreq:
                axs[1, 0].set_yscale('log')
            axs[1, 0].set_ylim(bottom=bottom, top=top)
            axs[1, 0].plot(fftData[i], freqList, color=thisColor)
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

            #End Ch
        procTime.endTime(echo=True, echoStr=f"Done with each ch")

        axs[0, 0].legend(loc="lower center") #Display the ch list on our info window

        # Plot the wavelet transformed data
        # Data is cwt: time window number, ch, freq, time
        #logger.info(f"cwtData: {type(data_preparation.data)}, {data_preparation.data.shape}, {type(data_preparation.data[0,0,0,0])}")
        rgb_data = cwt_class.get3ChData(data_preparation.data[dataumNumber, :, :], chList, data_preparation.dataConfigs.chList, normTo_max, normTo_min)
        #rgb_data = cwt_class.get3ChData(chList, data_preparation.data[:, dataumNumber, :], data_preparation.dataConfigs.chList, normTo_max, normTo_min)
        #logger.info(f"rgb_data: {type(rgb_data)}, {rgb_data.shape}")
        #rgb_data is: Numpy Array (Height, width, ch)
        axs[1, 1].imshow(rgb_data, aspect='auto')

        #logger.info(f"Freqs: {data_preparation.cwtFrequencies}")
        valid_ticks, freq_labels = cwt_class.getYAxis(data_preparation.cwtFrequencies, plt.gca().get_yticks())
        #logger.info(f"freq_labels: {freq_labels}")
        plt.gca().set_yticks(valid_ticks)
        plt.gca().set_yticklabels([f"{f:.1f}" for f in freq_labels])

        plt.xlabel('Time (s)')
        valid_ticks, time_labels = cwt_class.getXAxis(data_preparation.data[0], plt.gca().get_xticks())
        plt.gca().set_xticks(valid_ticks)
        plt.gca().set_xticklabels([f"{t:.1f}" for t in time_labels])

        # Uncomment to show the plot
        #cwt_class.plotCWTransformed_data_3CH(data, cwtFrequencies, run, timeWindow, subjectLabel, configs['cwt']['rgbPlotChList'], data_preparation.dataConfigs.chList, logScale=True, save=False, display=True)
        # Save animation frames
        procTime.endTime(echo=True, echoStr=f"Done plotting cwt")
        if not showImageNoSave:
            # Save this frame and add to list
            fileName = f"{dataumNumber:04d}_subject-{subjectLabel}_run-{run}_timeStart-{timeWindow}.png"
            filePath = os.path.join(animDir, fileName)
            plt.savefig(filePath, dpi=250, bbox_inches=None)
            print(f"Saved image file: {filePath}, {procTime.endTime(echo=True, echoStr=f"FileSaveTime")}")
        else:
            plt.show()

        procTime.endTime(echo=True, echoStr=f"Finished with dataNum: {dataumNumber}")
        #End Data