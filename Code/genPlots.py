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

from jFFT import jFFT_cl

#ICE default IO error handler doing an exit(), pid = 12090, errno = 32
#import matplotlib
#matplotlib.use('qt5agg')

import os
from ConfigParser import ConfigParser
config = ConfigParser(os.path.join(os.getcwd(), 'config.yaml'))
configs = config.get_config()

plotDir = configs['plts']['pltDir']
yLim = configs['plts']['yLim']

#Each sensors ch list:Sensor 1: ch1, Sensor8: Ch11, 12, 13
#sensorChList = [[1], [2], [3], [4], [5], [6], [7], [8, 9, 10], [11, 12, 13], [14], [15], [16], [17], [18], [19], [20] ]
#sensorList = configs['data']['sensorList'] 
chList = configs['data']['chList'] 
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