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

#ICE default IO error handler doing an exit(), pid = 12090, errno = 32
#import matplotlib
#matplotlib.use('qt5agg')

plotDir = "../plots"

#yLim = [-0.03, 0.03]
yLim = [-0.001, 0.001]

#Each sensors ch list:Sensor 1: ch1, Sensor8: Ch11, 12, 13
sensorChList = [[1], [2], [3], [4], [5], [6], [7], [8, 9, 10], [11, 12, 13], [14], [15], [16], [17], [18], [19], [20] ]
# x is vert
#sensorChList = [[1], [2], [3], [4], [5], [6], [7], [10], [11], [14], [15], [16], [17], [18], [19], [20] ]

def plotOverlay(nSensors, time, acclData, runStr, plotTitle_str):
    plt.figure(figsize=(10, 4))
            
    for i in range(0,nSensors-1):
        plt.plot(time, acclData[i])
        #plt.plot(time, accelerometer_data)
        plt.title(f"{plotTitle_str}\nSensors 1-20 ")
        #plt.title(f"Accelerometer Data: Trial {trialNum+1}, Sensor {accelNum+1}")
        plt.xlabel("Time (seconds)")
        plt.ylabel("Acceleration")
        plt.ylim(yLim)
        #plt.xlim(6.6,7)
        plt.grid(True)



        # Save the plots
        pltSaveDir = Path(f"{plotDir}/overLay")
        pltSaveDir.mkdir(parents=True, exist_ok=True)
        plt.savefig(f"{pltSaveDir}/{runStr}_overlayed.jpg")

        #plt.show()
        plt.close

def plotInLine(time, acclData, testNum, runStr, plotTitle_str):
    if(testNum == 2):
        #sensorList = [8, 7, 6, 5, 4, 3, 2, 12, 1]
        sensorList = [7, 6, 5, 4, 3, 2, 12, 1] # the tri-ax does not look good
    elif (testNum == 3):
        #sensorList = [6, 5, 4, 3, 2, 1, 10, 9, 11, 12, 13]
        sensorList = [6, 5, 4, 3, 2, 1, 10, 11, 12, 13] # the tri-ax does not look good

    fig, axs = plt.subplots(len(sensorList), figsize=(12,12)) #figsize in inches?
    #Start and end the plot at x percent of the page, no space between each plot
    fig.subplots_adjust(top = 0.95, bottom = 0.05, hspace=0, left = 0.10, right=0.99) 
    fig.suptitle(plotTitle_str)

    thisRow = 0
    for sensor in sensorList:
        for ch in sensorChList[sensor-1]:
            #add all the ch for this sensor
            axs[thisRow].plot(time, acclData[ch-1])
        axs[thisRow].set_ylabel(f'S#{sensor}, Ch{sensorChList[sensor-1]}', fontsize=8)
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

    #plt.show()
    fig.savefig(f"{pltSaveDir}/{runStr}_inLine.jpg")
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