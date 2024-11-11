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

plotDir = "plots"

yLim = [-0.025, 0.025]

#Each sensors ch list:Sensor 1: ch1, Sensor8: Ch11, 12, 13
sensorChList = [[1], [2], [3], [4], [5], [6], [7], [8, 9, 10], [11, 12, 13], [14], [15], [16], [17], [18], [19], [20] ]

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
        plt.savefig(f"{plotDir}/{runStr}_overlayed.png")

        #plt.show()

def plotInLine(time, acclData, runStr, plotTitle_str):
    sensorList = [8, 7, 6, 5, 4, 3, 2, 12, 1]

    fig, axs = plt.subplots(len(sensorList), figsize=(12,12)) #figsize in inches?
    #Start and end the plot at x percent of the page, no space between each plot
    fig.subplots_adjust(top = 0.95, bottom = 0.05, hspace=0) 
    fig.suptitle(plotTitle_str)

    thisRow = 0
    for sensor in sensorList:
        for ch in sensorChList[sensor-1]:
            #add all the ch for this sensor
            axs[thisRow].plot(time, acclData[ch-1])
        axs[thisRow].set_ylabel(f'Sensor {sensor}:{sensorChList[sensor-1]}')
        axs[thisRow].set_ylim(yLim)
        axs[thisRow].get_xaxis().set_visible(False)

        thisRow +=1
    #Only show the x-axis on the last plot
    axs[thisRow-1].get_xaxis().set_visible(True)
    axs[thisRow-1].set_xlabel("Time (s)")

    # Save the plots
    pltSaveDir = Path(f"{plotDir}/inLine")
    pltSaveDir.mkdir(parents=True, exist_ok=True)

    fig.savefig(f"{pltSaveDir}/{runStr}_combined.png")
    #plt.show()

def plotCombined(time, acclData, runStr, plotTitle_str):
    # Plotting the data with time on the x-axis
    fig, axs = plt.subplots(4, 4, figsize=(12,12)) #figsize in inches?
    fig.suptitle(plotTitle_str)
    fig.supxlabel(f"Time (s)")
    fig.supylabel(f"Acceleration (g?)")
    #ylim = (-maxData, maxData)
    ylim = (-0.02, 0.02)
    row = 0
    sensor = 0
    axs[row, 0].plot(time, acclData[sensor])
    axs[row, 0].set_title(f'Sensor {sensor+1}')
    axs[row, 0].set_ylim(ylim)
    sensor += 1
    axs[row, 1].plot(time, acclData[sensor])
    axs[row, 1].set_title(f'Sensor {sensor+1}')
    axs[row, 1].set_ylim(ylim)
    sensor += 1
    axs[row, 2].plot(time, acclData[sensor])
    axs[row, 2].set_title(f'Sensor {sensor+1}')
    axs[row, 2].set_ylim(ylim)
    sensor += 1
    axs[row, 3].plot(time, acclData[sensor])
    axs[row, 3].set_title(f'Sensor {sensor+1}')
    axs[row, 3].set_ylim(ylim)

    row +=1
    sensor += 1
    axs[row, 0].plot(time, acclData[sensor])
    axs[row, 0].set_title(f'Sensor {sensor+1}')
    axs[row, 0].set_ylim(ylim)
    sensor += 1
    axs[row, 1].plot(time, acclData[sensor])
    axs[row, 1].set_title(f'Sensor {sensor+1}')
    axs[row, 1].set_ylim(ylim)
    sensor += 1
    axs[row, 2].plot(time, acclData[sensor])
    axs[row, 2].set_title(f'Sensor {sensor+1}')
    axs[row, 2].set_ylim(ylim)
    sensor += 1
    # Sensor 8 is x, y, z
    axs[row, 3].plot(time, acclData[sensor])
    axs[row, 3].set_title(f'Sensor {sensor+1}, x,y,z')
    axs[row, 3].set_ylim(ylim)
    sensor += 1
    axs[row, 3].plot(time, acclData[sensor])
    sensor += 1
    axs[row, 3].plot(time, acclData[sensor])
            
    row +=1
    sensor += 1
    # Sensor 9 is x, y, z
    axs[row, 0].plot(time, acclData[sensor])
    axs[row, 0].set_title(f'Sensor {sensor+1}, x,y,z')
    axs[row, 0].set_ylim(ylim)
    sensor += 1
    axs[row, 0].plot(time, acclData[sensor])
    sensor += 1
    axs[row, 0].plot(time, acclData[sensor])
            
    sensor += 1
    axs[row, 1].plot(time, acclData[sensor])
    axs[row, 1].set_title(f'Sensor {sensor+1}')
    axs[row, 1].set_ylim(ylim)
    sensor += 1
    axs[row, 2].plot(time, acclData[sensor])
    axs[row, 2].set_title(f'Sensor {sensor+1}')
    axs[row, 2].set_ylim(ylim)
    sensor += 1
    axs[row, 3].plot(time, acclData[sensor])
    axs[row, 3].set_title(f'Sensor {sensor+1}')
    axs[row, 3].set_ylim(ylim)

    row +=1
    sensor += 1
    axs[row, 0].plot(time, acclData[sensor])
    axs[row, 0].set_title(f'Sensor {sensor+1}')
    axs[row, 0].set_ylim(ylim)
    sensor += 1
    axs[row, 1].plot(time, acclData[sensor])
    axs[row, 1].set_title(f'Sensor {sensor+1}')
    axs[row, 1].set_ylim(ylim)
    sensor += 1
    axs[row, 2].plot(time, acclData[sensor])
    axs[row, 2].set_title(f'Sensor {sensor+1}')
    axs[row, 2].set_ylim(ylim)
    sensor += 1
    axs[row, 3].plot(time, acclData[sensor])
    axs[row, 3].set_title(f'Sensor {sensor+1}')
    axs[row, 3].set_ylim(ylim)
            
    print(f"Saving plot {runStr}")
    fig.savefig(f"{plotDir}/{runStr}_combined.png")