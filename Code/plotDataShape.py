###
# main.py
# Joshua Mehlman
# MIC Lab
# Spring, 2025
###
# Plot the data shapes
###

from matplotlib import pyplot, cm
import numpy as np

fig = pyplot.figure()
ax = fig.add_subplot(111, projection="3d")

dataCount = 1950 #64 # subject * runs
dataRate = 1652
dataLen = 2 # Seconds window len = 2, record len = 30
cwtFreqsCount = 128
nSensors = 9

plotNum = 2
# Raw Data
if plotNum == 1: # Raw time d
    chCount = nSensors #y
    totalRuns = np.linspace(0, dataCount, num=10, dtype=int)  
    timePoints = np.linspace(0, dataRate * dataLen - 1, num=20, dtype=int)  
    X, Z  = np.meshgrid(totalRuns, timePoints )
    yCount = chCount
elif plotNum == 2: # Data format to LeNet
    #For LeNet un-reshaped
    colorCh = 1 
    chCount = np.linspace(0,nSensors) #y
    timePoints = np.linspace(0, dataRate * dataLen - 1, num=20, dtype=int)  
    X, Z  = np.meshgrid(timePoints, chCount)
    colors = cm.rainbow(np.linspace(0, 1, colorCh+1, endpoint=True))  # Generates 'chCount' distinct colors
    yCount = colorCh
elif plotNum == 3: # Reshaped for image processing
    colorCh = nSensors 
    #imageSize = np.sqrt(dataRate * dataLen)
    width = np.linspace(0, 52) 
    height = np.linspace(0, 64)  
    X, Z  = np.meshgrid(width, height)
    yCount = colorCh
elif plotNum == 4: # Data format to Cwt
    colorCh = nSensors 
    cwtFreqs = np.linspace(0,cwtFreqsCount) #y
    timePoints = np.linspace(0, dataRate * dataLen - 1, num=20, dtype=int)  
    X, Z  = np.meshgrid(timePoints, cwtFreqs)
    yCount = colorCh

# Repeat Y for chCount slices
colors = cm.rainbow(np.linspace(0, 1, yCount+1, endpoint=True))  # Generates 'chCount' distinct colors
for i in range(yCount):
    Y = np.full_like(X, i)  # Set Y to the channel index
    ax.plot_wireframe(X, Y, Z, linewidth=0.5, color=colors[i])  # Add each channel slice 
ax.plot_wireframe(X, Y, Z)

# Set tics

# Set axis labels
if plotNum == 1:
    ax.set_xlabel("Total Runs")
    ax.set_ylabel("Channel Count")
    ax.set_zlabel("Time Points")
    ax.set_title("Time Domain Data: Windowed")
    #ax.set_title("Time Domain Data")
elif plotNum == 2:
    ax.set_xlabel("Image Width: Time Points")
    ax.set_zlabel("Image Height: Sensor Ch")
    ax.set_ylabel("Number of Color Ch: 1")
    ax.set_title("Time Domain Data: LeNet, Un-Reshaped")
    ax.set_yticks(np.arange(1, yCount)) #Every tick for ch
    ax.set_zticks(np.arange(1, nSensors+1)) #Every tick for ch
elif plotNum == 3:
    ax.set_xlabel("Width: 64")
    ax.set_zlabel("Height: 52")
    ax.set_ylabel("Num Color Ch: Data Ch")
    ax.set_title("Data ReShaped for Image Models")
    ax.set_yticks(np.arange(1, yCount+1)) #Every tick for ch
elif plotNum == 4:
    ax.set_xlabel("Width: Time Points")
    ax.set_zlabel("Height: CWT Frequencies")
    ax.set_ylabel("Num Color Ch: Data Ch")
    ax.set_title("CWT Transformed Data")
    ax.set_yticks(np.arange(1, yCount+1)) #Every tick for ch

pyplot.show()