###
# main.py
# Joshua Mehlman
# MIC Lab
# Fall, 2024
###
# Train footfall data
###

# From MICLab
## Configuration
import os, sys
import datetime
import csv
import numpy as np #conda install numpy=1.26.4

#import time
from utils import timeTaken

from torchinfo import summary

from timeit import default_timer as timer

from Model import *
from trainer import Trainer

# CWT Transform
from cwtTransform import cwt

#from genPlots import saveMovieFrames

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

## What platform are we running on
import platform
machine = platform.machine()
logger.info(f"machine: {machine}")
if machine == "aarch64":
    device = "tpu"
else:
    import torch
    device = "cpu"
    if torch.cuda.is_available(): device = "cuda"
    if torch.backends.mps.is_available() and torch.backends.mps.is_built(): device = "mps"
logger.info(f"device: {device}")

def saveSumary(outputDir, dateTime_str, model, dataShape):
    sumFile = f'{outputDir}/{dateTime_str}_modelInfo.txt'
    logger.info(f"Save modelinfo | fileName: {sumFile} | dataShape: {type(dataShape)}, {dataShape}")
    with open(sumFile, 'w', newline='') as sumFile:
        sys.stdout = sumFile
        modelSum = summary(model=model, 
            #Batch Size, inputch, height, width
            input_size=dataShape, # make sure this is "input_size", not "input_shape"
            col_names=["input_size", "output_size", "num_params", "trainable"],
            col_width=20,
            row_settings=["var_names"])
        sys.stdout = sys.__stdout__

# Write a log
def getLogFileNames(dateTime_str, expNum):
    outputDir = f"{configs['outputDir']}/{dateTime_str}"
    if expNum > 0: outputDir = f"{outputDir}/run-{expNum}"
    if not os.path.isdir(outputDir): os.makedirs(outputDir)
    logfile = f'{outputDir}/{dateTime_str}_log.csv'
    return logfile, outputDir

def writeLogHdr(logfile, dataConfigs):
    with open(logfile, 'w', newline='') as csvFile:
        writer = csv.writer(csvFile, dialect='unix')
        writer.writerow(['Test', configs['data']['test']])
        writer.writerow(['Data Path', configs['data']['dataPath']])
        writer.writerow(['Ch List', dataConfigs.chList])
        writer.writerow(['windowLen', configs['data']['windowLen']])
        writer.writerow(['stepSize', configs['data']['stepSize']])
        writer.writerow(['batchSize', configs['data']['batchSize']])

        writer.writerow(['wavelets', configs['cwt']['wavelet']])
        writer.writerow(['centerFreqs', configs['cwt']['waveLet_center_freq']])
        writer.writerow(['bandwidths', configs['cwt']['waveLet_bandwidth']])

        writer.writerow(['dataScalers', configs['data']['dataScalers']])
        writer.writerow(['labelScalers', configs['data']['labelScalers']])
        writer.writerow(['dataScale_values', configs['data']['dataScale_values']])
        writer.writerow(['labelScale_values', configs['data']['labelScale_values']])
    
        writer.writerow(['Regression loss', configs['trainer']['loss_regresh']])
        writer.writerow(['Clasification loss', configs['trainer']['loss_class']])
        writer.writerow(['optimizer', configs['trainer']['optimizer']])
        writer.writerow(['learning_rate', configs['trainer']['learning_rate']])
        writer.writerow(['weight_decay', configs['trainer']['weight_decay']])
        writer.writerow(['epochs', configs['trainer']['epochs']])
        writer.writerow(['seed', configs['trainer']['seed']])

        writer.writerow(['model', configs['model']['name']])

        writer.writerow(['---------'])

def writeThisLogHdr(logDir, expNum, cwt_class:cwt, logScaleData, dataScaler, dataScale, labelScaler, labelScale, lossFunction, optimizer, learning_rate, weight_decay):
    outputDir = f"{logDir}/run-{expNum}"
    logfile = f'{outputDir}/run-{expNum}_log.csv'
    if not os.path.isdir(outputDir): os.makedirs(outputDir)

    with open(logfile, 'w', newline='') as csvFile:
        writer = csv.writer(csvFile, dialect='unix')
        writer.writerow(['wavelet', cwt_class.wavelet_base])
        writer.writerow(['wavelet_center_freq', cwt_class.f0])
        writer.writerow(['wavelet_bandwidth', cwt_class.bw])
        writer.writerow(['logScaleData', logScaleData])
        writer.writerow(['dataScaler', dataScaler])
        writer.writerow(['dataScale', dataScale])
        writer.writerow(['labelScaler', labelScaler])
        writer.writerow(['labelScale', labelScale])
        writer.writerow(['loss', lossFunction])
        writer.writerow(['optimizer', optimizer])
        writer.writerow(['learning_rate', learning_rate])
        writer.writerow(['weight_decay', weight_decay])
        writer.writerow(['---------'])
    return logfile, outputDir

torch.manual_seed(configs['trainer']['seed'])
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

dateTime_str = '{date:%Y%m%d-%H%M%S}'.format(date=datetime.datetime.now())
logfile, outputDir = getLogFileNames(dateTime_str, expNum=0)

"""
Data Preparation
"""
logger.info(f"INIT: --------------------  Get Data   ----------------------")
from dataLoader import dataLoader
data_preparation = dataLoader(configs, outputDir, logfile)

if configs['data']['dataSetDir'] != "" and os.path.exists(f"{data_preparation.dataSaveDir}/data.npy"):
      data_preparation.loadDataSet()
else: data_preparation.get_data()
logger.info(f"Time domain data shape: {data_preparation.data_raw.shape}")

if configs['model']['regression']: accStr = f"Acc (RMS Error)"
else                             : accStr = f"Acc (%)"

writeLogHdr(logfile, data_preparation.dataConfigs)
# Plots for each window of data
#data_preparation.plotWindowdData()
#data_preparation.plotFFTWindowdData()


# The hyperperamiters setup for expTracking
logger.info(f"Get cwt peramiters")
cwt_class = cwt(configs, dataConfigs = data_preparation.dataConfigs)



#Get the data for the wavelet transform
#For the stomp triggered data: 15 sec in for subject 1, run 1 is the 3rd dataum
'''
thisTimeWindow = 15 #3 # Subjects and runs are in here
thisChannel = 0 #Ch 0 is all channels
cwt_class.trackWavelet(data_preparation, thisTimeWindow, thisChannel)
'''


expTrackFile = f'{outputDir}/{dateTime_str}_dataTrack.csv'
expFieldnames = ['Test', 'Epochs', 'Data Scaler', 'Data Scale', 'Label Scaler', 'Label Scale', 'Loss', 'Optimizer', 'Learning Rate', 'Weight Decay', 
                 'Train Loss', f'Train {accStr}', 'Val Loss', f'Val {accStr}', f'Class Acc {accStr}', 'Time(s)']
with open(expTrackFile, 'w', newline='') as csvFile:
    print(f"Writing hdr: {expTrackFile}")
    writer = csv.DictWriter(csvFile, fieldnames=expFieldnames, dialect='unix')
    writer.writeheader()

def getModel(wavelet_name, model_name, dataShape):
    logger.info(f"Loading model: {model_name}")
    #      Each model gets a cwt and a non-cwt version
    #Batch Size, inputch, height, width
    #Info for the models: x, ch, datapoints, x
    nCh = dataShape[1]

    if model_name == "multilayerPerceptron":
        nDataPts = dataShape[2]
        model = multilayerPerceptron(input_features=nCh*nDataPts, num_classes=data_preparation.nClasses, config=configs['model']['multilayerPerceptron'])
    elif model_name == "leNetV5":
        # For now use the ch as the height, and the npoints as the width
        if wavelet_name == "None":
            #TODO: rewrite lenet to take timeD as an argument
            model = leNetV5_timeDomain(numClasses=data_preparation.nClasses,nCh=nCh, config=configs)
        else:
            model = leNetV5_cwt(numClasses=data_preparation.nClasses,nCh=nCh, config=configs)
    elif model_name == "MobileNet_v2":
        model = MobileNet_v2(numClasses=data_preparation.nClasses, nCh=nCh, config=configs)
    else: 
        print(f"{model_name} is not a model that we have")
        exit()
    return model


    
"""
Data Preparation
"""
logger.info(f"INIT: Get Data")
from dataLoader import dataLoader
data_preparation = dataLoader(configs, outputDir, logfile)

if configs['data']['dataSetDir'] != "" and os.path.exists(f"{data_preparation.dataSaveDir}/data.npy"):
      data_preparation.loadDataSet()
else: data_preparation.get_data()
logger.info(f"Time domain data shape: {data_preparation.data_raw.shape}")


#def runExp(outputDir, expNum, dateTime_str, wavelet_base, wavelet_center_freq, wavelet_bandwidth, logScaleData, dataScaler, dataScale, labelScaler, labelScale, lossFunction, optimizer, learning_rate, weight_decay, epochs):
def runExp(outputDir, expNum, dateTime_str, logScaleData, dataScaler, dataScale, labelScaler, labelScale, lossFunction, optimizer, learning_rate, weight_decay, epochs):
    logfile, outputDir = writeThisLogHdr(outputDir, expNum, wavelet_base, cwt_class, data_preparation, logScaleData, dataScaler, dataScale, labelScaler, labelScale, lossFunction, optimizer, learning_rate, weight_decay)
    dataAsCWT = True
    if cwt_class.wavelet_base == "None": dataAsCWT = False


    # TODO: Set to save the transformed data
    if not dataAsCWT:
        cwt_class.plotWavelet(saveDir=outputDir, expNum=expNum, sRate=data_preparation.dataConfigs.sampleRate_hz, save=True, show=False )

    #Make sure we start with a fresh time d dataset
    data_preparation.resetData() 
    #logger.info(f"After preprocessing data shape: {data_preparation.data.shape}")

    #Log scale the lin data
    scaleStr = f"d: {data_preparation.dataNormConst.type} {data_preparation.dataNormConst.scale}, l: {labelScaler} {labelScale}"
    if(logScaleData): scaleStr = f"{scaleStr}, Log"

    if configs['model']['regression']: 
        logger.info(f"Norm the labels: {labelScaler}, {labelScale}")
        data_preparation.labels_norm, data_preparation.labNormConst = data_preparation.scale_data(data=data_preparation.labels, logFile=logfile, scaler=labelScaler , scale=labelScale, debug=False)
        #print(f"{data_preparation.labNormConst.type}")
    else: 
        data_preparation.labels_norm = data_preparation.labels

    # Save movie goes to cwtTransform data
    #if cwt_class.wavelet_base != "None" and configs['plts']['saveFilesForAnimation']:
    #    expDir = f"exp-{expNum}_{cwt_class.wavelet_name}_logScaleData-{logScaleData}_dataScaler-{dataScaler}_dataScale-{dataScale}/images"

    #    #timeStart = time.time()
    #    saveMovieTime = timeTaken()
    #    saveMovieFrames(data_preparation, cwt_class, asLogScale=logScaleData, showImageNoSave=configs['plts']['showFilesForAnimation'], expDir=expDir) 
    #    saveMovieTime.endTime(echo=True, echoStr=f"Save Movie Frames")
    #    #timeTaken = timeStart - time.time()
    #    #logger.info(f"Movie Frames saved: {timeTaken:.2f}s")


    #TODO: Move to experTrack
    logger.info(f"Get Model")
    model_name = configs['model']['name']
    # Data is currently: datapoints, height(sensorch), width(datapoints)
    print(f"Data Shape of Time D raw: {data_preparation.data.shape}")
    batchSize = configs['data']['batchSize'] 
    runs = data_preparation.data.shape[0]
    timePts = data_preparation.data.shape[2]
    if dataAsCWT:
        nCh = data_preparation.data.shape[1]
        height = cwt_class.numScales

    else:
        # Note, for timeD, height = data ch
        nCh = 1
        height = data_preparation.data.shape[1]
        # We want: datapoints, image channels, height, width 
        data_preparation.data = np.expand_dims(data_preparation.data, axis=1)  # Equivalent to unsqueeze(1)
    dataShape = (batchSize, nCh, height, timePts) #Batch, Ch, Freqs, TimePts
    #dataShape = (batchSize, runs, nCh, height, timePts) #Batch, Runs, Ch, Freqs, TimePts
    print(f"DataShape Now: {dataShape}")

    model = getModel(cwt_class.wavelet_name, model_name, dataShape)
    if np.iscomplexobj(cwt_class.wavelet_fun):
        # This is only partialy implemented
        # and conv2d is not implemented :(
        model = model.to(torch.complex128)
        #model = model.to(torch.complex64)
    if configs['debugs']['saveModelInfo']: 
        saveSumary( outputDir, dateTime_str, model, dataShape)


    exp_StartTime = timer()
    if configs['debugs']['runModel']:
        logger.info(f"Create Dataloaders")
        data_preparation.createDataloaders(expNum) 

        logger.info(f"Load Trainer")
        # Data scaling info?
        trainer = Trainer(model=model, device=device, dataPrep=data_preparation, configs=configs, logFile=logfile, logDir=outputDir, expNum=expNum, 
                           cwtClass=cwt_class, scaleStr=scaleStr, lossFunction=lossFunction, optimizer=optimizer, learning_rate=learning_rate, weight_decay=weight_decay, epochs=epochs)
        logger.info(f"Train")
        trainLoss, trainAcc = trainer.train()
        logger.info(f"Run Validation")
        valLoss, valAcc, classAcc = trainer.validation()
        del trainer
    else:
        trainLoss, valLoss = 0, 0
        trainAcc , valAcc  = 0, 0
        classAcc = 0

    exp_runTime = timer() - exp_StartTime

    with open(expTrackFile, 'a', newline='') as csvFile:
        print(f"Writing data: {expTrackFile}")
        writer = csv.DictWriter(csvFile, fieldnames=expFieldnames, dialect='unix')
        writer.writerow({'Test': expNum,
                         'Epochs': epochs,
                         'Data Scaler': data_preparation.dataNormConst.type, 
                         'Data Scale': data_preparation.dataNormConst.scale, 
                         'Label Scaler': labelScaler, 
                         'Label Scale': labelScale, 
                         'Loss': lossFunction,
                         'Optimizer': optimizer,
                         'Learning Rate': learning_rate,
                         'Weight Decay': weight_decay,
                         'Train Loss': trainLoss, 
                         f'Train {accStr}': trainAcc, 
                         'Val Loss': valLoss, 
                         f'Val {accStr}': valAcc,
                         f'Class Acc {accStr}': classAcc,
                         'Time(s)': exp_runTime
        })

    del model


# Experiments:
# Wavelet
# Center Frequency
# Bandwidth
# logData: logscale the data or not
# Loss function
# Optimiser
# Learning rate
# Weight decay
# Number of epochs
# TODO:
# Sliding window size
# Sliding window overlap
# Model
# Model Peramiters: 
expNum = 1
for wavelet_base in configs['cwt']['wavelet']:
    centerFreqs = configs['cwt']['waveLet_center_freq']
    bandwidths = configs['cwt']['waveLet_bandwidth']
    if wavelet_base == 'mexh':
        centerFreqs = [1]
        bandwidths = [1]
    if wavelet_base == 'fstep': #what is the bw?
        bandwidths = [1]
    if wavelet_base == 'morl':
        centerFreqs = [0.8125]
        bandwidths = [6.0]

    if configs['model']['regression']:
        lossFunctions = configs['trainer']['loss_regresh']
    else:
        lossFunctions = configs['trainer']['loss_class']

    for center_freq in centerFreqs:
        for bandwidth in bandwidths:
            # Load the CWT Here
            logScaleFreq = configs['cwt']['logScaleFreq'] #keep as var in case we want to add to exp tracker
            cwt_class.setupWavelet(wavelet_base=wavelet_base, f0=center_freq, bw=bandwidth, useLogForFreq=logScaleFreq)

            for logScaleData in [False]: #Probably not interesting

                for dataScaler in configs['data']['dataScalers']:
                    #TODO: Normalize the cwt data by dividing both the real and image by the magnitude
                    if dataScaler == "minMaxNorm": dataScale_values = configs['data']['dataScale_values']
                    else:                          dataScale_values = [1]

                    for dataScale_value in dataScale_values:
                        #Load the norm perams, or calculate if the file is not there
                        data_preparation.getNormPerams(cwt_class=cwt_class, logScaleData=logScaleData)
                        data_preparation.dataNormConst.type = dataScaler
                        data_preparation.dataNormConst.scale = dataScale_value

                        if configs['plts']['generatePlots']: #We can't plot unless we have the norm perams
                            data_preparation.plotDataSet(cwt_class=cwt_class, logScaleData=logScaleData)
                            exit()

                        for labelScaler in configs['data']['labelScalers']:
                            if labelScaler == "std": 
                                labelScale_values = [1]
                            else:                   labelScale_values = configs['data']['labelScale_values']
                            for labelScale_value in labelScale_values:

                                for lossFunction in lossFunctions:

                                    for optimizer in configs['trainer']['optimizer']:

                                        for learning_rate in configs['trainer']['learning_rate']:

                                            for weight_decay in configs['trainer']['weight_decay']:

                                                for epochs in configs['trainer']['epochs']:

                                                    logger.info(f"==============================")
                                                    logger.info(f"Wavelet: {wavelet_base}, Center Frequency: {center_freq}, Bandwidth: {bandwidth}, logData: {logScaleData}")
                                                    logger.info(f"Experiment:{expNum}, dataScaler: {dataScaler}, labelScaler: {labelScaler}, dataScale: {dataScale_value}, labelScale: {labelScale_value}")
                                                    logger.info(f"Loss: {lossFunction}, Optimizer: {optimizer}, Learning Rate: {learning_rate}, Weight Decay: {weight_decay}, Epochs: {epochs}")

                                                    #TODO: just send the cwtClass 
                                                    runExp(outputDir=outputDir, expNum=expNum, dateTime_str=dateTime_str, logScaleData=logScaleData,
                                                            dataScaler=dataScaler, dataScale=dataScale_value, labelScaler=labelScaler, labelScale=labelScale_value, 
                                                            lossFunction=lossFunction, optimizer=optimizer, learning_rate=learning_rate, weight_decay=weight_decay, epochs=epochs)

                                                    expNum += 1