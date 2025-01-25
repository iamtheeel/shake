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

from torchinfo import summary

from timeit import default_timer as timer

from Model import multilayerPerceptron, leNetV5
from trainer import Trainer

from genPlots import saveMovieFrames

from ConfigParser import ConfigParser
config = ConfigParser(os.path.join(os.getcwd(), 'config.yaml'))
configs = config.get_config()


if configs['model']['regression']: critName = configs['trainer']['loss_regresh']
else:                             critName = configs['trainer']['criterion_class']

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

def saveSumary(outputDir, dateTime_str, model, dataShape):
    sumFile = f'{outputDir}/{dateTime_str}_modelInfo.txt'
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
#def writeLogHdr(dateTime_str, expNum):
    #dateTime_str = '{date:%Y%m%d-%H%M%S}'.format(date=datetime.datetime.now())
    #outputDir = f"{configs['outputDir']}/{dateTime_str}"
    #if expNum > 0: outputDir = f"{outputDir}/run-{expNum}"
    #if not os.path.isdir(outputDir): os.makedirs(outputDir)
    #logfile = f'{outputDir}/{dateTime_str}_log.csv'

    with open(logfile, 'w', newline='') as csvFile:
        writer = csv.writer(csvFile, dialect='unix')
        writer.writerow(['Test', configs['data']['test']])
        writer.writerow(['Data Path', configs['data']['dataPath']])
        writer.writerow(['Ch List', dataConfigs.chList])
        writer.writerow(['windowLen', configs['data']['windowLen']])
        writer.writerow(['stepSize', configs['data']['stepSize']])
        writer.writerow(['batchSize', configs['data']['batchSize']])

        writer.writerow(['dataScalers', configs['data']['dataScalers']])
        writer.writerow(['labelScalers', configs['data']['labelScalers']])
        writer.writerow(['dataScale_values', configs['data']['dataScale_values']])
        writer.writerow(['labelScale_values', configs['data']['labelScale_values']])
    
        writer.writerow(['loss', critName])
        writer.writerow(['optimizer', configs['trainer']['optimizer']])
        writer.writerow(['learning_rate', configs['trainer']['learning_rate']])
        writer.writerow(['weight_decay', configs['trainer']['weight_decay']])
        writer.writerow(['epochs', configs['trainer']['epochs']])
        writer.writerow(['seed', configs['trainer']['seed']])

        writer.writerow(['model', configs['model']['name']])

        writer.writerow(['---------'])
    #return logfile, dateTime_str, outputDir

def writeThisLogHdr(logDir, expNum, wavelet_name, wavelet_center_freq, wavelet_bandwidth, logScaleData, dataScaler, dataScale, labelScaler, labelScale, lossFunction, optimizer, learning_rate, weight_decay):
    outputDir = f"{logDir}/run-{expNum}"
    logfile = f'{outputDir}/run-{expNum}_log.csv'
    if not os.path.isdir(outputDir): os.makedirs(outputDir)

    with open(logfile, 'w', newline='') as csvFile:
        writer = csv.writer(csvFile, dialect='unix')
        writer.writerow(['wavelet', wavelet_name])
        writer.writerow(['wavelet_center_freq', wavelet_center_freq])
        writer.writerow(['wavelet_bandwidth', wavelet_bandwidth])
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
logger.info(f"INIT: Get Data")
from dataLoader import dataLoader
data_preparation = dataLoader(configs, outputDir, logfile)

if configs['data']['dataSetDir'] != "" and os.path.exists(f"{data_preparation.dataSaveDir}/data.npy"):
      data_preparation.loadDataSet()
else: data_preparation.get_data()
logger.info(f"data_preparation.data.shape: {data_preparation.data_raw.shape}")

if configs['model']['regression']: accStr = f"Acc (RMS Error)"
else                             : accStr = f"Acc (%)"

writeLogHdr(logfile, data_preparation.dataConfigs)
# Plots for each window of data
#data_preparation.plotWindowdData()
#data_preparation.plotFFTWindowdData()

# CWT Transform
from cwtTransform import cwt

# The hyperperamiters setup for expTracking
cwt_class = cwt(configs, dataConfigs = data_preparation.dataConfigs)

'''
wavelet_base = configs['cwt']['wavelet'][0]
wavelet_name = f"{wavelet_base}-{configs['cwt']['waveLet_center_freq'][0]}-{configs['cwt']['waveLet_bandwidth'][0]}"
cwt_class.setupWavelet(wavelet_name)
cwt_class.setFreqScale(freqLogScale=True)
data_preparation.resetData() #makes a fresh copy of the data and labels from _raw
#cwt_class.plotWavelet()
saveMovieFrames(data_preparation, cwt_class)
#exit()
'''


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

def getModel(wavelet_name, model_name, dataShape, config):
    #      Each model gets a cwt and a non-cwt version
    #Batch Size, inputch, height, width
    #Info for the models: x, ch, datapoints, x
    nCh = dataShape[1]
    if wavelet_base == "None":
        nDataPts = dataShape[2]
        dataShape = (configs['data']['batchSize'], 1, nCh, nDataPts)
        logger.info(f"nCh: {nCh}, nDataPts: {nDataPts}, dataShape: {dataShape}")
    else:
        nFreqs = dataShape[2]
        nTimePts = dataShape[3]
        dataShape = (configs['data']['batchSize'], nCh, nFreqs, nTimePts)
        logger.info(f"nCh: {nCh}, dataShape: {dataShape}")
    #logger.info(f"dataset size: {len(data)},  Data: {tuple(data.shape)}, labels:  {tuple(label.shape)[0]}")

    if model_name == "multilayerPerceptron":
        model = multilayerPerceptron(input_features=nCh*nDataPts, num_classes=data_preparation.nClasses, config=configs['model']['multilayerPerceptron'])
    elif model_name == "leNetV5":
        # For now use the ch as the height, and the npoints as the width
        model = leNetV5(numClasses=data_preparation.nClasses,nCh=nCh, config=configs)
    else: 
        print(f"{model_name} is not a model that we have")
        exit()
    return model

def runExp(outputDir, expNum, dateTime_str, wavelet_base, wavelet_center_freq, wavelet_bandwidth, logScaleData, dataScaler, dataScale, labelScaler, labelScale, lossFunction, optimizer, learning_rate, weight_decay, epochs):
    logfile, outputDir = writeThisLogHdr(outputDir, expNum, wavelet_base, wavelet_center_freq, wavelet_bandwidth, logScaleData, dataScaler, dataScale, labelScaler, labelScale, lossFunction, optimizer, learning_rate, weight_decay)


    # TODO: Set to save the transformed data
    if wavelet_base != "None":
        if wavelet_base != "mexh":
            wavelet_name = f"{wavelet_base}-{configs['cwt']['waveLet_center_freq'][0]}-{configs['cwt']['waveLet_bandwidth'][0]}"
        cwt_class.setupWavelet(wavelet_name)
        cwt_class.setFreqScale(freqLogScale=True)
        data_preparation.getCWTData(cwt_class) # Will load the files if they exist, otherwise will transform the data

    #Make sure we start with a fresh dataset
    data_preparation.resetData(wavelet_name=wavelet_name)
    logger.info(f"data_preparation.data.shape: {data_preparation.data.shape}")


    #logger.info(f"data: {type(data)}, labels: {type(labels)}")
    logger.info(f"Norm the data: {dataScaler}, {dataScale}")
    data_preparation.data_norm, data_preparation.dataNormConst = data_preparation.scale_data(data_preparation.data, dataScaler, logfile, dataScale)
    if configs['model']['regression']: 
        logger.info(f"Norm the labels: {labelScaler}, {labelScale}")
        data_preparation.labels_norm, data_preparation.labNormConst = data_preparation.scale_data(data_preparation.labels, labelScaler, logfile, labelScale)
        #print(f"{data_preparation.labNormConst.type}")

    if configs['plts']['saveFilesForAnimation']:
        expDir = f"run-{expNum}_{cwt_class.wavelet_name}_logScaleData-{logScaleData}_dataScaler-{dataScaler}_dataScale-{dataScale}"
        saveMovieFrames(data_preparation, cwt_class, showImageNoSave=True, expDir=expDir) 

    #TODO: Get model to function with if statement on CWT or not
    model_name = configs['model']['name']
    dataShape = data_preparation.data.shape
    model = getModel(wavelet_name, model_name, dataShape, config)

    
    if configs['debugs']['saveModelInfo']: saveSumary( outputDir, dateTime_str, model, dataShape)


    exp_StartTime = timer()
    if configs['debugs']['runModel']:
        data_preparation.createDataloaders(expNum) 

        trainer = Trainer(model=model, device=device, dataPrep=data_preparation, configs=configs, logFile=logfile, logDir=outputDir, expNum=expNum, 
                          lossFunction=lossFunction, optimizer=optimizer, learning_rate=learning_rate, weight_decay=weight_decay, epochs=epochs)
        trainLoss, trainAcc = trainer.train()
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
                         'Data Scaler': dataScaler, 
                         'Data Scale': dataScale, 
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

expNum = 1
# Wavelet
# Center Frequency
# Bandwidth
# logData
for wavelet_base in configs['cwt']['wavelet']:
    for center_freq in configs['cwt']['waveLet_center_freq']:
        for bandwidth in configs['cwt']['waveLet_bandwidth']:
            for logScaleData in [True, False]:
                for dataScaler in configs['data']['dataScalers']:
                    #TODO: Normalize the cwt data by dividing both the real and image by the magnitude
                    if dataScaler == "std": dataScale_values = [1]
                    else:                   dataScale_values = configs['data']['dataScale_values']

                    for dataScale_value in dataScale_values:

                        for labelScaler in configs['data']['labelScalers']:
                            if labelScaler == "std": 
                                labelScale_values = [1]
                            else:                   labelScale_values = configs['data']['labelScale_values']
                            for labelScale_value in labelScale_values:

                                for lossFunction in configs['trainer']['loss_regresh']:

                                    for optimizer in configs['trainer']['optimizer']:

                                        for learning_rate in configs['trainer']['learning_rate']:

                                            for weight_decay in configs['trainer']['weight_decay']:

                                                for epochs in configs['trainer']['epochs']:

                                                    logger.info(f"==============================")
                                                    logger.info(f"Wavelet: {wavelet_base}, Center Frequency: {center_freq}, Bandwidth: {bandwidth}, logData: {logScaleData}")
                                                    logger.info(f"Experiment:{expNum}, dataScaler: {dataScaler}, labelScaler: {labelScaler}, dataScale: {dataScale_value}, labelScale: {labelScale_value}")
                                                    logger.info(f"Loss: {lossFunction}, Optimizer: {optimizer}, Learning Rate: {learning_rate}, Weight Decay: {weight_decay}, Epochs: {epochs}")

                                                    runExp(outputDir=outputDir, expNum=expNum, dateTime_str=dateTime_str, 
                                                            wavelet_base=wavelet_base, wavelet_center_freq=center_freq, wavelet_bandwidth=bandwidth, logScaleData=logScaleData,
                                                            dataScaler=dataScaler, dataScale=dataScale_value, labelScaler=labelScaler, labelScale=labelScale_value, 
                                                            lossFunction=lossFunction, optimizer=optimizer, learning_rate=learning_rate, weight_decay=weight_decay, epochs=epochs)
                                                    expNum += 1