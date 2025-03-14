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
from utils import checkFor_CreateDir

from ConfigParser import ConfigParser
config = ConfigParser(os.path.join(os.getcwd(), 'config.yaml'))
configs = config.get_config()


from fileStructure import fileStruct
fileStructure = fileStruct()

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

def saveSumary(model, dataShape):
    sumFile = f'{fileStructure.expTrackFiles.expNumDir.expTrackDir_Name}/{model.__class__.__name__}_modelInfo.txt'
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
def writeLogHdr(dataConfigs):
    logfile = f"{fileStructure.expTrackFiles.expTrackDir_name}/{fileStructure.expTrackFiles.expTrack_sumary_file}"
    with open(logfile, 'w', newline='') as csvFile:
        writer = csv.writer(csvFile, dialect='unix')
        writer.writerow(['Test', configs['data']['test']])
        writer.writerow(['Data Path', configs['data']['inputData']])
        writer.writerow(['Ch List', dataConfigs.chList])
        writer.writerow(['windowLen', configs['data']['windowLen']])
        writer.writerow(['stepSize', configs['data']['stepSize']])
        writer.writerow(['batchSize', configs['trainer']['batchSize']])

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

def writeDataTrackHdr(cwt_class:cwt, logScaleData, dataScaler, dataScale, labelScaler, labelScale, lossFunction, optimizer, learning_rate, weight_decay):
    logfile = f'{fileStructure.expTrackFiles.expNumDir.expTrackDir_Name}/{fileStructure.expTrackFiles.expNumDir.expTrackLog_file}'

    with open(logfile, 'a', newline='') as csvFile:
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
    return logfile 

torch.manual_seed(configs['trainer']['seed'])
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

dateTime_str = '{date:%Y%m%d-%H%M%S}'.format(date=datetime.datetime.now())
fileStructure.setExpTrack_dir(dateTime_str=dateTime_str)

"""
Data Preparation
"""
from dataLoader import dataLoader
data_preparation = dataLoader(configs, fileStructure)
writeLogHdr(data_preparation.dataConfigs)

if not os.path.exists(f"{fileStructure.dataDirFiles.saveDataDir.saveDataDir_name}/{fileStructure.dataDirFiles.saveDataDir.timeDData_file}"):
    data_preparation.get_data()

if configs['model']['regression']: accStr = f"Acc (RMS Error)"
else                             : accStr = f"Acc (%)"
expTrackFile = f'{fileStructure.expTrackFiles.expTrackDir_name}/{fileStructure.expTrackFiles.expTrack_log_file}'
expFieldnames = ['Test', 'BatchSize', 'Epochs', 'Data Scaler', 'Data Scale', 'Label Scaler', 'Label Scale', 'Loss', 'Optimizer', 'Learning Rate', 'Weight Decay', 
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
            model = leNetV5_folded(numClasses=data_preparation.nClasses, dataShape=dataShape, config=configs)
        else:
            model = leNetV5_cwt(numClasses=data_preparation.nClasses,nCh=nCh, config=configs)
    elif model_name == "leNetV5_unFolded":
            model = leNetV5_timeDomain(numClasses=data_preparation.nClasses, dataShape=dataShape, config=configs)
    elif model_name == "MobileNet_v2":
        model = MobileNet_v2(numClasses=data_preparation.nClasses, dataShape=dataShape, config=configs)
    else: 
        print(f"{model_name} is not a model that we have")
        exit()
    return model


    

def runExp(expNum, logScaleData, dataScaler, dataScale, labelScaler, labelScale, 
           lossFunction, optimizer, learning_rate, weight_decay,  
           batchSize, model_name):

    epochs = configs['trainer']['epochs']
    fileStructure.setExpTrack_run(expNum=expNum)
    exp_StartTime = timer()

    writeDataTrackHdr(cwt_class, logScaleData, dataScaler, dataScale, labelScaler, labelScale, lossFunction, optimizer, learning_rate, weight_decay)


    # TODO: Set to save the transformed data

    #Log scale the lin data
    scaleStr = f"d: {data_preparation.dataNormConst.type} {data_preparation.dataNormConst.scale}, l: {labelScaler} {labelScale}"
    if(logScaleData): scaleStr = f"{scaleStr}, Log"


    logger.info(f"Get Model")
    # Add the batch size to the dataloader shape, but don't include the number of items
    if configs['cwt']['doCWT']:
        dataShape = (batchSize,) + data_preparation.CWTDataSet.shape[1:]
    else:
        dataShape = (batchSize,) + data_preparation.timeDDataSet.shape[1:]
    logger.info(f"Data Shape loaded data : {dataShape}") # Batch size, nCh, width, height

    model = getModel(cwt_class.wavelet_name, model_name, dataShape)
    if cwt_class.wavelet_name != 'None':
        if np.iscomplexobj(cwt_class.wavelet_fun):
            # This is only partialy implemented
            # and conv2d is not implemented :(
            model = model.to(torch.complex128)
            #model = model.to(torch.complex64)
    if configs['debugs']['saveModelInfo']: saveSumary(model, dataShape)

    logger.info(f"Load Trainer")
    trainer = Trainer(model=model, device=device, dataPrep=data_preparation, fileStru=fileStructure, configs=configs, expNum=expNum, 
                           cwtClass=cwt_class, scaleStr=scaleStr, lossFunction=lossFunction, optimizer=optimizer, learning_rate=learning_rate, weight_decay=weight_decay, epochs=epochs)

    trainLoss, trainAcc = trainer.train(batchSize)
    valLoss, valAcc, classAcc = trainer.validation(epochs)
    

    exp_runTime = timer() - exp_StartTime
    # Log the results
    with open(expTrackFile, 'a', newline='') as csvFile:
        print(f"Writing data: {expTrackFile}")
        writer = csv.DictWriter(csvFile, fieldnames=expFieldnames, dialect='unix')
        writer.writerow({'Test': expNum,
                         'BatchSize': batchSize,
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
    del trainer


expNum = 1
wavelet_bases = ['None']
if configs['cwt']['doCWT']: wavelet_bases = configs['cwt']['wavelet']

for batchSize in configs['trainer']['batchSize']:
    data_preparation.loadDataSet(writeLog=True, batchSize=batchSize) #Load the timed dataset even if we are doing a cwt
    # we get data freq rate here, and need it for below
    # The hyperperamiters setup for expTracking
    #if configs['cwt']['doCWT']: Put back in when we sort out norm
    cwt_class = cwt(fileStructure=fileStructure, configs=configs,  dataConfigs = data_preparation.dataConfigs)
    data_preparation.plotDataByWindow(cwt_class=cwt_class, logScaleData=False)
    for wavelet_base in wavelet_bases:
        #logger.info(f"Wavelet: {wavelet_base}")
        if wavelet_base == 'mexh':
            centerFreqs = [1]
            bandwidths = [1]
        elif wavelet_base == 'fstep': #what is the bw?
            centerFreqs = configs['cwt']['waveLet_center_freq']
            bandwidths = [1]
        elif wavelet_base == 'morl':
            centerFreqs = [0.8125]
            bandwidths = [6.0]
        elif wavelet_base == 'None':
            centerFreqs = [0] # Dummy to 0... Gotta have something to chew
            bandwidths = [0]
        else:
            centerFreqs = configs['cwt']['waveLet_center_freq']
            bandwidths = configs['cwt']['waveLet_bandwidth']
    
        for center_freq in centerFreqs:
            #logger.info(f"Center Freq: {center_freq}")
            for bandwidth in bandwidths:
                # Load the CWT Here
                logScaleFreq = configs['cwt']['logScaleFreq'] #keep as var in case we want to add to exp tracker
                # Go here even on None just to setup the name
                #if wavelet_base != 'None': Put back in when we sort norm out
                cwt_class.setupWavelet(wavelet_base=wavelet_base, sampleRate_hz=data_preparation.dataConfigs.sampleRate_hz, f0=center_freq, bw=bandwidth, useLogForFreq=logScaleFreq)
                data_preparation.generateCWTDataByWindow(cwt_class=cwt_class)
    
                for logScaleData in [False]: #Probably not interesting
    
                    for dataScaler in configs['data']['dataScalers']:
                        if dataScaler == "minMaxNorm": dataScale_values = configs['data']['dataScale_values']
                        else:                          dataScale_values = [1]
    
                        for dataScale_value in dataScale_values:
                            data_preparation.dataNormConst.type = dataScaler
                            for labelScaler in configs['data']['labelScalers']:
                                data_preparation.labNormConst.type = labelScaler
                                if labelScaler == "std": 
                                    labelScale_values = [1]
                                else:                   labelScale_values = configs['data']['labelScale_values']
                                for labelScale_value in labelScale_values:
    
                                    if configs['model']['regression']:
                                        lossFunctions = configs['trainer']['loss_regresh']
                                    else:
                                        lossFunctions = configs['trainer']['loss_class']
                                    for lossFunction in lossFunctions:
                                        for optimizer in configs['trainer']['optimizer']:
                                            for learning_rate in configs['trainer']['learning_rate']:
                                                for weight_decay in configs['trainer']['weight_decay']:
                                                    for model_name in configs['model']['name']:
        
                                                        logger.info(f"==============================")
                                                        logger.info(f"Wavelet: {wavelet_base}, Center Frequency: {center_freq}, Bandwidth: {bandwidth}, logData: {logScaleData}")
                                                        logger.info(f"Experiment:{expNum}, type: {dataScaler}, labelScaler: {labelScaler}, dataScale: {dataScale_value}, labelScale: {labelScale_value}")
                                                        logger.info(f"Loss: {lossFunction}, Optimizer: {optimizer}, Learning Rate: {learning_rate}, Weight Decay: {weight_decay}")
          
                                                        #TODO: just send the cwtClass 
                                                        if configs['debugs']['runModel']:
                                                            runExp(expNum=expNum, logScaleData=logScaleData,
                                                                   dataScaler=dataScaler, dataScale=dataScale_value, labelScaler=labelScaler, labelScale=labelScale_value, 
                                                                   lossFunction=lossFunction, optimizer=optimizer, learning_rate=learning_rate, weight_decay=weight_decay,  
                                                                   batchSize = batchSize, model_name=model_name)
         
                                                        expNum += 1