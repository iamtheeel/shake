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

from Model import multilayerPerceptron, leNetV5
from trainer import Trainer

from ConfigParser import ConfigParser
config = ConfigParser(os.path.join(os.getcwd(), 'config.yaml'))
configs = config.get_config()


if configs['model']['regression']: critName = configs['trainer']['criterion_regresh']
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

def writeLogHdr():
    dateTime_str = '{date:%Y%m%d-%H%M%S}'.format(date=datetime.datetime.now())
    outputDir = f"{configs['outputDir']}/{dateTime_str}"
    if not os.path.isdir(outputDir): os.makedirs(outputDir)
    logfile = f'{outputDir}/{dateTime_str}_log.csv'

    with open(logfile, 'w', newline='') as csvFile:
        writer = csv.writer(csvFile, dialect='unix')
        writer.writerow(['Test', configs['data']['test']])
        writer.writerow(['Data Path', configs['data']['dataPath']])
        writer.writerow(['sensorList', configs['data']['sensorList']])
        writer.writerow(['windowLen', configs['data']['windowLen']])
        writer.writerow(['stepSize', configs['data']['stepSize']])
        writer.writerow(['batchSize', configs['data']['batchSize']])
    
        writer.writerow(['criterion', critName])
        #writer.writerow(['criterion', configs['trainer']['criterion']])
        writer.writerow(['optimizer', configs['trainer']['optimizer']])
        writer.writerow(['learning_rate', configs['trainer']['learning_rate']])
        writer.writerow(['weight_decay', configs['trainer']['weight_decay']])
        writer.writerow(['epochs', configs['trainer']['epochs']])
        writer.writerow(['seed', configs['trainer']['seed']])

        writer.writerow(['model', configs['model']['name']])
        writer.writerow(['---------'])
    return logfile, dateTime_str, outputDir

torch.manual_seed(configs['trainer']['seed'])
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

logfile, dateTime_str, outputDir = writeLogHdr()

"""
Data Preparation
"""
logger.info(f"INIT: Get Data")
from dataLoader import dataLoader
data_preparation = dataLoader(configs, outputDir, logfile)

#data, labels = data_preparation.get_data()
data_preparation.get_data()
#data_raw, labels_raw = data_preparation.get_data()

#TODO: exptracking for loops start here
expTrackFile = f'{configs['outputDir']}/{dateTime_str}_dataTrack.csv'
expFieldnames = ['Test', 'Scaler', 'Train Loss', 'Train Acc(%)', 'Val Loss', 'Val Acc(%)']
with open(expTrackFile, 'w', newline='') as csvFile:
    print(f"Writing hdr: {expTrackFile}")
    writer = csv.DictWriter(csvFile, fieldnames=expFieldnames, dialect='unix')
    writer.writeheader()

firstRun = True
for scaler in configs['data']['scalers']:
    logger.info(f"==============================")
    logger.info(f"Experiment: {scaler}")

    if(firstRun): firstRun = False
    else:         logfile, dateTime_str, outputDir = writeLogHdr()

    #Make sure we start with a fresh dataset
    data_preparation.resetData()

    #logger.info(f"data: {type(data)}, labels: {type(labels)}")
    logger.info(f"Norm the data")
    data_preparation.data_norm, data_preparation.dataNormConst = data_preparation.scale_data(data_preparation.data, scaler, logfile)
    if configs['model']['regression']: 
        logger.info(f"Norm the labels")
        data_preparation.labels_norm, data_preparation.labNormConst = data_preparation.scale_data(data_preparation.labels, scaler, logfile)
        #print(f"{data_preparation.labNormConst.type}")

    
    #Info for the models
    nCh = data_preparation.data.shape[1]
    nDataPts = data_preparation.data.shape[2]
    dataShape = (configs['data']['batchSize'], 1, nCh, nDataPts)
    logger.info(f"nCh: {nCh}, nDataPts: {nDataPts}, dataShape: {dataShape}")
    #logger.info(f"dataset size: {len(data)},  Data: {tuple(data.shape)}, labels:  {tuple(label.shape)[0]}")

    model_name = configs['model']['name']
    if model_name == "multilayerPerceptron":
        model = multilayerPerceptron(input_features=nCh*nDataPts, num_classes=data_preparation.nClasses, config=configs['model']['multilayerPerceptron'])
    elif model_name == "leNetV5":
        # For now use the ch as the height, and the npoints as the width
        model = leNetV5(numClasses=data_preparation.nClasses,nCh=1, config=configs['model']['leNetV5'] )
    else: 
        print(f"{model_name} is not a model that we have")
        exit()
    
    if configs['debugs']['saveModelInfo']: saveSumary( outputDir, dateTime_str, model, dataShape)


    if configs['debugs']['runModel']:
        data_preparation.createDataloaders() 
        trainer = Trainer(model, device, data_preparation, configs, logfile, dateTime_str)
        #trainer = Trainer(model, device, train_data_loader, val_data_loader, configs, logfile, dateTime_str)
        trainLoss, trainAcc = trainer.train()
        valLoss, valAcc = trainer.validation()
        del trainer
    else:
        trainLoss, valLoss = 0, 0
        trainAcc , valAcc  = 0, 0

    with open(expTrackFile, 'a', newline='') as csvFile:
        print(f"Writing data: {expTrackFile}")
        writer = csv.DictWriter(csvFile, fieldnames=expFieldnames, dialect='unix')
        writer.writerow({'Test': dateTime_str,
                         'Scaler': scaler, 
                         'Train Loss': trainLoss, 
                         'Train Acc(%)': trainAcc, 
                         'Val Loss': valLoss, 
                         'Val Acc(%)': valAcc 
        })

    del model
    #del train_data_loader
    #del val_data_loader
    #del labels, normData
#end scailer
    