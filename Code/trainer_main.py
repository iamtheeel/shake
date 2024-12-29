###
# trainer_main.py
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

# Write a log
dateTime_str = '{date:%Y%m%d-%H%M%S}'.format(date=datetime.datetime.now())
logfile = f'{configs['outputDir']}/{dateTime_str}_log.csv'

with open(logfile, 'w', newline='') as csvFile:
    writer = csv.writer(csvFile, dialect='unix')
    writer.writerow(['Test', configs['data']['test']])
    writer.writerow(['Data Path', configs['data']['dataPath']])
    writer.writerow(['sensorList', configs['data']['sensorList']])
    writer.writerow(['windowLen', configs['data']['windowLen']])
    writer.writerow(['stepSize', configs['data']['stepSize']])
    writer.writerow(['batchSize', configs['data']['batchSize']])

    writer.writerow(['criterion', configs['trainer']['criterion']])
    writer.writerow(['optimizer', configs['trainer']['optimizer']])
    writer.writerow(['learning_rate', configs['trainer']['learning_rate']])
    writer.writerow(['weight_decay', configs['trainer']['weight_decay']])
    writer.writerow(['epochs', configs['trainer']['epochs']])
    writer.writerow(['seed', configs['trainer']['seed']])

    writer.writerow(['model', configs['model']['name']])
    writer.writerow(['---------'])


"""
Data Preparation
"""
logger.info(f"INIT: Get Data")
from dataLoader import dataLoader
data_preparation = dataLoader(configs, logfile)

#data, labels = data_preparation.get_data()
train_data_loader, val_data_loader = data_preparation.get_data()

#Info for the models
data_Batch, label_batch = next(iter(train_data_loader))
data, label = data_Batch[0], label_batch[0]
dataShape =tuple(data_Batch.shape) 
nCh = dataShape[2]
nDataPts = dataShape[3]
#logger.info(f"dataset size:  train: {len(train_data_loader)}, val: {len(val_data_loader)}, batch: {dataShape}:  Data: {tuple(data.shape)}, labels:  {tuple(label.shape)[0]}")

model_name = configs['model']['name']
if model_name == "multilayerPerceptron":
    model = multilayerPerceptron(input_features=nCh*nDataPts, num_classes=data_preparation.nClasses, config=configs['model']['multilayerPerceptron'])
elif model_name == "leNetV5":
    #train_data = train_data[np.newaxis, :, :, :]
    # For now use the ch as the height, and the npoints as the width
    model = leNetV5(numClasses=data_preparation.nClasses,nCh=1, config=configs['model']['leNetV5'] )
else: 
    print(f"{model_name} is not a model that we have")
    exit()
'''
elif model_name == "MobileNetV3":
    model = MobileNetV3(num_classes=len(data_preparation.classes))
elif model_name == "AlexNet":
    model = AlexNet(num_classes=len(data_preparation.classes))
elif model_name == "MobileNetV1":
    model = mobileNetV1(input_shape=image_depth, output_shape=len(data_preparation.classes))
'''

sumFile = f'{configs['outputDir']}/{dateTime_str}_modelInfo.txt'
with open(sumFile, 'w', newline='') as sumFile:
    sys.stdout = sumFile
    modelSum = summary(model=model, 
            #Batch Size, inputch, height, width
            input_size=dataShape, # make sure this is "input_size", not "input_shape"
            #input_size=(1, 1, nCh, nDataPts), # make sure this is "input_size", not "input_shape"
            # col_names=["input_size"], # uncomment for smaller output
            col_names=["input_size", "output_size", "num_params", "trainable"],
            col_width=20,
            row_settings=["var_names"])
        #saveInfo(model=model, thingOne=model, fileName='_modelInfo.txt')
        #MACs, mPerams = countOperations(model=model, image=testImage)
    sys.stdout = sys.__stdout__

trainer = Trainer(model, device, train_data_loader, val_data_loader, configs, logfile, dateTime_str)

trainer.train()
trainer.validation()