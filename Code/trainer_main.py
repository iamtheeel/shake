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
import os
import numpy as np

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


"""
Data Preparation
"""
logger.info(f"INIT: Get Data")
from dataLoader import dataLoader
data_preparation = dataLoader(configs)

data, labels = data_preparation.get_data()

logger.info(f"data: {type(data)}, {np.shape(data)}")
logger.info(f"labels: {type(labels)}, {np.shape(labels)}")


nCh = np.shape(data)[1]
nDataPts = np.shape(data)[2]
print(f"Number Channels: {nCh}, number dataPonts:{nDataPts}")

mean = np.average(data)
max = np.max(data)
logger.info(f"Data: Mean = {mean}, Min = {np.min(data)},Max = {max}")


train_data, val_data, train_labels, val_labels = data_preparation.split_trainVal(data,labels)
logger.info(f"Train data: {type(train_data)}, {np.shape(train_data)}")
logger.info(f"Train labels: {type(train_labels)}, {np.shape(train_labels)}")
logger.info(f"Validation data: {type(val_data)}, {np.shape(val_data)}")
logger.info(f"Validation labels: {type(val_labels)}, {np.shape(val_labels)}")

model_name = configs['model']['name']

if model_name == "multilayerPerceptron":
    model = multilayerPerceptron(input_features=nCh*nDataPts, num_classes=len(data_preparation.classes), config=configs['model']['multilayerPerceptron'])
elif model_name == "leNetV5":
    #train_data = train_data[np.newaxis, :, :, :]
    model = leNetV5(numClasses=len(data_preparation.classes), config=configs['model']['leNetV5'] )
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

modelSum = summary(model=model, 
            input_size=(1, 1, nCh, nDataPts), # make sure this is "input_size", not "input_shape"
            # col_names=["input_size"], # uncomment for smaller output
            col_names=["input_size", "output_size", "num_params", "trainable"],
            col_width=20,
            row_settings=["var_names"])
        #saveInfo(model=model, thingOne=modelSum, fileName='_sumary.txt')
        #saveInfo(model=model, thingOne=model, fileName='_modelInfo.txt')
        #MACs, mPerams = countOperations(model=model, image=testImage)

trainer = Trainer(model, train_data, train_labels, val_data, val_labels, configs)
trainLoss, trainAcc = trainer.train()

testLoss, testAcc = trainer.validation() # Unit test reqires singletion 