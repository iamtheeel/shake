###
# main.py
# Joshua Mehlman
# MIC Lab
# Fall, 2024
###
# Train footfall data
###

## Installs
# #conda install numpy=1.26.4  #But copmplex needs > 2
# pip install torch torchvision
# pip install torch torchinfo
# pip install matplotlib
# pip install pywavelets #pywt
# pip install scipy
# pip install PyYAML
# pip install h5py
# pip install scikit-learn
# pip install seaborn

# For complex numbers
# pip install deprecated

# From MICLab
## Configuration
import os, sys
import datetime
import csv
import numpy as np #conda install numpy=1.26.4  #But copmplex needs > 2

#import time
from utils import timeTaken

from torchinfo import summary  #install torch torchinfo

from timeit import default_timer as timer

from Model import *
from trainer import Trainer

# CWT Transform
from cwtTransform import cwt

#from genPlots import saveMovieFrames
from utils import checkFor_CreateDir

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--local_rank', type=int, default=0)
args = parser.parse_args()

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
logger.info(f"Importing torch")
import torch
if torch.cuda.is_available(): 
    device = "cuda"
    torch.cuda.set_device(args.local_rank)
    print(f"[GPU {args.local_rank}] CUDA: {torch.cuda.get_device_name(args.local_rank)} available = {torch.cuda.is_available()}")
if torch.backends.mps.is_available() and torch.backends.mps.is_built(): device = "mps"
device = "cpu" # Force CPU for complex on the MAC
logger.info(f"device: {device}")

def saveSumary(model, dataShape, complex=False):
    sumFile = f'{fileStructure.expTrackFiles.expNumDir.expTrackDir_Name}/{model.__class__.__name__}_modelInfo.txt'
    logger.info(f"Save modelinfo | fileName: {sumFile} | dataShape: {type(dataShape)}, {dataShape}")

    if complex:
        dummyInput = torch.randn(dataShape, dtype=torch.complex64)
    else:
        dummyInput = torch.randn(dataShape)
    logger.info(f"dummyInput: {type(dummyInput)}, {dummyInput.shape}, {dummyInput.dtype}")

    with open(sumFile, 'w', newline='') as sumFile:
        sys.stdout = sumFile
        modelSum = summary(model=model, 
            #Batch Size, inputch, height, width
            #input_size=dataShape, # make sure this is "input_size", not "input_shape"
            input_data=dummyInput,
            col_names=["input_size", "output_size", "num_params", "trainable"],
            col_width=20,
            row_settings=["var_names"])
        sys.stdout = sys.__stdout__

# Write a log
def writeDataTrackSum_hdr(dataConfigs):
    dataTrackSum_fileName = f"{fileStructure.expTrackFiles.expTrackDir_name}/{fileStructure.expTrackFiles.expTrack_sumary_file}"
    # Write from config.yaml
    with open(dataTrackSum_fileName, 'w', newline='') as csvFile:
        writer = csv.writer(csvFile, dialect='unix')
        writer.writerow([f'--------- from config.yaml ----------'])
        writer.writerow([f'', '--------- data '])

        writer.writerow(['Test', configs['data']['test']])
        writer.writerow(['Data Path', configs['data']['inputData']])
        writer.writerow(['Ch List', dataConfigs.chList])
        writer.writerow(['Classes', configs['data']['classes']])

        writer.writerow(['windowLen', configs['data']['windowLen']])
        writer.writerow(['stepSize', configs['data']['stepSize']])

        writer.writerow(['stompThresh', configs['data']['stompThresh'], "If not file, use:", configs['data']['stompSens']])
        writer.writerow(['dataThresh', configs['data']['dataThresh']])

        writer.writerow(['dataScalers', configs['data']['dataScalers'], configs['data']['dataScale_values']])
        writer.writerow(['labelScalers', configs['data']['labelScalers'], configs['data']['labelScale_values']])

        writer.writerow(['limitRuns', configs['data']['limitRuns']])
        writer.writerow(['limitWindowLen', configs['data']['limitWindowLen']])

        writer.writerow(['Down Sample Factor', configs['data']['downSample']])
    

        writer.writerow([f'', '--------- CWT '])
        #writer.writerow(['Do CWT', configs['cwt']['doCWT']])
        writer.writerow(['log scale freq', configs['cwt']['logScaleFreq']])
        writer.writerow(['log scale data', configs['cwt']['logScale']])
        writer.writerow(['Num Scales', configs['cwt']['numScales']])
        writer.writerow(['Freq range', configs['cwt']['fMin'], configs['cwt']['fMax']])

        writer.writerow(['wavelets/center freq/Bandwidth', configs['cwt']['wavelet'], configs['cwt']['waveLet_center_freq'], configs['cwt']['waveLet_bandwidth_freq']])
        writer.writerow(['Norm to min/max', configs['cwt']['normTo_min'], configs['cwt']['normTo_max'] ])


        writer.writerow([f'', '--------- trainer '])
        writer.writerow(['batchSize', configs['trainer']['batchSize']])
        writer.writerow(['Regression loss', configs['trainer']['loss_regresh']])
        writer.writerow(['Clasification loss', configs['trainer']['loss_class']])
        writer.writerow(['optimizer', configs['trainer']['optimizer']])
        writer.writerow(['learning_rate', configs['trainer']['learning_rate']])
        writer.writerow(['weight_decay', configs['trainer']['weight_decay']])
        writer.writerow(['Learning rate sch: T_0/T-mult/eta_min', configs['trainer']['LR_sch'], configs['trainer']['T_0'], configs['trainer']['T-mult'], configs['trainer']['eta_min']])
        writer.writerow(['gradiant_noise', configs['trainer']['gradiant_noise']])
        writer.writerow(['epochs', configs['trainer']['epochs']])
        writer.writerow(['seed', configs['trainer']['seed']])

        writer.writerow([f'', '--------- model '])
        writer.writerow(['model', configs['model']['name']])
        writer.writerow(['dropout Layers', configs['model']['dropOut']])

        writer.writerow(['--------- end configs -----------'])
    #The rest of the file is written in dataLoader

def writeExpSum(cwt_class:cwt, 
                logScaleData, dataScaler, dataScale, labelScaler, labelScale, 
                modelName, dropoutLayers, batchSize, lossFunction, optimizer, learning_rate, weight_decay, gradiant_noise):
    expTrackSum_fileName = f'{fileStructure.expTrackFiles.expNumDir.expTrackDir_Name}/{fileStructure.expTrackFiles.expNumDir.expTrackSum_fileName}'

    with open(expTrackSum_fileName, 'a', newline='') as csvFile:
        writer = csv.writer(csvFile, dialect='unix')
        writer.writerow(['---- wavelet peramiters -----'])
        writer.writerow(['wavelet', cwt_class.wavelet_base])
        writer.writerow(['wavelet_center_freq', cwt_class.f0])
        writer.writerow(['wavelet_bandwidth', cwt_class.bw])
        writer.writerow(['---- data peramiters -----'])
        writer.writerow(['logScaleData', logScaleData])
        writer.writerow(['dataScaler', dataScaler])
        writer.writerow(['dataScale', dataScale])
        writer.writerow(['labelScaler', labelScaler])
        writer.writerow(['labelScale', labelScale])
        writer.writerow(['---- training peramiters -----'])
        writer.writerow(['modelName', modelName])
        writer.writerow(['dropoutLayers', dropoutLayers])
        writer.writerow(['batchSize', batchSize])
        writer.writerow(['loss', lossFunction])
        writer.writerow(['optimizer', optimizer])
        writer.writerow(['learning_rate', learning_rate])
        writer.writerow(['weight_decay', weight_decay])
        writer.writerow(['gradiant_noise', gradiant_noise])
    #return logfile 

torch.manual_seed(configs['trainer']['seed'])
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

dateTime_str = '{date:%Y%m%d-%H%M%S}'.format(date=datetime.datetime.now())
fileStructure.setExpTrack_dir(dateTime_str=dateTime_str)

"""
Data Preparation
"""
from dataLoader import dataLoader
data_preparation = dataLoader(configs, fileStructure, device)
writeDataTrackSum_hdr(data_preparation.dataConfigs)

if not os.path.exists(f"{fileStructure.dataDirFiles.saveDataDir.saveDataDir_name}/{fileStructure.dataDirFiles.saveDataDir.timeDData_file}"):
    data_preparation.get_data()

if configs['model']['regression']: accStr = f"Acc (RMS Error)"
else                             : accStr = f"Acc (%)"
expTrackFile = f'{fileStructure.expTrackFiles.expTrackDir_name}/{fileStructure.expTrackFiles.expTrack_log_file}'
lastStats_n = configs['trainer']['nEpochsStats']
expFieldNames = ['Test', 'BatchSize', 'Epochs', 'wavelet', 'Data Scaler', 'Data Scale', 'Label Scaler', 'Label Scale', 'Loss', 'Optimizer', 'Learning Rate', 'Weight Decay', 'Gradiant Noise',
                 'Model', 'Dropout Layers',
                 'Train Loss', f'Train {accStr}', 'Last Epoch Val Loss', 
                 f'Last Epoch Val {accStr}', f'Last {lastStats_n} epochs min',
                 f'Last {lastStats_n} epochs max', f'Last {lastStats_n} epochs mean', f'Last {lastStats_n} epochs std',
                 f'Class Acc {accStr}', 'Time(s)']
with open(expTrackFile, 'w', newline='') as csvFile:
    print(f"Writing hdr: {expTrackFile}")
    writer = csv.DictWriter(csvFile, fieldnames=expFieldNames, dialect='unix')
    writer.writeheader()

def getModel(wavelet_name, model_name, dataShape, dropOut_layers = None, timeD= False, complex= False):
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
            model = leNet(numClasses=data_preparation.nClasses,nCh=nCh, config=configs, complex=complex)
            #model = leNetV5_cwt(numClasses=data_preparation.nClasses,nCh=nCh, config=configs)
    #elif model_name == "leNetV5_unFolded":
    #        model = leNetV5_timeDomain(numClasses=data_preparation.nClasses, dataShape=dataShape, config=configs)
    elif model_name == "MobileNet_v2":
        model = MobileNet_v2(numClasses=data_preparation.nClasses, dataShape=dataShape, folded=False, dropOut=dropOut_layers , config=configs, timeD=timeD)
    elif model_name == "MobileNet_v2_folded":
        model = MobileNet_v2(numClasses=data_preparation.nClasses, dataShape=dataShape, dropOut=dropOut_layers, config=configs)
    else: 
        print(f"{model_name} is not a model that we have")
        exit()

    '''
    from torchviz import make_dot
    x = torch.randn(dataShape)
    y = model(x)
    make_dot(y, params=dict(model.named_parameters())).render("modelFile", format="png")
    '''
    return model


    

def runExp(expNum, logScaleData, dataScaler, dataScale, labelScaler, labelScale, 
           cwt_class, #, f0, bw, #sgTimeRes, sgOverlap,
           lossFunction, optimizer, learning_rate, weight_decay, gradiant_noise,
           batchSize, model_name, dropOut_layers):

    epochs = configs['trainer']['epochs']
    fileStructure.setExpTrack_run(expNum=expNum)
    exp_StartTime = timer()

    writeExpSum(cwt_class, 
                logScaleData, dataScaler, dataScale, labelScaler, labelScale, 
                model_name, dropOut_layers, batchSize, lossFunction, optimizer, learning_rate, weight_decay, gradiant_noise)


    # TODO: Set to save the transformed data

    #Log scale the lin data
    scaleStr = f"d: {data_preparation.dataNormConst.type} {data_preparation.dataNormConst.scale}, l: {labelScaler} {labelScale}"
    if(logScaleData): scaleStr = f"{scaleStr}, Log"


    logger.info(f"Get Model")
    # Add the batch size to the dataloader shape, but don't include the number of items
    #if configs['cwt']['doCWT']:
    #if configs['cwt']['wavelet'] != "None":
    print(f"Wavelet base: {cwt_class.wavelet_base}")
    if wavelet_base != "None":
        dataShape = (batchSize,) + data_preparation.CWTDataSet.shape[1:]
        timeD = False
    else:
        dataShape = (batchSize,) + data_preparation.timeDDataSet.shape[1:]
        timeD = True

    isComplex = np.iscomplexobj(cwt_class.wavelet_fun) and configs['cwt']['runAsMagnitude'] == False
    model = getModel(cwt_class.wavelet_name, model_name, dataShape, dropOut_layers = dropOut_layers, timeD=timeD, 
                     complex= isComplex)

    #print(model)
    if cwt_class.wavelet_name != 'None' and cwt_class.wavelet_name != 'spectroGram':
        if isComplex:
            print(model)
            # This is only partialy implemented
            # and conv2d is not implemented :(
            logger.info(f"  !!!!!  TODO: put model in complex format  !!!!!!")
            #model = model.to(torch.complex64) # We do not want the whole model in complex

    if configs['debugs']['saveModelInfo']: saveSumary(model, dataShape, complex=isComplex)

    logger.info(f"Load Trainer")
    model = model.to(device)
    trainer = Trainer(model=model, device=device, dataPrep=data_preparation, fileStru=fileStructure, configs=configs, expNum=expNum, 
                           cwtClass=cwt_class, scaleStr=scaleStr, lossFunction=lossFunction, optimizer=optimizer, learning_rate=learning_rate, weight_decay=weight_decay, gradiant_noise=gradiant_noise, epochs=epochs)

    trainLoss, trainAcc, valAccStats = trainer.train(batchSize)
    valLoss, valAcc, classAcc = trainer.validation(epochs)
    

    exp_runTime = timer() - exp_StartTime
    # Log the results
    with open(expTrackFile, 'a', newline='') as csvFile:
        print(f"Writing data: {expTrackFile}")
        writer = csv.DictWriter(csvFile, fieldnames=expFieldNames, dialect='unix')
        writer.writerow({'Test': expNum,
                         'BatchSize': batchSize,
                         'Epochs': epochs,
                         'wavelet': cwt_class.wavelet_name,
                         'Data Scaler': data_preparation.dataNormConst.type, 
                         'Data Scale': data_preparation.dataNormConst.scale, 
                         'Label Scaler': labelScaler, 
                         'Label Scale': labelScale, 
                         'Loss': lossFunction,
                         'Optimizer': optimizer,
                         'Learning Rate': learning_rate,
                         'Weight Decay': weight_decay,
                         'Gradiant Noise': gradiant_noise,
                         'Model': model.__class__.__name__,
                         'Dropout Layers': dropOut_layers,
                         'Train Loss': trainLoss, 
                         f'Train {accStr}': trainAcc, 
                         'Last Epoch Val Loss': valLoss, 
                         f'Last Epoch Val {accStr}': valAcc,
                         f'Last {lastStats_n} epochs min': valAccStats.min,
                         f'Last {lastStats_n} epochs max': valAccStats.max,
                         f'Last {lastStats_n} epochs mean': valAccStats.mean,
                         f'Last {lastStats_n} epochs std': valAccStats.std,
                         f'Class Acc {accStr}': classAcc,
                         'Time(s)': exp_runTime
        })

    del model
    del trainer


expNum = 1
wavelet_bases = configs['cwt']['wavelet']
#wavelet_bases = ['None']
#if configs['cwt']['doCWT']: wavelet_bases = configs['cwt']['wavelet']

for batchSize in configs['trainer']['batchSize']:
    data_preparation.loadDataSet(writeLog=True, batchSize=batchSize) #Load the timed dataset even if we are doing a cwt
    # we get data freq rate here, and need it for below
    # The hyperperamiters setup for expTracking
    cwt_class = cwt(fileStructure=fileStructure, configs=configs,  dataConfigs = data_preparation.dataConfigs)
    data_preparation.plotDataByWindow(cwt_class=cwt_class, logScaleData=False)
    for wavelet_base in wavelet_bases:
        #logger.info(f"Wavelet: {wavelet_base}")
        if wavelet_base == 'ricker':
            centerFreqs = [1]
            bandwidths = [1]
        elif wavelet_base == 'fstep' or wavelet_base == 'cfstep': 
            centerFreqs = configs['cwt']['waveLet_center_freq']
            bandwidths = [1]
        elif wavelet_base == 'morl':
            centerFreqs = [0.8125]
            bandwidths = [6.0]
        elif wavelet_base == 'None' or wavelet_base == 'spectroGram':
            centerFreqs = [0] # Dummy to 0... Gotta have something to chew
            bandwidths = [0]
        else:
            centerFreqs = configs['cwt']['waveLet_center_freq']
            bandwidths = configs['cwt']['waveLet_bandwidth_freq']
    
        for center_freq in centerFreqs:
            #logger.info(f"Center Freq: {center_freq}")
            for bandwidth in bandwidths:
                # Load the CWT Here
                logScaleFreq = configs['cwt']['logScaleFreq'] #keep as var in case we want to add to exp tracker
                # Go here even on None just to setup the name
                #if wavelet_base != 'None': Put back in when we sort norm out
                #if wavelet_base == 'spectroGram':
                #    data_preparation.generateSpectraDataByWindow()
                if wavelet_base != 'None':
                    cwt_class.setupWavelet(wavelet_base=wavelet_base, sampleRate_hz=data_preparation.dataConfigs.sampleRate_hz, f0=center_freq, bw=bandwidth, useLogForFreq=logScaleFreq)
                    data_preparation.generateCWTDataByWindow(cwt_class=cwt_class, logScaleData=False)
    
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
                                                    for gradiant_noise in configs['trainer']['gradiant_noise']:
                                                        for model_name in configs['model']['name']:
                                                            for dropOut_layers in configs['model']['dropOut']:
                                                                sgTimeRes = 1.0
                                                                sgOverlap = 0.95
        
                                                                logger.info(f"==============================")
                                                                logger.info(f"Wavelet: {wavelet_base}, Center Frequency: {center_freq}, Bandwidth: {bandwidth}, logData: {logScaleData}")
                                                                logger.info(f"Experiment:{expNum}, type: {dataScaler}, labelScaler: {labelScaler}, dataScale: {dataScale_value}, labelScale: {labelScale_value}")
                                                                logger.info(f"Loss: {lossFunction}, Optimizer: {optimizer}, Learning Rate: {learning_rate}, Weight Decay: {weight_decay}, Gradiant Noise: {gradiant_noise}")
              
                                                                #TODO: just send the cwtClass 
                                                                if configs['debugs']['runModel']:
                                                                    runExp(expNum=expNum, logScaleData=logScaleData,
                                                                           dataScaler=dataScaler, dataScale=dataScale_value, labelScaler=labelScaler, labelScale=labelScale_value, 
                                                                           cwt_class=cwt_class, #wavelet=wavelet_base, #f0=center_freq, bw=bandwidth, #sgTimeRes=sgTimeRes, sgOverlap=sgOverlap,
                                                                           lossFunction=lossFunction, optimizer=optimizer, learning_rate=learning_rate, weight_decay=weight_decay,  gradiant_noise=gradiant_noise,
                                                                           batchSize = batchSize, model_name=model_name, dropOut_layers = dropOut_layers)
                                                                expNum += 1
