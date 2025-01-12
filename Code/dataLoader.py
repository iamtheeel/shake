###
# trainer_main.py
# Joshua Mehlman
# MIC Lab
# Fall, 2024
###
# Data Loader
###

import h5py, csv
import numpy as np
import copy

import torch
import torch.nn.functional as tFun
from torch.utils.data import DataLoader, TensorDataset, random_split


from genPlots import *

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from dataclasses import dataclass
from typing import Optional
@dataclass
class normClass:
    type: str
    min: Optional[float] = None
    max: Optional[float] = None 
    mean: Optional[float] = None
    std: Optional[float] = None
    scale: Optional[float] = None

# TODO: downsample 

class dataLoader:
    def __init__(self, config, dir, logfile):
        self.regression = config['model']['regression']
        self.seed = config['trainer']['seed'] 
        torch.manual_seed(self.seed)
        # Load up the dataset info
        self.dataPath = config['data']['dataPath']# Where the data is
        self.test = config['data']['test']         # e.x. "Test_2"
        self.valPercen = config['data']['valSplitPercen']
        self.sensorList = config['data']['sensorList']
        self.batchSize = config['data']['batchSize']
        self.sensorChList = config['data']['sensorChList'] 
        self.dataDir = dir

        #TODO: Get from file
        self.logfile = logfile

        self.classes = config['data']['classes']
        if self.regression: self.nClasses = 1
        else:               self.nClasses = len(self.classes)

        #Data
        self.stompCh = config['data']['stompSens']
        self.dataThresh = config['data']['dataThresh']
        self.stompThresh = config['data']['stompThresh']

        self.samRate_hz = 0
        self.units = None
        self.dataLen_pts = 0
        self.windowLen_s = config['data']['windowLen'] 
        self.stepSize_s = config['data']['stepSize']
        self.windowLen = None
        self.stepSize = None

        self.dataPoints = 0 #99120
        self.nSensors = 0
        self.nTrials = 0

        self.scale = config['data']['scale']

        self.data_raw = None
        self.labels_raw = None
        self.data = None
        self.labels = None
        self.data_norm = None
        self.labels_norm = None
        self.dataLoader_t = None
        self.dataLoader_v = None

        self.dataNormConst = None
        self.labNormConst = None


    def get_data(self):
        # Load all the data to a 3D numpy matrix:
        # 0 = trial: around 20
        # 1 = channels: 20
        # 2 = dataPoints: 99120

        # The data is [1,3]
        # Each image is 2

        # The labels are an array:
        # labels = subject/run
        fieldnames = ['subject', 'data file', 'label file', 'dataRate', 'nSensors', 'nTrials', 'dataPoints']
        with open(self.logfile, 'a', newline='') as csvFile:
            writer = csv.DictWriter(csvFile, fieldnames=fieldnames, dialect='unix')
            writer.writeheader()

        self.subjects = self.getSubjects()
        for subjectNumber in self.subjects:
            data_file_hdf5, label_file_csv = self.getFileName(subjectNumber)
            logger.info(f"Dataloader, datafile: {data_file_hdf5}")

            # Load data file
            subjectData = self.getSubjectData(data_file_hdf5) # Only the chans we are interested, in the order we want
            subDataShape = np.shape(subjectData) 
            #logger.info(f"Subject: {subjectNumber}, subject shape: {np.shape(subjectData)}")

            speed =  self.getSpeedLabels(label_file_csv)

            # Window the data
            windowedBlock, labelBlock = self.windowData(data=subjectData, window_len=self.windowLen, step_len=self.stepSize, subject=subjectNumber, speed=speed)
            #logger.info(f"data: {windowedBlock.shape}, label: {labelBlock.shape}")


            # Append the data to the set
            try:              data = np.append(data, windowedBlock, axis=0)  # or should this be a torch tensor?
            except NameError: data = windowedBlock

            # Onehot the labels
            labelBlock = torch.from_numpy(labelBlock)

            if self.regression:
                thisSubLabels = labelBlock.unsqueeze(1)
            else: 
                thisSubLabels = tFun.one_hot(labelBlock, num_classes=self.nClasses) # make 0 = [1,0,0], 1 = [0,1,0]... etc
                #logger.info(f"one_hot label: {thisSubLabels.shape}")


            try:              labels = torch.cat((labels, thisSubLabels), 0) 
            except NameError: labels = thisSubLabels
            #logger.info(f"Labels: {thisSubLabels}")
            logger.info(f"Up to: {subjectNumber}, Labels, data shapes: {thisSubLabels.shape}, {data.shape}")

            with open(self.logfile, 'a', newline='') as csvFile:
                writer = csv.DictWriter(csvFile, fieldnames=fieldnames, dialect='unix')
                writer.writerow({'subject': subjectNumber,
                                'data file': data_file_hdf5, 
                                'label file': label_file_csv, 
                                'dataRate': self.samRate_hz, 
                                'nSensors': subDataShape[1], 
                                'nTrials': subDataShape[0], 
                                'dataPoints': subDataShape[2]
                                })

        logger.info(f"Data shapes: Labels, data: {labels.shape}, {data.shape}")
        # Plot the data
        '''
        sTime = 0
        for row in range(data.shape[0]):
        #for row in data:
            # subject, run, time
            # rms
            thisLab = torch.argmax(labels[row])
            fileN = f"{sTime}_.jpg"
            title = f"{sTime} {labels[row]}"
            print(f"data shape: {data[row].shape}")
            plotInLine(data[row], 0, fileN, title, sFreq=self.samRate_hz)
            sTime += self.stepSize_s
        '''
        self.data_raw = data
        self.labels_raw = labels.float() 
        logger.info(f"====================================================")

    def resetData(self):
        self.data = copy.deepcopy(self.data_raw) #Data is numpy
        self.labels = self.labels_raw.clone().detach() #labels are tensor

    def createDataloaders(self):
        # Add the "ch"
        # Data is currently: datapoints, height(sensorch), width(datapoints)
        self.data_norm = torch.tensor(self.data_norm, dtype=torch.float32) # dataloader wants a torch tensor
        self.data_norm = self.data_norm.unsqueeze(1) # datapoints, image channels, height, width
        #logger.info(f"shape data: {self.data_norm.shape}, labels: {type(labels)}, {labels.shape}")
        #plotLabels(self.labels_norm)

        logger.info(f"labels: {self.labels_norm.shape}")
        dataSet = TensorDataset(self.data_norm, self.labels_norm)
        #plotRegreshDataSetLab(dataSet, "total set")

        # Split sizes
        trainRatio = 1 - self.valPercen
        train_size = int(trainRatio * len(dataSet))  # 80% for training
        val_size = len(dataSet) - train_size  # 20% for validation

        logger.info(f"dataset: {len(dataSet)}, valPer: {self.valPercen}, train: {train_size}, val: {val_size}")
        # Rand split was not obeying  config['trainer']['seed'], so force the issue
        # random_split sorts the data, this makes it hard to look at validation
        dataSet_t, dataSet_v = random_split(dataSet, [train_size, val_size], torch.Generator().manual_seed(self.seed))
        dataSet_v.indices.sort() # put the validation data back in order
        #plotRegreshDataSetLab(dataSet_v, "validation set")

        self.dataLoader_t = DataLoader(dataSet_t, batch_size=self.batchSize, shuffle=False)
        self.dataLoader_v = DataLoader(dataSet_v, batch_size=1, shuffle=False)
        #plotRegreshDataLoader(self.dataLoader_t)
        #plotRegreshDataLoader(self.dataLoader_v)


        with open(self.logfile, 'a', newline='') as csvFile:
            print(f"datashpe: {dataSet_t[0][0].shape}")

            writer = csv.writer(csvFile, dialect='unix')
            writer.writerow(['train size', 'validation size (batch size = 1)', 'batch ch height width', 'classes'])
            writer.writerow([len(dataSet_t), len(dataSet_v), dataSet_t[0][0].shape, self.classes])
            writer.writerow(['---------'])

        #return data_loader_t, data_loader_v
    

    def windowData(self, data:np.ndarray, window_len, step_len, subject, speed):
        #logger.info(f"Window length: {window_len}, step: {step_len}, data len: {data.shape}")
        # Strip the head/tails

        dataFile= f"{self.dataDir}/data.csv"
        with open(dataFile , 'a', newline='') as csvFile:
            csvFile.write('Subject, speed (m/s), run, startTime (s), label')
            for i in range(data.shape[1]):
                thisCh = self.sensorList[i]
                csvFile.write(f", sensor {thisCh} rms")
            csvFile.write(f", startTime(s)")
            csvFile.write("\n")

            # do while endPoint <= thisPoint + data len
            logger.debug(f"windowData, Data: {data.shape}")
            for run in range(data.shape[0]): # make sure the data is in order one run at a time
                startPoint = 0 #points
                hasStomp = -1 # -1: no stomp yet, then = #since stomp

                while True:
                    endPoint = startPoint + window_len
                    #logger.info(f"window run: {run},  startPoint: {startPoint}, windowlen: {window_len}, endPoint: {endPoint}, dataLen: {self.datalen_pts}")
                    if self.dataLen_pts <= endPoint: break
    
                    thisDataBlock = data[run, :, startPoint:endPoint]  # trial, sensor, dataPoint
                    #logger.info(f"window data shape: {thisDataBlock.shape}")

                    for i in range(thisDataBlock.shape[0]): # Each ch
                        rms_thisCh = np.sqrt(np.mean(np.square(thisDataBlock[i,:])))
                        try:              rms_allCh = np.append(rms_allCh, rms_thisCh)
                        except NameError: rms_allCh = rms_thisCh
                        #print(f"rms_allCh: {rms_allCh.shape}")

                    # Look for stomp
                    if startPoint == 0: rms_BaseLine = rms_allCh.copy()
                    rms_ratio = rms_allCh/rms_BaseLine
                    #print(f"rms_allCh = {rms_allCh}")
                    #print(f"rms_BaseLin = {rms_BaseLine}")
                    thisSubjectId = 0 # we don't know what we have yet
                    if hasStomp  < 0:
                        for i in self.stompCh:
                            dataNum = self.sensorList.index(i)
                            value = rms_ratio[dataNum]
                            #logger.info(f"ch: {i}, {dataNum}, rmsRatio: {value}, thresh: {self.stompThresh}")
                            if value > self.stompThresh: 
                                thisSubjectId = -1
                                hasStomp = 0 
                                break
                    else: hasStomp += 1 # Would probably be nice to just increment startPoint, but that makes another can of worms


                    ### for investigation
                    #plotThis = rms_ratio
                    plotThis = rms_allCh
                    try:              plot_run = np.append(plot_run, plotThis.reshape(-1, 1), axis=1)
                    except NameError: plot_run = plotThis.reshape(-1, 1)
                    ###
    
                    # append the data
                    if hasStomp >= 3:
                        thisDataBlock = np.expand_dims(thisDataBlock, axis=0) # add the run dim back to append
                        try:              windowedData = np.append(windowedData, thisDataBlock, axis=0) # append on trials, now trials/windows
                        except NameError: windowedData = thisDataBlock
                        #append the labels
                        # The detection of no step is done in getSubjecteLabel

                        if thisSubjectId >=0: # Negitives reservered 
                            thisSubjectId = self.getSubjectLabel(subject, rms_ratio) 
                            if self.regression: thisLabel = speed[run]
                            else:               thisLabel = thisSubjectId

                        try:              labels = np.append(labels, thisLabel)
                        except NameError: labels = thisLabel
    
                        csvFile.write(f"{subject}, {speed[run]}, {run}, {startPoint/self.samRate_hz}, {thisSubjectId}")
                        for i in range(len(rms_allCh)): # Each ch
                            csvFile.write(f", {rms_allCh[i]}")
                        csvFile.write("\n")
    
                    startPoint += step_len
                    del rms_allCh

                    ## End each window
                #logger.info(f"Data Block: {windowedData.shape}, labels: {labels.shape}")

                # Plot the rms, and max of each run
                #title = f"rmsRatio of t=1 Subject: {subject}, run: {run}, speed: {speed[run]:.2f} "
                title = f"rms Subject: {subject}, run: {run}, speed: {speed[run]:.2f} "
                fileN = f"{self.test}-subject_{subject}-trial_{run}"
                #plotOverlay(plot_run, fileN, title, self.stepSize_s)
                del plot_run


        return windowedData, labels


    def getSubjectData(self, data_file_name):
        with h5py.File(data_file_name, 'r') as file:
            # Get the data from the datafile
            for sensor in self.sensorList:
                ch = self.sensorChList[sensor]-1 
                #print(f"sensors: {sensor}, ch: {ch}")
                thisChData = file['experiment/data'][:, ch-1, :]  # trial, sensor, dataPoint
                thisChData = np.expand_dims(thisChData, axis=1)

                try:              accelerometer_data = np.append(accelerometer_data, thisChData, axis=1)
                except NameError: accelerometer_data = thisChData

            #logger.info(f"get subject data shape: {np.shape(accelerometer_data)} ")

            # Get just the sensors we want
            if self.samRate_hz == 0: 
                # get the peramiters if needed
                # ex: Sample Freq
                self.getDataInfo(file) # get the sample rate from the file

                #logger.info(f"window len: {self.windowLen_s}, step size: {self.stepSize_s}, sample Rate: {self.samRate_hz}")
                self.windowLen = self.windowLen_s * self.samRate_hz
                self.stepSize  = self.stepSize_s  * self.samRate_hz
                #logger.info(f"window len: {self.windowLen}, step size: {self.stepSize}")

        return accelerometer_data 

    def getSubjectLabel(self, subjectNumber, vals):
        # TODO: get from config
        label = 0

        #print(f"vals: {vals}")
        #If any ch is above the thresh, we call it a step
        for chVal in vals:
            if chVal > self.dataThresh: 
                if(subjectNumber == '001'): label = 1
                elif(subjectNumber == '002'): label = 2
                elif(subjectNumber == '003'): label = 3
                break

        return label

    def getSpeedLabels(self, csv_file_name ):
        with open(csv_file_name, mode='r') as speedFile:
            speedFile_csv = csv.DictReader(speedFile)
            for line_number, row in enumerate(speedFile_csv, start=1):
                speed_L = float(row['Gait - Lower Limb - Gait Speed L (m/s) [mean]'])
                speed_R = float(row['Gait - Lower Limb - Gait Speed R (m/s) [mean]'])
                #logger.info(f"Line {line_number}: mean: L={speed_L}, R={speed_R}(m/s) ")
                aveSpeed = (speed_L+speed_R)/2
                try:              speedList = np.append(speedList, aveSpeed)
                except NameError: speedList = aveSpeed

        #print(f"Labels: {speedList}")
        return speedList


    def getSubjects(self):
        if((self.test == "Test_2") or (self.test == "Test_3")):
            subjects = ['001', '002', '003']
        elif(self.test == "Test_4"):
            subjects = ['three_people_ritght_after_the_other_001_002_003', 
                        'two_people_next_to_each_other_001_003' , 
                        'two_people_next_to_each_other_002_003']

        return subjects

    def getFileName(self, subjectNumber):
        trial_str = None
        csv_str = None
        if(self.test == "Test_2"):
            trial_str = f"walking_hallway_single_person_APDM_{subjectNumber}"
            csv_str = f"APDM_data_fixed_step/MLK Walk_trials_{subjectNumber}_fixedstep"
            #walking_hallway_single_person_APDM_
        elif(self.test == "Test_3"):
            trial_str = f"walking_hallway_single_person_APDM_{subjectNumber}_free_pace"
            csv_str = f"APDM_data_freepace/MLK Walk_trials_{subjectNumber}_freepace"
        elif(self.test == "Test_4"):
            trial_str = f"walking_hallway_classroom_{subjectNumber}"
            #TestData/Test_4/data/walking_hallway_classroom_three_people_ritght_after_the_other_001_002_003.hdf5
        else:
            logger.error(f"ERROR: No such subject, test: {self.test}, {subjectNumber}")
        
        trial_str = f"{self.dataPath}/{self.test}/data/{trial_str}.hdf5"
        csv_str = f"{self.dataPath}/{self.test}/{csv_str}.csv"

        return trial_str, csv_str

    def getDataInfo(self, file):
        general_parameters = file['experiment/general_parameters'][:]
        #logger.info(f"Data File parameters: {general_parameters}")
        self.samRate_hz = int(general_parameters[0]['value'].decode('utf-8'))
        self.units = general_parameters[0]['units'].decode('utf-8')
        #logger.info(f"Data cap rate: {self.samRate_hz} {units}")

        dataBlockSize = file['experiment/data'].shape 
        self.dataLen_pts = dataBlockSize[2]
        #logger.info(f"File Size: {dataBlockSize}")
        #nSensors = 19 # There are 20 acceleromiters
        self.nSensors = dataBlockSize[1]
        self.nTrials = dataBlockSize[0]
        #logger.info(f"nsensor: {self.nSensors}, nTrials: {self.nTrials}, dataPoints: {self.dataLen_pts} ")


    def scale_data(self, data, scaler, logFile):
        isTensor = False
        if isinstance(data, torch.Tensor): #convert to numpy
            data = data.numpy()
            isTensor = True

        if scaler == "std": dataScaled, norm = self.std_data(data, logFile)
        else:               dataScaled, norm = self.norm_data(data, logFile, scaler)
        #elif scaler == "minMaxNorm": dataScaled, norm = self.norm_data(data, logFile, scaler)

        if isTensor: # and back to tensor
            dataScaled = torch.from_numpy(dataScaled)

        return dataScaled, norm

    def unScale_data(self, data, scalerClass):
        print(f"data: {type(data)}, {type(data[0])}, {len(data)}")
        print(f" scalerClass: mean: {type(scalerClass.std)}")
        data = np.array(data) # make numpy so we can work with it

        if scalerClass.type == "std":
            data = self.unScale_std(data, scalerClass)
        else:
            data = self.unScale_norm(data, scalerClass)

        return data

    def unScale_std(self, data, scalerClass):
        data = data * scalerClass.std + scalerClass.mean

        return data

    def unScale_norm(self, data, scalerClass:normClass):
        if   scalerClass.type == 'meanNorm'  : normTo = scalerClass.mean
        elif scalerClass.type == 'minMaxNorm': normTo = scalerClass.min

        data = normTo + data * (scalerClass.max - scalerClass.min) / scalerClass.scale

        return data

    def std_data(self, data, logFile):
        # Normalize the data
        # float: from 0 to 1
        # 0.5 = center

        norm = normClass(type="std", mean=np.mean(data), std=np.std(data))
        #norm = normClass(type="std", min=np.min(data), max=np.max(data), mean=mean, std=std)
        #logger.info(f"Orig: {data[0:3, 0:5, 0:2]}")

        # scale the data
        normData = (data - norm.mean)/norm.std # standardise

        self.logScaler(logFile,'standardize', 'mean', 'std dev', norm.mean, norm.std)
        logger.info(f"newmin: {np.min(normData)},  newmax: {np.max(normData)}")
        return normData, norm

    def norm_data(self, data, logFile, scaler):
        newMin = -self.scale
        newMax = self.scale
        norm = normClass(type="norm", min=np.min(data), max=np.max(data), mean=np.mean)

        if scaler == 'meanNorm':
            normTo = norm.mean
            norm.scale = 1
        elif scaler == 'minMaxNorm':
            normTo = norm.min
            norm.scale = (newMax - newMin)

        normData = norm.scale*(data-normTo)/(norm.max - norm.min) 

        self.logScaler(logFile, scaler, 'min', 'max', norm.min, norm.mean)
        logger.info(f"newmin: {np.min(normData)},  newmax: {np.max(normData)}")
        return normData, norm

    def logScaler(self, logFile, scaler, name_a, name_b, data_a, data_b, scale=1):
        with open(logFile, 'a', newline='') as csvFile:
            writer = csv.writer(csvFile, dialect='unix')
            writer.writerow([f'--------- {scaler}  -------'])
            writer.writerow([name_a, name_b, 'scale'])
            writer.writerow([data_a, data_b, scale])
            writer.writerow(['---------'])
        logger.info(f"Data: {name_a} = {data_a}, {name_b} = {data_b}, scale = {scale}")