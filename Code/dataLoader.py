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
import os

import torch
import torch.nn.functional as tFun
from torch.utils.data import DataLoader, TensorDataset, random_split
import pickle
import time
from tqdm import tqdm  #progress bar

from genPlots import *
from cwtTransform import cwt

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from dataclasses import dataclass
from typing import Optional
@dataclass
class normClass:
    type: Optional[str] = None
    min: Optional[float] = None
    max: Optional[float] = None 
    mean: Optional[float] = None
    std: Optional[float] = None
    scale: Optional[float] = None

class dataConfigs:
    cwtDataShape: Optional[list] = None
    sampleRate_hz: Optional[int] = None
    units: Optional[str] = None
    dataLen_pts: Optional[int] = None
    nSensors: Optional[int] = None
    nTrials: Optional[int] = None
    chList: Optional[list] = None



class dataLoader:
    def __init__(self, config, dir, logfile):
        self.regression = config['model']['regression']
        self.seed = config['trainer']['seed'] 
        torch.manual_seed(self.seed)
        # Load up the dataset info
        self.dataPath = config['data']['dataPath']# Where the data is
        self.test = config['data']['test']         # e.x. "Test_2"
        self.valPercen = config['data']['valSplitPercen']
        self.batchSize = config['data']['batchSize']
        self.inputDataDir = dir

        #TODO: Get from file
        self.logfile = logfile

        self.classes = config['data']['classes']
        if self.regression: self.nClasses = 1
        else:               self.nClasses = len(self.classes)

        #Data
        self.stompCh = config['data']['stompSens']
        self.dataThresh = config['data']['dataThresh']
        self.stompThresh = config['data']['stompThresh']

        self.dataConfigs = dataConfigs()
        self.dataConfigs.sampleRate_hz = 0
        self.dataConfigs.units = None
        self.dataConfigs.dataLen_pts = 0
        self.dataConfigs.chList = config['data']['chList']
        self.dataConfigs.nSensors = len(self.dataConfigs.chList)

        self.windowLen_s = config['data']['windowLen']
        self.stepSize_s = config['data']['stepSize']
        self.windowLen = 0
        self.stepSize = 0

        self.dataPoints = 0 #99120
        self.nTrials = 0

        self.data_raw = None
        self.labels_raw = None
        self.subjectList_raw = None
        self.runList_raw = None
        self.startTimes_raw = None


        self.data = None
        self.labels = None
        self.labels_norm = None
        self.dataLoader_t = None
        self.dataLoader_v = None

        self.dataNormConst = None
        self.labNormConst = None

        self.dataNormConst = normClass()

        #Set up a string for saving the dataset so we can see if we have already loaded this set
        chList_str = "_".join(map(str, self.dataConfigs.chList))
        #TODO: add more info:
        # windows limit, runs limit
        if self.regression: regClas = "regression"
        else:               regClas = "classification"
        self.dataSaveDir = f"{self.dataPath}/{config['data']['dataSetDir']}_{regClas}_chList-{chList_str}"
        runLimit = config['data']['limitRuns']
        if runLimit > 0: self.dataSaveDir = f"{self.dataSaveDir}_runLim-{runLimit}"
        winLimit = config['data']['limitWindowLen']
        if winLimit > 0: self.dataSaveDir = f"{self.dataSaveDir}_winCountLim-{winLimit}"
        self.dataSaveDir = f"{self.dataSaveDir}_StompThresh-{config['data']['stompThresh']}"
        self.dataSaveDir = f"{self.dataSaveDir}_DataThresh-{config['data']['dataThresh']}"

        self.configs = config

    def get_data(self):
        # Load all the data to a 3D numpy matrix:
        # 0 = trial: around 20
        # 1 = channels: 20
        # 2 = dataPoints: 99120

        # The data is [1,3]
        # Each image is 2

        # The labels are an array:
        # labels = subject/run
        fieldnames = ['subject', 'data file', 'label file', 'dataRate', 'nSensors', 'nTrials', 'dataPoints', 'chList']
        with open(self.logfile, 'a', newline='') as csvFile:
            writer = csv.DictWriter(csvFile, fieldnames=fieldnames, dialect='unix')
            writer.writeheader()

        self.subjects = self.getSubjects()
        for subjectNumber in self.subjects:
            data_file_hdf5, label_file_csv = self.getFileName(subjectNumber)
            logger.info(f"Dataloader, datafile: {data_file_hdf5}")

            # Load data file
            subjectData = self.getSubjectData(data_file_hdf5) # Only the chans we are interested, in the order we want. Sets the sample rate and friends
            subDataShape = np.shape(subjectData) 
            #logger.info(f"Subject: {subjectNumber}, subject shape: {np.shape(subjectData)}")

            #self.plotTimeData(subjectData, subjectNumber)
            #plotRunFFT(subjectData, self.dataConfigs.sampleRate_hz, subjectNumber, 0, "ByRun")

            speed =  self.getSpeedLabels(label_file_csv)
            #print(f"speeds: {speed}")

            # Window the data
            windowedBlock, labelBlock, subjectBlock, runBlock, startTimes = self.windowData(data=subjectData, subject=subjectNumber, speed=speed)
            #logger.info(f"data: {windowedBlock.shape}, label: {labelBlock.shape}")


            # Append the data to the set
            try:              data = np.append(data, windowedBlock, axis=0)  # or should this be a torch tensor?
            except NameError: data = windowedBlock

            labelBlock = torch.from_numpy(labelBlock)
            if self.regression:
                thisSubLabels = labelBlock.unsqueeze(1)
            else: 
                thisSubLabels = tFun.one_hot(labelBlock, num_classes=self.nClasses) # make 0 = [1,0,0], 1 = [0,1,0]... etc
                #logger.info(f"one_hot label: {thisSubLabels.shape}")

            # The labels and subjects are torch 
            try:              labels = torch.cat((labels, thisSubLabels), 0) 
            except NameError: labels = thisSubLabels

            subjectBlock = torch.from_numpy(subjectBlock)
            try:              subject_list = torch.cat((subject_list, subjectBlock), 0) 
            except NameError: subject_list = subjectBlock

            runBlock = torch.from_numpy(runBlock)
            try:              run_list = torch.cat((run_list, runBlock), 0) 
            except NameError: run_list = runBlock

            startTimesBlock = torch.from_numpy(startTimes)
            try:              startTimes_list = torch.cat((startTimes_list, startTimesBlock), 0) 
            except NameError: startTimes_list = startTimesBlock


            #logger.info(f"Labels: {thisSubLabels}")
            logger.info(f"Up to: {subjectNumber}, Labels, data shapes: {thisSubLabels.shape}, {data.shape}")

            with open(self.logfile, 'a', newline='') as csvFile:
                writer = csv.DictWriter(csvFile, fieldnames=fieldnames, dialect='unix')
                writer.writerow({'subject': subjectNumber,
                                'data file': data_file_hdf5, 
                                'label file': label_file_csv, 
                                'dataRate': self.dataConfigs.sampleRate_hz, 
                                'nSensors': subDataShape[1], 
                                'nTrials': subDataShape[0], 
                                'dataPoints': subDataShape[2],
                                'chList': self.dataConfigs.chList
                                })


        logger.info(f"Labels: {type(labels)}, {labels.shape}, Data: {type(data)}, {data.shape}")
        if not self.configs['data']['dataSetDir'] == "":
            # Create dataset directory if it doesn't exist
            if not os.path.exists(self.dataSaveDir): os.makedirs(self.dataSaveDir)

            logger.info(f"Saved data to {self.dataSaveDir}/data.npy")
            logger.info(f"Saved labels to {self.dataSaveDir}/labels.pt")
            np.save(f"{self.dataSaveDir}/data.npy", data)
            torch.save(labels, f"{self.dataSaveDir}/labels.pt")
            torch.save(subject_list, f"{self.dataSaveDir}/subjects.pt")
            torch.save(run_list, f"{self.dataSaveDir}/runs.pt")
            torch.save(startTimes_list, f"{self.dataSaveDir}/startTimes.pt")
            # Save data configs
            with open(f"{self.dataSaveDir}/dataConfigs.pkl", 'wb') as f:
                pickle.dump(self.dataConfigs, f)
            logger.info(f"Saved data configs to {self.dataSaveDir}/dataConfigs.pkl")
            #np.save(f"{self.dataSaveDir}/subjects.npy", subjects)

        self.data_raw = data
        self.labels_raw = labels.float()
        self.subjectList_raw = subject_list
        self.runList_raw = run_list
        self.startTimes_raw = startTimes_list
        logger.info(f"====================================================")

    def loadDataSet(self):
        logger.info(f"Loading data from {self.dataSaveDir}/data.npy")
        logger.info(f"Loading labels from {self.dataSaveDir}/labels.pt")
        logger.info(f"Loading subjects from {self.dataSaveDir}/subjects.pt")
        self.data_raw = np.load(f"{self.dataSaveDir}/data.npy")
        self.labels_raw = torch.load(f"{self.dataSaveDir}/labels.pt", weights_only=False).float()
        self.subjectList_raw = torch.load(f"{self.dataSaveDir}/subjects.pt", weights_only=False)
        self.runList_raw = torch.load(f"{self.dataSaveDir}/runs.pt", weights_only=False)
        self.startTimes_raw = torch.load(f"{self.dataSaveDir}/startTimes.pt", weights_only=False)

        with open(f"{self.dataSaveDir}/dataConfigs.pkl", 'rb') as f:
            #This will load the ch list from the saved file
            self.dataConfigs = pickle.load(f)
        logger.info(f"Loaded data configs from {self.dataSaveDir}/dataConfigs.pkl")

    def plotFFTWindowdData(self):
        show = configs['plots']['showFFTPlots']

        logger.info(f"Plotting FFT of Each windowed block")
        for i, windowBlockData in enumerate(self.data_raw): #each window
            plotFFT(windowBlockData, self.dataConfigs.sampleRate_hz, self.subjectList_raw[i], self.runList_raw[i], self.startTimes_raw[i], "ByWindow", show=show)

    def plotWindowdData(self):
        show = configs['plots']['showWindowPlots']

        logger.info(f"Plotting windowed data: {self.data_raw.shape}, samRate: {self.dataConfigs.sampleRate_hz}")
        sTime = 0
        for i, windowBlockData in enumerate(self.data_raw): #each window
            fileN = f"subject-{self.subjectList_raw[i]}_run{self.runList_raw[i]+1}_time-{self.startTimes_raw[i]}"
            title = f"subject:{self.subjectList_raw[i]}, run: {self.runList_raw[i]+1}, time: {self.startTimes_raw[i]}"
            #print(f"data shape: {data[row].shape}, sampRate: {self.dataConfigs.sampleRate_hz}")
            plotInLine(windowBlockData, fileN, title, sFreq=self.dataConfigs.sampleRate_hz, show=show)
            sTime += self.stepSize_s

    def plotTimeData(self, data, subject):
        # Plot the data
        runNum = 1
        for row in range(data.shape[0]):
            # subject, run, time
            # rms
            #thisLab = torch.argmax(labels[row])
            fileN = f"subject-{subject}_run-{runNum}"
            title = f"subject:{subject}, run:{runNum}"
            #print(f"data shape: {data[row].shape}, sampRate: {self.dataConfigs.sampleRate_hz}")
            plotInLine(data[row], fileN, title, sFreq=self.dataConfigs.sampleRate_hz)
            runNum += 1


    def resetData(self):
        logger.info(f"Copy the data so we don't overwrite the loaded data: {self.data_raw.shape}")
        # Add the "ch"
        # Data is currently: datapoints, height(sensorch), width(datapoints)
        self.data = copy.deepcopy(self.data_raw) #Data is numpy
        #self.data = np.expand_dims(self.data, axis=1)  # Equivalent to unsqueeze(1)
        self.labels = self.labels_raw.clone().detach() #labels are tensor

    def createDataloaders(self, expNum):
        self.data = torch.tensor(self.data, dtype=torch.float32) # dataloader wants a torch tensor

        logger.info(f"data: {self.data.shape}")
        logger.info(f"labels: {self.labels_norm.shape}")
        logger.info(f"subjects: {type(self.subjectList_raw)}, {self.subjectList_raw.shape}")
        dataSet = TensorDataset(self.data, self.labels_norm, self.subjectList_raw)
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

        if expNum == 1: # Only for the first experiment
            with open(self.logfile, 'a', newline='') as csvFile:
                logger.info(f"data shape: {dataSet_t[0][0].shape}")

                writer = csv.writer(csvFile, dialect='unix')
                writer.writerow(['train size', 'validation size (batch size = 1)', 'batch ch height width', 'classes'])
                writer.writerow([len(dataSet_t), len(dataSet_v), dataSet_t[0][0].shape, self.classes])
                writer.writerow(['---------'])

    

    def windowData(self, data:np.ndarray, subject, speed):
        logger.info(f"Window length: {self.windowLen}, step: {self.stepSize}, data len: {data.shape}")
        # Strip the head/tails

        dataFile= f"{self.inputDataDir}/data.csv"
        with open(dataFile , 'a', newline='') as csvFile:
            csvFile.write('Subject, speed (m/s), run, startTime (s), label')
            for i in range(data.shape[1]):
                thisCh = self.dataConfigs.chList[i]
                csvFile.write(f", sensor {thisCh} rms")
            csvFile.write("\n")

            # do while endPoint <= thisPoint + data len
            logger.debug(f"windowData, Data: {data.shape}")
            if self.configs['data']['limitRuns'] > 0:
                dataEnd = self.configs['data']['limitRuns']
            else:
                dataEnd = data.shape[0]
            #for run in range(data.shape[0]): # make sure the data is in order one run at a time
            for run in range(dataEnd): # test with the first few dataums

                startPoint = 0 #points
                hasStomp = -1 # -1: no stomp yet, then = #since stomp
                nWindows = 0

                while True:
                    nWindows += 1
                    endPoint = startPoint + self.windowLen
                    #logger.info(f"window run: {run},  startPoint: {startPoint}, windowlen: {window_len}, endPoint: {endPoint}, dataLen: {self.dataConfigs.dataLen_pts}")
                    if self.dataConfigs.dataLen_pts <= endPoint: break
                    if self.configs['data']['limitWindowLen'] > 0:
                        if nWindows >= self.configs['data']['limitWindowLen']: break

                    thisDataBlock = data[run, :, startPoint:endPoint]  # trial, sensor, dataPoint
                    #logger.info(f"window data shape: {thisDataBlock.shape}")

                    for i in range(thisDataBlock.shape[0]): # Each ch
                        rms_thisCh = np.sqrt(np.mean(np.square(thisDataBlock[i,:])))
                        try:              rms_allCh = np.append(rms_allCh, rms_thisCh)
                        except NameError: rms_allCh = rms_thisCh
                        #print(f"rms_allCh: {rms_allCh.shape}")

                    # Keep the RMS of time = 0 for a baseline
                    if startPoint == 0: rms_BaseLine = rms_allCh.copy()
                    rms_ratio = rms_allCh/rms_BaseLine  # The ratio of the RMS of the data to the baseline for stomp and no step
                    #print(f"rms_allCh = {rms_allCh}")
                    #print(f"rms_BaseLin = {rms_BaseLine}")

                    # Look for stomp, and keeps track of how many windows since the stomp
                    # The detection of no step is done in getSubjecteLabel
                    if self.stompThresh == 0: nSkips = 0
                    else                    : nSkips = 3
                    thisSubjectId, hasStomp = self.findDataStart(hasStomp, rms_ratio, nSkips)
                    #print(f"stompThresh: {self.stompThresh}, nSkips: {nSkips}")

                    ### for investigation
                    #plotThis = rms_ratio
                    plotThis = rms_allCh
                    try:              plot_run = np.append(plot_run, plotThis.reshape(-1, 1), axis=1)
                    except NameError: plot_run = plotThis.reshape(-1, 1)
                    ###
    
                    # append the data
                    if hasStomp >= nSkips:
                        # TODO: make this one data structure
                        thisStartTime = startPoint/self.dataConfigs.sampleRate_hz
                        thisSubjectId = self.getSubjectLabel(subject, rms_ratio) 
                        #print(f"this | subjectId: {thisSubjectId}, run:{run}, startTime: {thisStartTime}")
                        if not self.regression or thisSubjectId > 0:
                            #print(f"using | subjectId: {thisSubjectId}, run:{run}, startTime: {thisStartTime}")
                            # Are the lables speeds, or subject Id
                            if thisSubjectId >=0: # Negitives reservered 
                                if self.regression: thisLabel = speed[run]
                                else:               thisLabel = thisSubjectId

                            thisDataBlock = np.expand_dims(thisDataBlock, axis=0) # add the run dim back to append

                            # Append the data, labels, and all that junk
                            try:              windowedData = np.append(windowedData, thisDataBlock, axis=0) # append on trials, now trials/windows
                            except NameError: windowedData = thisDataBlock
                            try:              labels = np.append(labels, thisLabel)
                            except NameError: labels = thisLabel
                            thisSubjectNumber = self.getSubjectID(subject) #Keep track of the subject number appart from the label
                            try:              subjects = np.append(subjects, thisSubjectNumber)
                            except NameError: subjects = thisSubjectNumber
                            try:              runs = np.append(runs, run+1)
                            except NameError: runs = run+1
                            try:              startTimes = np.append(startTimes, thisStartTime)
                            except NameError: startTimes = thisStartTime

                        # Do we want to log the 0 vel data for regresion too? 
                        csvFile.write(f"{subject}, {speed[run]}, {run}, {thisStartTime}, {thisSubjectId}")
                        for i in range(len(rms_allCh)): # Each ch
                            csvFile.write(f", {rms_allCh[i]}")
                        csvFile.write("\n")
    
                    startPoint += self.stepSize
                    del rms_allCh

                    ## End each window
                #logger.info(f"Data Block: {windowedData.shape}, rms: {plot_run.shape}, labels: {labels.shape}")

                #End window
            #end Run


        return windowedData, labels, subjects, runs, startTimes

    def findDataStart(self, hasStomp, rms_ratio, nSkips):
        thisSubjectID = 0
        if hasStomp  < 0:
            if self.stompThresh == 0:
                hasStomp = nSkips
            else:
                for i in self.stompCh:
                    dataNum = self.dataConfigs.chList.index(i)
                    value = rms_ratio[dataNum]
                    #logger.info(f"ch: {i}, {dataNum}, rmsRatio: {value}, thresh: {self.stompThresh}")
                    if value > self.stompThresh: 
                        thisSubjectId = -1
                        hasStomp = 0 
                        break
        else: hasStomp += 1 # Would probably be nice to just increment startPoint, but that makes another can of worms

        return thisSubjectID, hasStomp

    def getSubjectData(self, data_file_name):
        with h5py.File(data_file_name, 'r') as file:
            # Get the data from the datafile
            #for ch in self.sensorChList[sensor]-1: 
            for ch in self.dataConfigs.chList:
                #ch = self.sensorChList[sensor]-1 
                #print(f"sensors: {sensor}, ch: {ch}")
                thisChData = file['experiment/data'][:, ch-1, :]  # trial, sensor, dataPoint
                thisChData = np.expand_dims(thisChData, axis=1)

                try:              accelerometer_data = np.append(accelerometer_data, thisChData, axis=1)
                except NameError: accelerometer_data = thisChData

            #logger.info(f"get subject data shape: {np.shape(accelerometer_data)} ")

            # Get just the sensors we want
            if self.dataConfigs.sampleRate_hz == 0: 
                # get the peramiters if needed
                # ex: Sample Freq
                self.getDataInfo(file) # get the sample rate from the file

                #logger.info(f"window len: {self.windowLen_s}, step size: {self.stepSize_s}, sample Rate: {self.dataConfigs.sampleRate_hz}")
                self.windowLen = self.windowLen_s * self.dataConfigs.sampleRate_hz
                self.stepSize  = self.stepSize_s  * self.dataConfigs.sampleRate_hz
                #logger.info(f"window len: {self.windowLen}, step size: {self.stepSize}")

        return accelerometer_data 

    def getSubjectLabel(self, subjectNumber, vals):
        #Vals is currently the RMS ratio of the data to the baseline
        # TODO: get from config
        label = 0

        #print(f"vals: {vals}")
        #If any ch is above the thresh, we call it a step
        for chVal in vals:
            if chVal > self.dataThresh:
                label = self.getSubjectID(subjectNumber)
                break

        return label

    def getSubjectID(self, subjectNumber):
        if subjectNumber == '001': return 1
        elif subjectNumber == '002': return 2
        elif subjectNumber == '003': return 3
        else: return 0

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
        self.dataConfigs.sampleRate_hz = int(general_parameters[0]['value'].decode('utf-8'))
        self.dataConfigs.units = general_parameters[0]['units'].decode('utf-8')
        logger.info(f"Data cap rate: {self.dataConfigs.sampleRate_hz} {self.dataConfigs.units}")

        dataBlockSize = file['experiment/data'].shape 
        self.dataConfigs.dataLen_pts = dataBlockSize[2]
        #logger.info(f"File Size: {dataBlockSize}")
        #nSensors = 19 # There are 20 acceleromiters
        self.dataConfigs.nSensors = dataBlockSize[1]
        self.dataConfigs.nTrials = dataBlockSize[0]
        #logger.info(f"nsensor: {self.nSensors}, nTrials: {self.nTrials}, dataPoints: {self.dataLen_pts} ")


    # If you don't send the normClass, it will calculate based on the data
    def scale_data(self, data, logFile, norm:normClass=None, scaler=None, scale=None, debug=False):
        isTensor = False
        if isinstance(data, torch.Tensor): #convert to numpy
            data = data.numpy()
            isTensor = True

        if scaler==None: scaler= norm.type
        if debug: logger.info(f"{norm}")

        if np.iscomplexobj(data):
            if scaler == "std": dataScaled, scalerClass = self.std_complexData(data, logFile, norm)
            else:               dataScaled, scalerClass = self.norm_complexData(data, logFile, norm)
        else:
            if scaler == "std": dataScaled, scalerClass = self.std_data(data, logFile, norm, debug)
            else:               dataScaled, scalerClass = self.norm_data(data, logFile, norm, scale, debug)

        if isTensor: # and back to tensor
            dataScaled = torch.from_numpy(dataScaled)

        return dataScaled, scalerClass

    def unScale_data(self, data, scalerClass, debug=False):
        #print(f"data: {type(data)}, {type(data[0])}, {len(data)}")
        #print(f" scalerClass: mean: {type(scalerClass.mean)}")
        data = np.array(data) # make numpy so we can work with it

        if scalerClass.type == "std":
            data = self.unScale_std(data, scalerClass, debug)
        else:
            data = self.unScale_norm(data, scalerClass, debug)

        return data

    def unScale_std(self, data, scalerClass:normClass, debug):
        if debug:
            print(f"{scalerClass}")
            print(f"before: {data}")
        data = data * scalerClass.std + scalerClass.mean
        if debug:
            print(f"After{data}")

        return data

    def unScale_norm(self, data, scalerClass:normClass, debug):
        if   scalerClass.type == 'meanNorm'  : 
            normTo = scalerClass.mean
            adjustMin = 0
        elif scalerClass.type == 'minMaxNorm': 
            normTo = scalerClass.min
            adjustMin = scalerClass.min

        #print(f"type: {scalerClass.type}")
        scaleMax = scalerClass.scale
        scaleMin = -scaleMax


        #data = newMin + normData* (norm.max - norm.min)/(norm.scale) + normTo 
        data =  ((data - adjustMin) * (scalerClass.max - scalerClass.min) / (scaleMax - scaleMin)) + normTo

        return data

    def std_complexData(self, data, logFile, norm:normClass):
        real = np.real(data)
        imag = np.imag(data)
        mean = np.mean(real) + 1j * np.mean(imag) 
        std = np.std(real) + 1j * np.std(imag) 
        self.dataNormConst = normClass(type="std", mean=mean, std=std)

        stdised_real = (real - norm.mean.real)/norm.std.real
        stdised_imag = (imag - norm.mean.imag)/norm.std.imag

        normData = stdised_real + 1j * stdised_imag

        self.logScaler(logFile, self.dataNormConst, complex=True)
        #logger.info(f"newmin: {np.min(np.abs(normData))},  newmax: {np.max(np.abs(normData))}")
        return normData, norm

    def std_data(self, data, logFile, norm:normClass, debug):
        # Normalize the data
        # float: from 0 to 1
        # 0.5 = center
        if norm == None:
            norm = normClass(type="std", mean=np.mean(data), std=np.std(data))

        # scale the data
        normData = (data - norm.mean)/norm.std # standardise

        if debug:
            logger.info(norm)
            logger.info(f"Orig: \n{data[0:8]}")
            logger.info(f"Orig: \n{normData[0:8]}")

        self.logScaler(logFile, norm)
        #logger.info(f"newmin: {np.min(np.abs(normData))},  newmax: {np.max(np.abs(normData))}")
        return normData, norm

    def norm_complexData(self, data, logFile, norm:normClass):
        #L2 norm is frobenious_norm, saved as mean

        if norm == None:
            norm = normClass(type="norm", min=np.min(data), max=np.max(data), mean=np.linalg.norm(data) )

        normData = data/norm.mean

        self.logScaler(logFile, norm)
        #logger.info(f"l2 norm: {self.dataNormConst.mean}, newmin: {np.min(normData)},  newmax: {np.max(normData)}")
        return normData, norm

    def norm_data(self, data, logFile, norm:normClass, scale, debug):
        if norm == None:
            norm = normClass(type="norm", min=np.min(data), max=np.max(data), mean=np.mean(data), scale=scale)

        if debug: 
            logger.info(f"{norm}")
        newMin = -norm.scale
        newMax = norm.scale

        if norm.type == 'meanNorm': 
            normTo = norm.mean
            adjustMin = 0
        elif norm.type == 'minMaxNorm': 
            normTo = norm.min
            adjustMin = newMin

        normData = adjustMin + 0.5*(newMax - newMin)*(data-normTo)/(norm.max - norm.min) 

        self.logScaler(logFile, norm)
        #logger.info(f"newmin: {np.min(normData)},  newmax: {np.max(normData)}")
        return normData, norm

    def logScale_Data(self, data, logFile):
        logger.info(f"Convert data to log scale | type: {type(data)}, shape: {data.shape}")

        logdata = np.log10(data)
        #unwrap the phase, this is computationaly expensive
        data = logdata + 2j * np.pi * np.floor(np.angle(data) / (2 * np.pi)) 

        max = np.max(np.abs(data))
        min = np.min(np.abs(data))

        logger.info(f"log scale | max: {max}, min: {min}")
        with open(logFile, 'a', newline='') as csvFile:
            writer = csv.writer(csvFile, dialect='unix')
            writer.writerow([f'--------- Convert Datq to log  -------'])
            writer.writerow(['min', 'max'])

    #def logScaler(self, logFile, scaler, name_a, name_b, data_a, data_b, scale=1):
    def logScaler(self, logFile, scaler:normClass, complex=False):
        # TODO:write min, man, std, mean, scail for everybody
        with open(logFile, 'a', newline='') as csvFile:
            writer = csv.writer(csvFile, dialect='unix')
            writer.writerow([f'--------- {scaler.type}, complex: {complex} -------'])
            writer.writerow(['min', 'max', 'mean', 'scale'])
            writer.writerow([scaler.min, scaler.max, scaler.mean, scaler.scale])
            writer.writerow(['---------'])
        #logger.info(f"Data: min: {scaler.min}, max: {scaler.max}, mean: {np.abs(scaler.mean)}, scale: {scaler.scale}")
    
    def getThisWindowData(self, dataumNumber, ch=0 ):
        # If ch is 0, then we want all the channels
        #logger.info(f"Getting data for wavelet tracking: {self.data_raw.shape}")

        if ch == 0:
            thisData = self.data_raw[dataumNumber]
        else:
            logger.info(f"dataConfigs.chList: {self.dataConfigs.chList}, ch: {ch}")
            chNumInList = self.dataConfigs.chList.index(ch)
            thisData = self.data_raw[dataumNumber][chNumInList]
        run = self.runList_raw[dataumNumber]
        timeWindow = self.startTimes_raw[dataumNumber]
        subjectLabel = self.subjectList_raw[dataumNumber]

        if ch == 0: ch = self.dataConfigs.chList
        #logger.info(f"thisData:{type(thisData)} {thisData.shape}, subjectLabel: {subjectLabel}, run: {run}, timeWindow: {timeWindow}, channel: {ch}")

        return thisData, run, timeWindow, subjectLabel
    
    def getFFTData(self, data):
        fftClass = jFFT_cl()
        #ch, datapoint
        nSamp = data.shape[1]
        freqList = fftClass.getFreqs(self.dataConfigs.sampleRate_hz, nSamp)
        fftData = np.zeros((len(self.dataConfigs.chList), len(freqList)))

        for ch, chData in enumerate(data):
            windowedData = fftClass.appWindow(chData, window="Hanning")
            freqData = fftClass.calcFFT(windowedData) #Mag, phase
            fftData[ch] = freqData[0]

        return freqList, fftData

    #TODO: move to cwt?
    def cwtTransformData(self, cwt_class, oneShot=True, saveNormPerams=False):
        # Can we transform the data in one shot? or dos this need a for loop?
        # Transform the RAW data. We do not actually have the data yet.
        timeData = self.data_raw
        logger.info(f"Starting transorm of data_raw (will apply norm later): {type(timeData)}, {timeData.shape}")
        timeStart = time.time()

        # Test the norm by loading the entire block and running the calcs there to compair
        # This takes too much mem for the entire set
        testCorrectness = self.configs['debugs']['testNormCorr']

        if saveNormPerams: 
            self.dataNormConst = normClass()
            self.dataNormConst.min = 10000
            self.dataNormConst.max = 0

        if oneShot:
            cwtData_raw , cwtFrequencies = cwt_class.cwtTransform(timeData) # This is: freqs, windows, ch, timepoints
        else:
            cwtData_raw = None
            cwtFrequencies = None
            sum = 0
            mean = 0
            variance = 0
            mean_Real = 0
            mean_Imag = 0
            variance_Real = 0
            variance_Imag = 0
            nElements = 0

            logger.info(f"Prossessing timeData: {len(timeData)} datapoints {timeData.shape}")
            for i, data in tqdm(enumerate(timeData), total=len(timeData), desc="Calculating CWT", unit="Transform"):

                #logger.info(f"Transforming data: {i}, {data.shape}")
                thisCwtData_raw, cwtFrequencies = cwt_class.cwtTransform(data)

                if saveNormPerams:
                    nElements += thisCwtData_raw.size
                    min = np.min(thisCwtData_raw)
                    max = np.max(thisCwtData_raw)
                    sum += np.sum(thisCwtData_raw)/thisCwtData_raw.size

                    if np.iscomplexobj(thisCwtData_raw):
                        real = np.real(thisCwtData_raw)
                        imag = np.imag(thisCwtData_raw)

                        #Each cwt mean
                        this_mean_real = np.mean(real)
                        this_mean_imag = np.mean(imag)

                        #The diff from the current mean
                        delta_real = this_mean_real - mean_Real
                        delta_imag = this_mean_imag - mean_Imag

                        # running mean
                        mean_Real += delta_real / (i+1)
                        mean_Imag += delta_imag / (i+1)

                        # Running variance
                        variance_Real += np.sum((real - mean_Real) **2)
                        variance_Imag += np.sum((imag - mean_Imag) **2)

                    else:
                        this_mean = np.mean(thisCwtData_raw)
                        delta = this_mean - mean
                        mean += delta/(i+1)
                        variance += np.sum((thisCwtData_raw - mean) **2)
                    #   mean
                    #   std dev

                    if min < self.dataNormConst.min: self.dataNormConst.min = min
                    if max > self.dataNormConst.max: self.dataNormConst.max = max
                    # std real
                    # std imag
                    #logger.info(f"#{i}: {self.dataNormConst.min}, max: {self.dataNormConst.max}, running Sum: {sum}")
                if testCorrectness:
                    thisCwtData_raw = np.expand_dims(thisCwtData_raw, axis=0) # add the run dim back to append
                    #logger.info(f"thisCwtData_raw: {type(thisCwtData_raw)}, {thisCwtData_raw.shape}")
                    if cwtData_raw is None: cwtData_raw = thisCwtData_raw.copy()
                    else:                        cwtData_raw = np.append(cwtData_raw, thisCwtData_raw, axis=0)


        #For testing to make sure we have the right thing
        if testCorrectness:
            cwtData_raw = np.transpose(cwtData_raw, (1, 2, 0, 3))           # we want: windows, ch, freqs, timepoints
            cwtTransformTime = time.time() - timeStart

        if saveNormPerams:
            if np.iscomplexobj(cwtData_raw):
                mean = mean_Real + 1j * mean_Imag
                std  = np.sqrt(variance_Real/nElements) + 1j * np.sqrt(variance_Imag/nElements) 
            else:
                std = np.sqrt(variance/nElements)
            self.dataNormConst.mean = mean
            self.dataNormConst.std = std

            #self.dataNormConst.mean = sum/(i+1)
            logger.info(f"Norm stats | min: {self.dataNormConst.min}, max: {self.dataNormConst.max}, mean: {self.dataNormConst.mean} ")
            logger.info(f"std dev | {self.dataNormConst.std}")
            if testCorrectness:
                logger.info(f"           | min: {np.min(cwtData_raw)}, max: {np.max(cwtData_raw)}, mean: {np.mean(cwtData_raw)}")
                logger.info(f"        | {np.std(np.real(cwtData_raw))}, + {np.std(np.imag(cwtData_raw))}i")


        if testCorrectness:
            logger.info(f"cwtData: {type(cwtData_raw)}, {cwtData_raw.shape}, cwtFrequencies: {type(cwtFrequencies)}, {cwtFrequencies.shape}, time: {cwtTransformTime:.2f}s")
        #if self.configs['cwt']['saveCWT']:
        #    logger.info(f"Saving cwt data: {self.dataSaveDir}/cwtData_{cwt_class.wavelet_name}.npy, {self.dataSaveDir}/cwtFrequencies_{cwt_class.wavelet_name}.npy")
        #    np.save(f"{self.dataSaveDir}/cwtData_{cwt_class.wavelet_name}.npy", cwtData_raw)
        #    np.save(f"{self.dataSaveDir}/cwtFrequencies_{cwt_class.wavelet_name}.npy", cwtFrequencies)


    def getNormPerams(self, cwt_class:cwt, logScaleData):
        logger.info(f"Get the norm/std peramiters | , wavelet_base: {cwt_class.wavelet_base}, wavelet_center_freq: {cwt_class.f0}, wavelet_bandwidth: {cwt_class.bw}, logScaleData: {logScaleData}")
        logger.info(f"Data dir: {self.dataSaveDir}")

        #logScaleFreq = configs['cwt']['logScaleFreq']
        #cwt_class.setupWavelet(cwt_class.wavelet_base, f0=cwt_class.f0, bw=cwt_class.bw, useLogForFreq=logScaleFreq)
        if logScaleData :cwt_class.normPeramsFileName = f"{cwt_class.normPeramsFileName}_logData"

        fileName = f"{self.dataSaveDir}/normPerams_{cwt_class.wavelet_name}.pkl"
        if os.path.isfile(fileName):
            logger.info(f"Loading norm/std perams from: {fileName}")
            with open(fileName, 'rb') as f:
                self.dataNormConst = pickle.load(f)
            logger.info(f"Loaded Norm stats from file | min: {self.dataNormConst.min}, max: {self.dataNormConst.max}, mean: {self.dataNormConst.mean} ")
        else: # Calculate the terms
            logger.info(f"Calculating norm/std perams")
            waveletPlotsDir = f"{self.dataSaveDir}/waveletPlots"
            cwt_class.plotWavelet(saveDir=waveletPlotsDir, sRate=self.dataConfigs.sampleRate_hz, save=True, show=False )
    
            # Transform the data one at a time to get the norm/std peramiters (e.x. min, max, mean, std)
            self.cwtTransformData(cwt_class=cwt_class, oneShot=False, saveNormPerams=True) 

            with open(fileName, 'wb') as f: pickle.dump(self.dataNormConst, f)
            logger.info(f"Saved norm/std peramiters to {fileName}")
