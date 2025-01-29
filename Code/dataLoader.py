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
import pickle
import time

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

class dataConfigs:
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
        self.data_norm = None
        self.labels_norm = None
        self.dataLoader_t = None
        self.dataLoader_v = None

        self.dataNormConst = None
        self.labNormConst = None

        self.dataSaveDir = f"{self.dataPath}/{config['data']['dataSetDir']}"

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
        if not configs['data']['dataSetDir'] == "":
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


    def resetData(self, wavelet_name):
        if wavelet_name == "None":
            # Add the "ch"
            # Data is currently: datapoints, height(sensorch), width(datapoints)
            self.data = copy.deepcopy(self.data_raw) #Data is numpy
            self.data = self.data.unsqueeze(1) # datapoints, image channels, height, width 
        else:
            # datapoints, image channels, height, width 
            self.data = copy.deepcopy(self.cwtData_raw) #cwtData is numpy
        self.labels = self.labels_raw.clone().detach() #labels are tensor
        #self.cwtData = copy.deepcopy(self.cwtData_raw) #cwtData is numpy
    

    def createDataloaders(self, expNum):
        self.data_norm = torch.tensor(self.data_norm, dtype=torch.float32) # dataloader wants a torch tensor

        logger.info(f"labels: {self.labels_norm.shape}")
        logger.info(f"subjects: {type(self.subjectList_raw)}, {self.subjectList_raw.shape}")
        dataSet = TensorDataset(self.data_norm, self.labels_norm, self.subjectList_raw)
        #dataSet = TensorDataset(self.data_norm, self.labels_norm, self.subjects)
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

        dataFile= f"{self.dataDir}/data.csv"
        with open(dataFile , 'a', newline='') as csvFile:
            csvFile.write('Subject, speed (m/s), run, startTime (s), label')
            for i in range(data.shape[1]):
                thisCh = self.dataConfigs.chList[i]
                csvFile.write(f", sensor {thisCh} rms")
            csvFile.write(f", startTime(s)")
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
                        thisDataBlock = np.expand_dims(thisDataBlock, axis=0) # add the run dim back to append
                        try:              windowedData = np.append(windowedData, thisDataBlock, axis=0) # append on trials, now trials/windows
                        except NameError: windowedData = thisDataBlock


                        if thisSubjectId >=0: # Negitives reservered 
                            thisSubjectId = self.getSubjectLabel(subject, rms_ratio) 
                            if self.regression: thisLabel = speed[run]
                            else:               thisLabel = thisSubjectId

                        # Append the data, labes, and all that junk
                        try:              labels = np.append(labels, thisLabel)
                        except NameError: labels = thisLabel
                        thisSubjectNumber = self.getSubjectID(subject) #Keep track of the subject number appart from the label
                        try:              subjects = np.append(subjects, thisSubjectNumber)
                        except NameError: subjects = thisSubjectNumber
                        try:              runs = np.append(runs, run+1)
                        except NameError: runs = run+1
                        thisStartTime = startPoint/self.dataConfigs.sampleRate_hz
                        try:              startTimes = np.append(startTimes, thisStartTime)
                        except NameError: startTimes = thisStartTime
    
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


    def scale_data(self, data, scaler, logFile, scale):
        isTensor = False
        if isinstance(data, torch.Tensor): #convert to numpy
            data = data.numpy()
            isTensor = True

        if scaler == "std": dataScaled, scalerClass = self.std_data(data, logFile)
        else:               dataScaled, scalerClass = self.norm_data(data, logFile, scaler, scale)

        scalerClass.type = scaler

        if isTensor: # and back to tensor
            dataScaled = torch.from_numpy(dataScaled)

        return dataScaled, scalerClass

    def unScale_data(self, data, scalerClass):
        #print(f"data: {type(data)}, {type(data[0])}, {len(data)}")
        #print(f" scalerClass: mean: {type(scalerClass.mean)}")
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

    def std_data(self, data, logFile):
        # Normalize the data
        # float: from 0 to 1
        # 0.5 = center

        norm = normClass(type="std", mean=np.mean(data), std=np.std(data))
        #logger.info(f"Orig: {data[0:3, 0:5, 0:2]}")

        # scale the data
        normData = (data - norm.mean)/norm.std # standardise

        self.logScaler(logFile, norm)
        logger.info(f"newmin: {np.abs(np.min(normData))},  newmax: {np.abs(np.max(normData))}")
        return normData, norm

    def norm_data(self, data, logFile, scaler, scale):
        norm = normClass(type="norm", min=np.min(data), max=np.max(data), mean=np.mean(data), scale=scale)
        newMin = -norm.scale
        newMax = norm.scale

        if scaler == 'meanNorm': 
            normTo = norm.mean
            adjustMin = 0
        elif scaler == 'minMaxNorm': 
            normTo = norm.min
            adjustMin = newMin

        normData = adjustMin + (newMax - newMin)*(data-normTo)/(norm.max - norm.min) 

        self.logScaler(logFile, norm)
        logger.info(f"newmin: {np.min(normData)},  newmax: {np.max(normData)}")
        return normData, norm

    #def logScaler(self, logFile, scaler, name_a, name_b, data_a, data_b, scale=1):
    def logScaler(self, logFile, scaler ):
        # TODO:write min, man, std, mean, scail for everybody
        with open(logFile, 'a', newline='') as csvFile:
            writer = csv.writer(csvFile, dialect='unix')
            writer.writerow([f'--------- {scaler.type}  -------'])
            writer.writerow(['min', 'max', 'mean', 'scale'])
            writer.writerow([scaler.min, scaler.max, scaler.mean, scaler.scale])
            writer.writerow(['---------'])
        logger.info(f"Data: min: {scaler.min}, max: {scaler.max}, mean: {np.abs(scaler.mean)}, scale: {scaler.scale}")
    
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
    def getCWTData(self, cwt_class):
        cwtFile = f"{self.dataSaveDir}/cwtData_{cwt_class.wavelet_name}.npy"
        if self.dataSaveDir != "" and os.path.exists(cwtFile):
            logger.info(f"loading: {self.dataSaveDir}, cwtFile: {cwtFile}")
            self.loadCWTData(cwt_class)
        else: 
            self.cwtTransofrmData(cwt_class)

    def loadCWTData(self, cwt_class):
        self.cwtData_raw = np.load(f"{self.dataSaveDir}/cwtData_{cwt_class.wavelet_name}.npy")
        self.cwtFrequencies = np.load(f"{self.dataSaveDir}/cwtFrequencies_{cwt_class.wavelet_name}.npy")

    def cwtTransofrmData(self, cwt_class):
        # Can we transform the data in one shot? or dos this need a for loop?
        # Transform the RAW data. We do not actually have the data yet.
        timeData = self.data_raw
        logger.info(f"Starting transorm of data_raw: {type(timeData)}, {timeData.shape}")
        timeStart = time.time()
        self.cwtData_raw , self.cwtFrequencies = cwt_class.cwtTransform(timeData) # This is: freqs, windows, ch, timepoints
        self.cwtData_raw = np.transpose(self.cwtData_raw, (1, 2, 0, 3))           # we want: windows, ch, freqs, timepoints
        '''
        self.cwtData_raw = None
        self.cwtFrequencies = None

        #TODO: This is a for loop, but we can do it in one shot
        for i, data in enumerate(timeData):
            #logger.info(f"Transforming data: {i}, {data.shape}")
            cwtData_raw, cwtFrequencies = cwt_class.cwtTransform(data)
            logger.info(f"cwtData_raw: {type(cwtData_raw)}, {cwtData_raw.shape}, cwtFrequencies: {type(cwtFrequencies)}, {cwtFrequencies.shape}")

            cwtData_raw = np.expand_dims(cwtData_raw, axis=0) # add the run dim back to append
            print(f"cwtData_raw: {type(cwtData_raw)}, {cwtData_raw.shape}")
            if self.cwtData_raw is None:
                self.cwtData_raw = cwtData_raw.copy()
                #logger.info(f"created self.cwtData_rawf.: {self.cwtData_raw.shape}")
            else:
                self.cwtData_raw = np.append(self.cwtData_raw, cwtData_raw, axis=0)
                #logger.info(f"appended: {self.cwtData_raw.shape}")

            #Everybody is the same freq list
            if self.cwtFrequencies is None: self.cwtFrequencies = cwtFrequencies


            logger.info(f"Data Number {i}: self.cwtData_raw: {self.cwtData_raw.shape}, self.cwtFrequencies: {self.cwtFrequencies.shape}")
        '''


        cwtTransformTime = time.time() - timeStart

        logger.info(f"cwtData: {type(self.cwtData_raw)}, {self.cwtData_raw.shape}, cwtFrequencies: {type(self.cwtFrequencies)}, {self.cwtFrequencies.shape}, time: {cwtTransformTime}s")
        logger.info(f"Saving cwt data: {self.dataSaveDir}/cwtData_{cwt_class.wavelet_name}.npy, {self.dataSaveDir}/cwtFrequencies_{cwt_class.wavelet_name}.npy")
        np.save(f"{self.dataSaveDir}/cwtData_{cwt_class.wavelet_name}.npy", self.cwtData_raw)
        np.save(f"{self.dataSaveDir}/cwtFrequencies_{cwt_class.wavelet_name}.npy", self.cwtFrequencies)
        #torch.save(labels, f"{self.dataSaveDir}/labels.pt")
