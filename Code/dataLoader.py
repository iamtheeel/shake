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
from tqdm import tqdm  #progress bar

import torch
import torch.nn.functional as tFun
from torch.utils.data import DataLoader, TensorDataset, random_split
import pickle
import time
from tqdm import tqdm  #progress bar

from utils import checkFor_CreateDir
from genPlots import *

from typing import Optional
if typing.TYPE_CHECKING: #Fix circular import
    from cwtTransform import cwt
    from fileStructure import fileStruct

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from dataclasses import dataclass
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
    def __init__(self, config, fileStruct:"fileStruct"):
        print(f"\n")
        logger.info(f"--------------------  Get Data   ----------------------")
        self.regression = config['model']['regression']
        self.seed = config['trainer']['seed'] 
        torch.manual_seed(self.seed)
        # Load up the dataset info
        self.inputData = config['data']['inputData']# Where the data is
        self.test = config['data']['test']         # e.x. "Test_2"
        self.valPercen = config['data']['valSplitPercen']
        self.batchSize = config['trainer']['batchSize']

        #TODO: Get from file
        self.logfile = f"{fileStruct.expTrackFiles.expTrackDir_name}/{fileStruct.expTrackFiles.expTrack_sumary_file}"

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

        self.dataNormConst = normClass()
        self.labNormConst = None

        self.configs = config

        #Set up a string for saving the dataset so we can see if we have already loaded this set
        self.fileStruct = fileStruct
        self.fileStruct.setData_dir(self.dataConfigs)

        # Setup the dataplotter
        self.dataPlotter = dataPlotter_class()

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


            speed =  self.getSpeedLabels(label_file_csv)
            #print(f"speeds: {speed}")

            if configs['plts']['generatePlots']:
                self.plotTime_FreqData(data=subjectData, freqYLim=2, subject=subjectNumber, speed=speed, folder="byRun")

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


        logger.info(f"Total Dataset: {data.shape}, Data min: {np.min(data)}, max: {np.max(data)}")

        timdDataFile_str = self.fileStruct.dataDirFiles.saveDataDir
        dataSaveDir_str = self.fileStruct.dataDirFiles.saveDataDir.saveDataDir_name
        np.save(f"{dataSaveDir_str}/{timdDataFile_str.timeDDataSave}", data)
        logger.info(f"Saved data to {dataSaveDir_str}/{timdDataFile_str.timeDDataSave}")
        torch.save(labels, f"{dataSaveDir_str}/labels.pt")
        logger.info(f"Saved labels to {dataSaveDir_str}/labels.pt")
        torch.save(subject_list, f"{dataSaveDir_str}/subjects.pt")
        torch.save(run_list, f"{dataSaveDir_str}/runs.pt")
        torch.save(startTimes_list, f"{dataSaveDir_str}/startTimes.pt")
        # Save data configs
        with open(f"{dataSaveDir_str}/dataConfigs.pkl", 'wb') as f:
            pickle.dump(self.dataConfigs, f)
        logger.info(f"Saved data configs to {dataSaveDir_str}/dataConfigs.pkl")

        self.data_raw = data
        self.labels_raw = labels.float()
        self.subjectList_raw = subject_list
        self.runList_raw = run_list
        self.startTimes_raw = startTimes_list
        logger.info(f"====================================================")

    def loadDataSet(self):
        timdDataFile_str = self.fileStruct.dataDirFiles.saveDataDir
        dataSaveDir_str = self.fileStruct.dataDirFiles.saveDataDir.saveDataDir_name
        logger.info(f"Loading data from {dataSaveDir_str}/{timdDataFile_str.timeDDataSave}")
        logger.info(f"Loading labels from {dataSaveDir_str}/labels.pt")
        logger.info(f"Loading subjects from {dataSaveDir_str}/subjects.pt")
        self.data_raw = np.load(f"{dataSaveDir_str}/{timdDataFile_str.timeDDataSave}")
        self.labels_raw = torch.load(f"{dataSaveDir_str}/labels.pt", weights_only=False).float()
        self.subjectList_raw = torch.load(f"{dataSaveDir_str}/subjects.pt", weights_only=False)
        self.runList_raw = torch.load(f"{dataSaveDir_str}/runs.pt", weights_only=False)
        self.startTimes_raw = torch.load(f"{dataSaveDir_str}/startTimes.pt", weights_only=False)

        with open(f"{dataSaveDir_str}/dataConfigs.pkl", 'rb') as f:
            #This will load the ch list from the saved file
            self.dataConfigs = pickle.load(f)
        logger.info(f"Loaded data configs from {dataSaveDir_str}/dataConfigs.pkl")

    def plotTime_FreqData(self, data, folder, freqYLim, subject=None, speed=None):
        logger.info(f" -----  Saving Plots of Full run subject: {subject} ---------")
        plotDirNames = self.fileStruct.dataDirFiles.plotDirNames 
        subFolder = self.fileStruct.dataDirFiles.saveDataDir.saveDataDir_name 
        subFolder = f"{subFolder}/{plotDirNames.baseDir}"
        timeDImageDir = f"{subFolder}/{folder}/{plotDirNames.time}"
        freqDImageDir = f"{subFolder}/{folder}/{plotDirNames.freq}"
        # Don't write for the windowed data if we already have it
        if subject == None and checkFor_CreateDir(timeDImageDir) == True:
            return #If the folder exists, we aleady have our plots
        logger.info(f"Writting files to: {timeDImageDir}")
        logger.info(f"                 : {freqDImageDir}")

        self.dataPlotter.generalConfigs(self.dataConfigs.sampleRate_hz)
        self.dataPlotter.configTimeD(timeDImageDir, configs['plts']['yLim_timeD'])
        self.dataPlotter.configFreqD(configs['plts']['yLim_freqD'])
        # Plot the data
        #for row in range(data.shape[0]):
        sTime = 0
        for row, dataRow in tqdm(enumerate(data), total=len(data), desc=f"Creating image set for {folder}", unit=" Image Set", leave=False):
            # subject, run, time
            # rms
            #thisLab = torch.argmax(labels[row])
            if subject != None: # This is by Run
                fileN = f"subject-{subject}_run-{row}"
                title = f"Domain subject:{subject}, run:{row}, speed:{speed[row]:.2f}"
            else:               # This is by wiundow
                thisSubject = self.classes[self.subjectList_raw[row]]
                thisRun = self.runList_raw[row]
                startTime = self.startTimes_raw[row]
                fileN = f"subject-{thisSubject}_run{thisRun}_time-{startTime}"
                title = f"subject:{thisSubject}, run: {thisRun}, time: {startTime}"
                #print(f"row: {row}, subject: {thisSubject}, run: {thisRun}, startTime: {startTime}")


            #print(f"data shape: {data[row].shape}, sampRate: {self.dataConfigs.sampleRate_hz}")

            #self.dataPlotter.plotInLineTime(data[row], fileN, f"Time {title}" )
            self.dataPlotter.plotInLineTime(dataRow, fileN, f"Time {title}" )
            self.dataPlotter.plotInLineFreq(freqDImageDir, dataRow, fileN, f"Freq {title}", xlim=[0, 10], show=False) #, subjectNumber, 0, "ByRun")
            self.dataPlotter.plotInLineFreq(freqDImageDir, dataRow, fileN, f"Freq {title}", xlim=[10, self.dataConfigs.sampleRate_hz/2],  xInLog=False, yLim = [0, freqYLim], show=False) #, subjectNumber, 0, "ByRun")
            self.dataPlotter.plotInLineFreq(freqDImageDir, dataRow, fileN, f"Freq {title}", xlim=[10, self.dataConfigs.sampleRate_hz/2],  xInLog=True, yLim = [0, freqYLim], show=False) #, subjectNumber, 0, "ByRun")

            #logger.info(f"Freq max: {self.dataPlotter.freqMax}")
            #plotRunFFT(subjectData, self.dataConfigs.sampleRate_hz, subjectNumber, 0, "ByRun")
            sTime += self.stepSize_s


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

        dataFile= f"{self.fileStruct.dataDirFiles.saveDataDir.saveDataDir_name}/{self.fileStruct.dataDirFiles.saveDataDir.timeDDataSumary}"
        with open(dataFile , 'a', newline='') as csvFile:
            csvFile.write('Subject, speed (m/s), run, startTime (s), dataPtsAfterStomp, label')
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
                dataPtsAfterStomp = -1 # -1: no stomp yet, then = #since stomp
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

                    # Look for stomp, and keeps track of how many windows since the stomp
                    # The detection of no step is done in getSubjecteLabel
                    if self.stompThresh == 0: nSkips = 0
                    else                    : nSkips = 3
                    thisSubjectId, dataPtsAfterStomp = self.findDataStart(dataPtsAfterStomp, rms_ratio, nSkips)
                    #print(f"stompThresh: {self.stompThresh}, nSkips: {nSkips}")

                    '''
                    ### for investigation
                    #plotThis = rms_ratio
                    plotThis = rms_allCh
                    try:              plot_run = np.append(plot_run, plotThis.reshape(-1, 1), axis=1)
                    except NameError: plot_run = plotThis.reshape(-1, 1)
                    '''
    
                    # append the data
                    thisStartTime = startPoint/self.dataConfigs.sampleRate_hz
                    if dataPtsAfterStomp >= nSkips:
                        # TODO: make this one data structure
                        thisSubjectId = self.getSubjectLabel(subject, rms_ratio) # Returns 0 for no step
                        #print(f"s: {subject} run: {run} t: {thisStartTime}, rmsRatio: {max(rms_ratio)}, label: {thisSubjectId}")
                        #print(f"rms_allCh = {rms_allCh}")
                        #print(f"rms_BaseLin = {rms_BaseLine}")
                        #print(f"rms_ratio = {rms_ratio}")

                        #print(f"this | subjectId: {thisSubjectId}, run:{run}, startTime: {thisStartTime}")
                        if (not self.regression) or (thisSubjectId > 0):
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
                            try:              runs = np.append(runs, run)
                            except NameError: runs = run
                            try:              startTimes = np.append(startTimes, thisStartTime)
                            except NameError: startTimes = thisStartTime

                    # Do we want to log the 0 vel data for regresion too? Yes
                    csvFile.write(f"{subject}, {speed[run]}, {run}, {thisStartTime}, {dataPtsAfterStomp}, {thisSubjectId}")
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

    def findDataStart(self, dataPtsAfterStomp, rms_ratio, nSkips):
        thisSubjectID = 0
        if dataPtsAfterStomp  < 0:
            if self.stompThresh == 0:
                dataPtsAfterStomp = nSkips
            else:
                for i in self.stompCh:
                    dataNum = self.dataConfigs.chList.index(i)
                    value = rms_ratio[dataNum]
                    #logger.info(f"ch: {i}, {dataNum}, rmsRatio: {value}, thresh: {self.stompThresh}")
                    if value > self.stompThresh: 
                        #thisSubjectID = -1
                        dataPtsAfterStomp = 0 
                        break
        else: dataPtsAfterStomp += 1 # Would probably be nice to just increment startPoint, but that makes another can of worms

        return thisSubjectID, dataPtsAfterStomp

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
        return(self.classes.index(subjectNumber))
        #if subjectNumber == '001': return 1
        #elif subjectNumber == '002': return 2
        #elif subjectNumber == '003': return 3
        #else: return 0

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
        
        trial_str = f"{self.inputData}/{self.test}/data/{trial_str}.hdf5"
        csv_str = f"{self.inputData}/{self.test}/{csv_str}.csv"

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
    def scale_data(self, data, log=False, norm:normClass=None, scaler=None, scale=None, debug=False):
        isTensor = False
        if isinstance(data, torch.Tensor): #convert to numpy
            data = data.numpy()
            isTensor = True

        if debug: 
            logger.info(f"scale_data: constants:{norm}")
            logger.info(f"Before scaling: min: {np.min(data)}, max: {np.max(data)}, shape: {data.shape}")
            #logger.info(f"{data[0:10, 0, 0]}")
            #logger.info(f"Complex data: {np.iscomplexobj(data)}")
        if scaler==None: scaler= norm.type

        if np.iscomplexobj(data):
            if scaler == "std": dataScaled, norm = self.std_complexData(data, log, norm, debug)
            else:               dataScaled, norm = self.norm_complexData(data, log, norm)
        else:
            if scaler == "std": dataScaled, norm = self.std_data(data, log, norm, debug)
            else:               dataScaled, norm = self.norm_data(data, log, norm, scale, debug)

        if isTensor: # and back to tensor
            dataScaled = torch.from_numpy(dataScaled)

        if debug: 
            logger.info(f"After scaling: min: {np.min(dataScaled)}, max: {np.max(dataScaled)}, {norm}")

        return dataScaled, norm

    def unScale_data(self, data, scalerClass, debug=False):
        #print(f"data: {type(data)}, {type(data[0])}, {len(data)}")
        #print(f" scalerClass: mean: {type(scalerClass.mean)}")
        data = np.array(data) # make numpy so we can work with it

        if np.iscomplexobj(data):
            if scalerClass.type == "std": data = self.unScale_std_complex(data, scalerClass, debug)
            else:                         data = self.unScale_norm_complex(data, scalerClass, debug)
        else:
            if scalerClass.type == "std": data = self.unScale_std(data, scalerClass, debug)
            else:                         data = self.unScale_norm(data, scalerClass, debug)

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

    def std_complexData(self, data, log, norm:normClass, debug):
        if debug:
            logger.info(f"std_complexData: {norm}")
        real = np.real(data)
        imag = np.imag(data)
        mean = np.mean(real) + 1j * np.mean(imag) 
        std = np.std(real) + 1j * np.std(imag) 
        self.dataNormConst = normClass(type="std", mean=mean, std=std)

        stdised_real = (real - norm.mean.real)/norm.std.real
        stdised_imag = (imag - norm.mean.imag)/norm.std.imag

        normData = stdised_real + 1j * stdised_imag

        if log:
            self.logScaler(self.logfile, self.dataNormConst, complex=True)
        #logger.info(f"newmin: {np.min(np.abs(normData))},  newmax: {np.max(np.abs(normData))}")
        if debug:
            logger.info(f"std_complexData done: {norm}")
        return normData, norm

    def std_data(self, data, log, norm:normClass, debug):
        # Normalize the data
        # float: from 0 to 1
        # 0.5 = center
        if norm == None:
            norm = normClass(type="std", mean=np.mean(data), std=np.std(data))

        # scale the data
        normData = (data - norm.mean)/norm.std # standardise

        if debug:
            logger.info(norm)
            #logger.info(f"Orig: \n{data[0:8]}")
            #logger.info(f"Norm: \n{normData[0:8]}")

        if log:
            self.logScaler(self.logfile, norm)
        #logger.info(f"newmin: {np.min(np.abs(normData))},  newmax: {np.max(np.abs(normData))}")
        return normData, norm

    def norm_complexData(self, data, log, norm:normClass):
        #L2 norm is frobenious_norm, saved as mean

        if norm == None:
            norm = normClass(type="norm", min=np.min(data), max=np.max(data), mean=np.linalg.norm(data) )

        normData = data/norm.mean

        if log:
            self.logScaler(self.logfile, norm)
        #logger.info(f"l2 norm: {self.dataNormConst.mean}, newmin: {np.min(normData)},  newmax: {np.max(normData)}")
        return normData, norm

    def norm_data(self, data, log, norm:normClass, scale, debug):
        #https://en.wikipedia.org/wiki/Feature_scaling
        if norm == None:
            norm = normClass(type="norm", min=np.min(data), max=np.max(data), mean=np.mean(data), scale=scale)

        if norm.type == 'meanNorm': 
            normTo = norm.mean
        elif norm.type == 'minMaxNorm': 
            normTo = norm.min


        normData = (data-normTo)/(norm.max - norm.min) 

        if debug:
            logger.info(f"norm_data:{data},  normTo: {normTo}, max: {norm.max}, min: {norm.min} | normData: {normData}")

        #Rescale for min/max
        if norm.scale!= 1:
            newMin = -norm.scale
            newMax = norm.scale
            normData = newMin + (newMin - newMax)*normData

        if debug:
            logger.info(f"After rescale: {normData}")

        if log:
            self.logScaler(self.logfile, norm)
        #logger.info(f"newmin: {np.min(normData)},  newmax: {np.max(normData)}")
        return normData, norm

    '''
    def logScale_Data(self, data ):
        logger.info(f"Convert data to log scale | type: {type(data)}, shape: {data.shape}")

        logdata = np.log10(data)
        #unwrap the phase, this is computationaly expensive
        data = logdata + 2j * np.pi * np.floor(np.angle(data) / (2 * np.pi)) 

        max = np.max(np.abs(data))
        min = np.min(np.abs(data))

        logger.info(f"log scale | max: {max}, min: {min}")
        with open(self.logfile, 'a', newline='') as csvFile:
            writer = csv.writer(csvFile, dialect='unix')
            writer.writerow([f'--------- Convert Datq to log  -------'])
            writer.writerow(['min', 'max'])
    '''

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

    def plotDataByWindow(self, cwt_class:"cwt", logScaleData:bool):
        # Plot the windowed data
        generatePlots = self.configs['plts']['generatePlots']
        if generatePlots:
            subFolder =self.fileStruct.dataDirFiles.saveDataDir.waveletDir.waveletDir_name
            if configs['cwt']['doCWT']:
                timeFFTCWT_dir= f"{subFolder}/{self.fileStruct.dataDirFiles.plotDirNames.time_fft_cwt}"
                if checkFor_CreateDir(timeFFTCWT_dir) == False:
                    dataPlotter = saveCWT_Time_FFT_images(data_preparation=self, cwt_class=cwt_class, expDir=timeFFTCWT_dir)
                    dataPlotter.generateAndSaveImages(logScaleData)

            self.plotTime_FreqData(data=self.data_raw, freqYLim=0.5, folder="byWindow")


    #TODO: Move to cwt
    def calculateCWTDataNormTerms(self, cwt_class:"cwt", oneShot=True, saveNormPerams=False):
        # Can we transform the data in one shot? or dos this need a for loop?
        # Transform the RAW data. We do not actually have the data yet.
        timeData = self.data_raw
        logger.info(f"Starting transorm of data_raw (will apply norm later): {type(timeData)}, {timeData.shape}")
        timeStart = time.time()

        # Test the norm by loading the entire block and running the calcs there to compair
        # This takes too much mem for the entire set
        testCorrectness = self.configs['debugs']['testNormCorr']

        if saveNormPerams: 
            #self.dataNormConst = normClass() #This is bug, like 99% sure, but late to class
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

                    if np.iscomplexobj(cwt_class.wavelet_fun):
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
            if np.iscomplexobj(cwt_class.wavelet_fun):
                mean = mean_Real + 1j * mean_Imag
                std  = np.sqrt(variance_Real/nElements) + 1j * np.sqrt(variance_Imag/nElements) 
            else:
                print("NOT COMPLEX")
                std = np.sqrt(variance/nElements)
            self.dataNormConst.mean = mean
            self.dataNormConst.std = std

            #self.dataNormConst.mean = sum/(i+1)
            logger.info(f"Norm stats: {self.dataNormConst}")
            logger.info(f"std dev | {self.dataNormConst.std}")
            if testCorrectness:
                logger.info(f"           | min: {np.min(cwtData_raw)}, max: {np.max(cwtData_raw)}, mean: {np.mean(cwtData_raw)}")
                logger.info(f"        | {np.std(np.real(cwtData_raw))}, + {np.std(np.imag(cwtData_raw))}i")


        if testCorrectness:
            logger.info(f"cwtData: {type(cwtData_raw)}, {cwtData_raw.shape}, cwtFrequencies: {type(cwtFrequencies)}, {cwtFrequencies.shape}, time: {cwtTransformTime:.2f}s")
    def calculateTimeDNormTerms(self):
        logger.info(f"Generate Time D norm from self.data_raw: {self.data_raw.shape}")
        self.dataNormConst.min = np.min(self.data_raw)
        self.dataNormConst.max = np.max(self.data_raw)
        self.dataNormConst.mean = np.mean(self.data_raw)
        self.dataNormConst.std = np.std(self.data_raw)
        logger.info(f"{self.dataNormConst}")

    def getNormPerams(self, cwt_class:"cwt", logScaleData, dataScaler, dataScale_value):
        print(f"\n")
        logger.info(f" -------------- Get the norm/std peramiters | logScaleData: {logScaleData}   ---------------")
        self.dataNormConst.type = dataScaler
        self.dataNormConst.scale = dataScale_value

        dataNormDir = self.fileStruct.setDataNorm_dir(self.dataNormConst, logScaleData)

        dataNormDir = self.fileStruct.dataDirFiles.saveDataDir.waveletDir.dataNormDir
        fileName = f"{dataNormDir.dataNormDir_name}/{dataNormDir.normPeramsFile_name}"
        logger.info(f"Looking for: {fileName}")
        if os.path.isfile(fileName):
            logger.info(f"Loading norm/std perams from: {fileName}")
            with open(fileName, 'rb') as f:
                self.dataNormConst = pickle.load(f)
            #logger.info(f"Loaded Norm stats from file | min: {self.dataNormConst.min}, max: {self.dataNormConst.max}, mean: {self.dataNormConst.mean}, std: {self.dataNormConst.std} ")
            logger.info(f"Loaded Norm stats from file: {self.dataNormConst}")

            ###### Fudge
            #self.dataNormConst.type = "std"
            #with open(fileName, 'wb') as f: pickle.dump(self.dataNormConst, f)
            ######
        else: # Calculate the terms
            #min, max, mean, std
            logger.info(f"Calculating norm/std perams")
            if configs['cwt']['doCWT']:
                # Transform the data one at a time to get the norm/std peramiters (e.x. min, max, mean, std)
                self.calculateCWTDataNormTerms(cwt_class=cwt_class, oneShot=False, saveNormPerams=True ) 
            else: self.calculateTimeDNormTerms()


            with open(fileName, 'wb') as f: pickle.dump(self.dataNormConst, f)
            logger.info(f"Saved norm/std peramiters to {fileName}")
