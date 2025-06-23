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
from scipy.signal import spectrogram    # For spectrogram

import torch
import torch.nn.functional as tFun
from torch.utils.data import DataLoader, Dataset, TensorDataset, random_split
import pickle
import time

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
    sampleRate_hz: Optional[int] = None
    origSRate_hz: Optional[int] = None
    units: Optional[str] = None
    dataLen_pts: Optional[int] = None
    origDataLen_pts: Optional[int] = None
    nSensors: Optional[int] = None
    nTrials: Optional[int] = None
    chList: Optional[list] = None

class HDF5Dataset(Dataset):
    def __init__(self, file_path):
        self.file_path = file_path
        self.h5_file = None  # Will be opened in `__getitem__`
        self._init_file()

        # Open file to get dataset size
        f = self.h5_file
        self.length = len(f["data"])  # Number of samples
        self.shape = f["data"].shape

        #Stats
        self.data_min = f["data"].attrs["min"]
        self.data_max = f["data"].attrs["max"]
        self.data_mean = f["data"].attrs["mean"]
        self.data_std = f["data"].attrs["std"]
        self.lab_min = f["labelsSpeed"].attrs["min"]
        self.lab_max = f["labelsSpeed"].attrs["max"]
        self.lab_mean = f["labelsSpeed"].attrs["mean"]
        self.lab_std = f["labelsSpeed"].attrs["std"]

    def __len__(self):
        return self.length

    def _init_file(self):
        if self.h5_file is None:
            self.h5_file = h5py.File(self.file_path, "r")

    def __getitem__(self, idx):
        self._init_file()
        f  = self.h5_file
        #with h5py.File(self.file_path, "r") as f:
        data = torch.tensor(f["data"][idx], dtype=torch.float32)
        label_speed = torch.tensor(f["labelsSpeed"][idx], dtype=torch.float32)
        label_subject = torch.tensor(f["labelsSubject"][idx], dtype=torch.long)
        subject = torch.tensor(f["subjects"][idx], dtype=torch.long)
        run = torch.tensor(f["runs"][idx], dtype=torch.long)
        sTime = torch.tensor(f["sTimes"][idx], dtype=torch.float16)

        #print(f"Label Shape after Squeeze: {label.shape}")  # Debugging step
        return data, label_speed, label_subject, subject, run, sTime
    
    def getConfig(self):
        """Retrieve and return general configurations stored in HDF5 attributes."""
        config = {}
        with h5py.File(self.file_path, "r") as f:
            for key, value in f.attrs.items():  # Read all top-level attributes
                config[key] = value
        return config

class dataLoader:
    def __init__(self, config, fileStruct:"fileStruct", device):
        print(f"\n")
        logger.info(f"--------------------  Get Data   ----------------------")
        self.device = device
        self.regression = config['model']['regression']
        self.seed = config['trainer']['seed'] 
        torch.manual_seed(self.seed)
        # Load up the dataset info
        self.inputData = config['data']['inputData']# Where the data is
        self.downSample = config['data']['downSample']# Where the data is
        self.test = config['data']['test']         # e.x. "Test_2"
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
        self.stompFromFile = False
        if isinstance(self.stompThresh, str):
            self.stompFromFile = True
            self.stompTimes_np = self.getStompTimes(f"{config['data']['inputData']}/{self.stompThresh}")

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

        # These all go away
        self.data_raw = None
        self.labels_raw = None
        self.subjectList_raw = None
        self.runList_raw = None
        self.startTimes_raw = None

        self.data = None
        self.labels = None

        self.timeDDataSet = None
        self.CWTDataSet = None
        self.dataLoader_t = None
        self.dataLoader_v = None

        self.dataNormConst = normClass()
        self.labNormConst = normClass()

        self.configs = config

        #Set up a string for saving the dataset so we can see if we have already loaded this set
        self.fileStruct = fileStruct
        self.fileStruct.setFullData_dir()
        self.fileStruct.setWindowedData_dir()

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
        dataLoadFieldNames = ['subject', 'data file', 'label file', 'Original Data Rate (Hz)', 'Original dataPoints', 'DS Data Rate (Hz)', 'DS dataPoints', 'nSensors in data', 'nTrials in data']
        with open(self.logfile, 'a', newline='') as csvFile:
            writer = csv.writer(csvFile, dialect='unix')
            writer.writerow(['--------- Data Loader -----------'])
            writer.writerow(['--------- Data Files -----------'])
        with open(self.logfile, 'a', newline='') as csvFile:
            writer = csv.DictWriter(csvFile, fieldnames=dataLoadFieldNames, dialect='unix')
            writer.writeheader()

        data_list = []
        speed_label_list = []
        subject_label_list = []
        subject_list = []
        run_list = []
        sTime_list = []

        self.subjects = self.getSubjects()
        for subjectName in self.subjects:
            data_file_hdf5, label_file_csv = self.getFileName(subjectName)
            logger.info(f"Dataloader, datafile: {data_file_hdf5}")

            # Load data file
            subjectData = self.getSubjectData(data_file_hdf5) # Only the chans we are interested, in the order we want. Sets the sample rate and friends
            subDataShape = np.shape(subjectData) 
            logger.info(f"Subject: {subjectName}, subject shape: {np.shape(subjectData)}")


            speed =  self.getSpeedLabels(label_file_csv)
            #print(f"speeds: {speed}")

            if configs['debugs']['generateTimeFreqPlots']:
                yLim = configs['plts']['yLim_freqD']
                self.plotTime_FreqData(data=subjectData, freqYLim=yLim, subject=subjectName, speed=speed, folder=f"../FullRunPlots/Subject-{subjectName}", fromRaw=True)

            # Window the data
            windowedBlock, labelBlock_speed, labelBlock_subject, subjectBlock, runBlock, startTimes = self.windowData(data=subjectData, subject=subjectName, speed=speed)
            logger.info(f"label speed {type(labelBlock_speed)}: {labelBlock_speed.shape}")
            logger.info(f"label subject {type(labelBlock_subject)}: {labelBlock_subject.shape}")
            logger.info(f"data: min:{np.min(windowedBlock)}, max: {np.max(windowedBlock)}, mean: {np.mean(windowedBlock)}, std: {np.std(windowedBlock)}")


            # Append the data to the set
            data_list.append(windowedBlock)
            speed_label_list.append(labelBlock_speed)
            subject_label_list.append(labelBlock_subject)
            subject_list.append(subjectBlock)
            run_list.append(runBlock)
            sTime_list.append(startTimes)

            #logger.info(f"Labels: {thisSubLabels}")
            #logger.info(f"Up to: {subjectName}")#, Labels, data shapes: {speed_label_list.shape}, {subject_label_list.shape} ")

            with open(self.logfile, 'a', newline='') as csvFile:
                writer = csv.writer(csvFile, dialect='unix')
                writer.writerow([f'', '--------- Load Data From Files '])
            with open(self.logfile, 'a', newline='') as csvFile:
                writer = csv.DictWriter(csvFile, fieldnames=dataLoadFieldNames, dialect='unix')
                writer.writerow({'subject': subjectName,
                                'data file': data_file_hdf5, 
                                'label file': label_file_csv, 
                                'Original Data Rate (Hz)': self.dataConfigs.origSRate_hz, 
                                'Original dataPoints': self.dataConfigs.origDataLen_pts,
                                'DS Data Rate (Hz)': self.dataConfigs.sampleRate_hz, 
                                'DS dataPoints': subDataShape[2],
                                'nSensors in data': subDataShape[1], 
                                'nTrials in data': subDataShape[0]
                                })

        ## Convert our lists to numpys
        data_np = np.vstack(data_list) # (datapoints, ch, timepoints)
        labelsSpeed_np = np.concatenate(speed_label_list, axis=0) # datapoints
        labelsSubject_np = np.concatenate(subject_label_list, axis=0) # datapoints
        subjects_np = np.concatenate(subject_list, axis=0)
        runs_np = np.concatenate(run_list, axis=0)
        sTimes_np = np.concatenate(sTime_list, axis=0)

        #Reshape the speed labels for batch processing
        labelsSpeed_np = labelsSpeed_np.reshape(-1,1) #go from (num,) to (num,1)
        #labelsSubject_np = labelsSubject_np.reshape(-1,1) #go from (num,) to (num,1)

        data_min = np.min(data_np)
        data_max = np.max(data_np)
        data_mean = np.mean(data_np)
        data_std = np.std(data_np)

        lab_min = np.min(labelsSpeed_np)
        lab_max = np.max(labelsSpeed_np)
        lab_mean = np.mean(labelsSpeed_np)
        lab_std = np.std(labelsSpeed_np)

        logger.info(f"Dataset: {data_np.shape}, labels Speed: {labelsSpeed_np.shape}")
        logger.info(f"Data min: {data_min}, max: {data_max}, mean: {data_mean}, std: {data_std}")
        logger.info(f"Label min: {lab_min}, max: {lab_max}, mean: {lab_mean}, std: {lab_std}")

        self.setNormConst(isData=True, norm=self.dataNormConst, dataSetFile="Original Time Domain Data", 
                          min=data_min, max=data_max, mean=data_mean, std=data_std)
        self.setNormConst(isData=False, norm=self.labNormConst, dataSetFile="Original Time Domain Data", 
                          min=lab_min, max=lab_max, mean=lab_mean, std=lab_std)
        timdDataFile_str = self.fileStruct.dataDirFiles.saveDataDir
        dataSaveDir_str = self.fileStruct.dataDirFiles.saveDataDir.saveDataDir_name
        timeDFileName = f"{dataSaveDir_str}/{timdDataFile_str.timeDData_file}"
        self.saveHDF5TimeData(timeDFileName, data_np, labelsSpeed_np, labelsSubject_np, subjects_np, runs_np, sTimes_np)
        self.saveHDF5MetaData(timeDFileName)

        logger.info(f"====================================================")

    def saveHDF5TimeData(self, filename, data_np, labelsSpeed_np, labelsSubject_np, subjects_np, runs_np, sTimes_np):
        with h5py.File(filename, "w") as h5dataFile:
            #TODO: add sameple rate, etc
            h5dataFile.create_dataset("data", data=data_np)
            h5dataFile.create_dataset("labelsSpeed", data=labelsSpeed_np)
            h5dataFile.create_dataset("labelsSubject", data=labelsSubject_np)
            h5dataFile.create_dataset("subjects", data=subjects_np)
            h5dataFile.create_dataset("runs", data=runs_np)
            h5dataFile.create_dataset("sTimes", data=sTimes_np)

    def saveHDF5MetaData(self, hd5File):
        logger.info(f"Saveing MetaData to File: {hd5File}")
        logger.info(f"data stats: {self.dataNormConst}")
        logger.info(f"label stats: {self.labNormConst}")
        with h5py.File(hd5File, "a") as h5dataFile:
            data_ds = h5dataFile["data"]
            label_ds = h5dataFile["labelsSpeed"]
            # Store statistics as attributes
            data_ds.attrs["min"] = self.dataNormConst.min
            data_ds.attrs["max"] = self.dataNormConst.max
            data_ds.attrs["mean"] = self.dataNormConst.max
            data_ds.attrs["std"] = self.dataNormConst.std
            label_ds.attrs["min"] = self.labNormConst.min
            label_ds.attrs["max"] = self.labNormConst.max
            label_ds.attrs["mean"] = self.labNormConst.mean
            label_ds.attrs["std"] = self.labNormConst.std

            h5dataFile.attrs["orig_sample_rate"] = self.dataConfigs.origSRate_hz
            h5dataFile.attrs["sample_rate"] = self.dataConfigs.sampleRate_hz
            h5dataFile.attrs["units"] = self.dataConfigs.units
            h5dataFile.attrs["orig_dataLen_pts"] = self.dataConfigs.origDataLen_pts
            h5dataFile.attrs["dataLen_pts"] = self.dataConfigs.dataLen_pts
            h5dataFile.attrs["nSensors"] = self.dataConfigs.nSensors
            h5dataFile.attrs["nTrials"] = self.dataConfigs.nTrials
            h5dataFile.attrs["chList"] = self.dataConfigs.chList

    import h5py

    '''
    Load the data by batch
    with h5py.File(f"{dataSaveDir_str}/{timdDataFile_str.timeDData_file}", "w") as h5dataFile:
        # Create resizable datasets
        data_ds = h5dataFile.create_dataset("data", shape=(0, *data_np.shape[1:]), maxshape=(None, *data_np.shape[1:]), dtype="float32", chunks=True)
        label_speed_ds = h5dataFile.create_dataset("labelsSpeed", shape=(0,), maxshape=(None,), dtype="float32", chunks=True)
        label_subject_ds = h5dataFile.create_dataset("labelsSubject", shape=(0,), maxshape=(None,), dtype="int32", chunks=True)
        subjects_ds = h5dataFile.create_dataset("subjects", shape=(0,), maxshape=(None,), dtype="int32", chunks=True)
        runs_ds = h5dataFile.create_dataset("runs", shape=(0,), maxshape=(None,), dtype="int32", chunks=True)
        sTimes_ds = h5dataFile.create_dataset("sTimes", shape=(0,), maxshape=(None,), dtype="float16", chunks=True)
    
        # Write data in batches
        for i in range(0, len(data_np), batch_size):
            batch_data = data_np[i : i + batch_size]
            batch_labels_speed = labelsSpeed_np[i : i + batch_size]
            batch_labels_subject = labelsSubject_np[i : i + batch_size]
            batch_subjects = subjects_np[i : i + batch_size]
            batch_runs = runs_np[i : i + batch_size]
            batch_sTimes = sTimes_np[i : i + batch_size]
    
            # Resize datasets before appending new batch
            data_ds.resize(data_ds.shape[0] + batch_data.shape[0], axis=0)
            label_speed_ds.resize(label_speed_ds.shape[0] + batch_labels_speed.shape[0], axis=0)
            label_subject_ds.resize(label_subject_ds.shape[0] + batch_labels_subject.shape[0], axis=0)
            subjects_ds.resize(subjects_ds.shape[0] + batch_subjects.shape[0], axis=0)
            runs_ds.resize(runs_ds.shape[0] + batch_runs.shape[0], axis=0)
            sTimes_ds.resize(sTimes_ds.shape[0] + batch_sTimes.shape[0], axis=0)
    
            # Append batch
            data_ds[-batch_data.shape[0] :] = batch_data
            label_speed_ds[-batch_labels_speed.shape[0] :] = batch_labels_speed
            label_subject_ds[-batch_labels_subject.shape[0] :] = batch_labels_subject
            subjects_ds[-batch_subjects.shape[0] :] = batch_subjects
            runs_ds[-batch_runs.shape[0] :] = batch_runs
            sTimes_ds[-batch_sTimes.shape[0] :] = batch_sTimes

        # Store metadata as attributes
        h5dataFile.attrs["sample_rate"] = self.dataConfigs.sampleRate_hz
        h5dataFile.attrs["units"] = self.dataConfigs.units
        h5dataFile.attrs["dataLen_pts"] = self.dataConfigs.dataLen_pts
        h5dataFile.attrs["nSensors"] = self.dataConfigs.nSensors
        h5dataFile.attrs["nTrials"] = self.dataConfigs.nTrials
        h5dataFile.attrs["chList"] = self.dataConfigs.chList
    '''

    def plotTime_FreqData(self, data, folder, freqYLim, subject=None, speed=None, fromRaw=False):
        if subject == None:
            echoStr = f"Time Windowed Data"
        else:
            echoStr = f"Full run, subject: {subject}"
        logger.info(f" -----  Saving Plots of {echoStr} ---------")
        plotDirNames = self.fileStruct.dataDirFiles.plotDirNames 
        subFolder = self.fileStruct.dataDirFiles.saveDataDir.saveDataDir_name 
        #subFolder = f"{subFolder}/{plotDirNames.baseDir}"
        timeDImageDir = f"{subFolder}/{folder}/{plotDirNames.time}"
        freqDImageDir = f"{subFolder}/{folder}/{plotDirNames.freq}"
        # Don't write for the windowed data if we already have it
        if checkFor_CreateDir(timeDImageDir) == True:
        #if subject == None and checkFor_CreateDir(timeDImageDir) == True:
            #logger.info(f"File exists, exiing")
            #TODO: This does not work cuz we run this 3 times with each subject
            return #If the folder exists, we aleady have our plots
        logger.info(f"Writting files to: {timeDImageDir}")
        logger.info(f"                 : {freqDImageDir}")

        self.dataPlotter.generalConfigs(self.dataConfigs.sampleRate_hz)
        self.dataPlotter.configTimeD(timeDImageDir, configs['plts']['yLim_timeD'])
        self.dataPlotter.configFreqD(configs['plts']['yLim_freqD'])
        # Plot the data
        #for row in range(data.shape[0]):
        if fromRaw:
            self.plotFromRaw(folder=folder, freqYLim=freqYLim, data=data, subject=subject, speed=speed, freqDImageDir=freqDImageDir)
        else: 
            self.plotFromDataSet(freqDImageDir=freqDImageDir, dataSet=self.timeDDataSet, freqYLim=freqYLim)


    def plotFromDataSet(self, freqDImageDir, dataSet, freqYLim):
        desc_str = f"Creating image set for {freqDImageDir}" 
        logger.info(desc_str)
        for row, (data_tensor, label_speed, label_subject, subject, run, startTime) in tqdm(enumerate(dataSet), total= len(dataSet), desc=desc_str, unit="Image Set" ):
            speed = label_speed.item()
            thisSubject = self.classes[subject]
            data_np = data_tensor.numpy()
            #thisRun = self.runList_raw[row]
            #startTime = self.startTimes_raw[row]
            fileN = f"subject-{thisSubject}_run{run}_time-{startTime}"
            title = f"subject:{thisSubject}, run: {run}, time: {startTime}, speed:{speed:.2f}"
            #logger.info(f"**** Window {speed}, {label_subject}, {subject}, {run}, {startTime}")
            self.gen_TimeFreq_plots(fileN=fileN, title=title, dataRow=data_np, freqDImageDir=freqDImageDir, freqYLim=freqYLim)

    def plotFromRaw(self, folder, freqYLim, data, subject, speed, freqDImageDir):
        sTime = 0
        desc_str = f"Creating image set for {folder}" 
        logger.info(desc_str)
        print(f"subject: {len(subject)}")
        for row, dataRow in tqdm(enumerate(data), total=len(data), desc=desc_str, unit=" Image Set", leave=False):
            thisSpeed = speed[row]
            fileN = f"subject-{subject}_run-{row}"
            title = f"Domain subject:{subject}, run:{row}, speed:{thisSpeed:.2f}"
            self.gen_TimeFreq_plots(fileN=fileN, title=title, dataRow=dataRow, freqDImageDir=freqDImageDir, freqYLim=freqYLim)
            #logger.info(f"Freq max: {self.dataPlotter.freqMax}")
            #plotRunFFT(subjectData, self.dataConfigs.sampleRate_hz, subjectName, 0, "ByRun")
            sTime += self.stepSize_s

    def gen_TimeFreq_plots(self, fileN, title, dataRow, freqDImageDir, freqYLim):
            # subject, run, time
            #thisLab = torch.argmax(labels[row])
            #print(f"data shape: {data[row].shape}, sampRate: {self.dataConfigs.sampleRate_hz}")

            fMin = configs['cwt']['fMin']
            fMax = configs['cwt']['fMax']
            self.dataPlotter.plotTime(dataRow, fileN, f"Time {title}" )
            #self.dataPlotter.plotFreq(freqDImageDir, dataRow, fileN, f"Freq {title}", xlim=[0, fMax], xInLog=False, yInLog=True, yLim=freqYLim,show=False) #, subjectNumber, 0, "ByRun")
            self.dataPlotter.plotFreq(freqDImageDir, dataRow, fileN, f"Freq {title}", xlim=[fMin, fMax], xInLog=True, yInLog=True, yLim=freqYLim,show=False) #, subjectNumber, 0, "ByRun")
            self.dataPlotter.plotFreq(freqDImageDir, dataRow, fileN, f"Freq {title}", xlim=[10, self.dataConfigs.sampleRate_hz/2],  xInLog=False, yInLog=True, yLim = freqYLim, show=False) #, subjectNumber, 0, "ByRun")
            self.dataPlotter.plotFreq(freqDImageDir, dataRow, fileN, f"Freq {title}", xlim=[10, self.dataConfigs.sampleRate_hz/2],  xInLog=True, yInLog=True, yLim = freqYLim, show=False) #, subjectNumber, 0, "ByRun")

    def loadDataSet(self, dataSetFile=None,writeLog=True, batchSize = None):
        if batchSize != None: self.batchSize = batchSize

        timeD = False
        if dataSetFile == None:
            timeD = True
            fileName_str = self.fileStruct.dataDirFiles.saveDataDir
            dataSaveDir_str = self.fileStruct.dataDirFiles.saveDataDir.saveDataDir_name
            dataSetFile = f"{dataSaveDir_str}/{fileName_str.timeDData_file}"

        logger.info(f"Loading dataset file: {dataSetFile}")
        dataSet = HDF5Dataset(dataSetFile) # Keep the full dataset in order so we can transform off of it
        self.getPeramsFromHDF5Dataset(dataSet=dataSet, debug=True)
        if timeD:
            self.timeDDataSet = dataSet # Keep the full dataset in order so we can transform off of it
        else:
            self.CWTDataSet = dataSet # Keep the full dataset in order so we can transform off of it
        #self.dataShape = dataSet.shape

        self.createDataloaders(dataSet=dataSet, writeLog=writeLog)

    def createDataloaders(self, dataSet, writeLog=True):
        # Split sizes
        trainRatio = configs['data']['trainRatio']
        train_size = int(trainRatio * len(dataSet))  # 80% for training
        val_size = len(dataSet) - train_size  # 20% for validation

        logger.info(f"dataset: {len(dataSet)}, trainRatio: {trainRatio}, train: {train_size}, val: {val_size}")
        # Rand split was not obeying  config['trainer']['seed'], so force the issue
        dataSet_t, dataSet_v = random_split(dataSet, [train_size, val_size], torch.Generator().manual_seed(self.seed))
        dataSet_v.indices.sort() # put the validation data back in order

        if self.device == "mps":
            self.dataLoader_t = DataLoader(dataSet_t, batch_size=self.batchSize, shuffle=True)
            self.dataLoader_v = DataLoader(dataSet_v, batch_size=1, shuffle=False)
        else:
            nWorkers = 8
            self.dataLoader_t = DataLoader(dataSet_t, batch_size=self.batchSize, 
                                num_workers=nWorkers, persistent_workers=True, pin_memory=True, 
                                shuffle=True)
            self.dataLoader_v = DataLoader(dataSet_v, batch_size=1, 
                                num_workers=nWorkers, persistent_workers=True, pin_memory=True, 
                                shuffle=False)

        if writeLog: self.logDataShape()
    

    def windowData(self, data:np.ndarray, subject, speed):
        logger.info(f"Window length: {self.windowLen} points, step: {self.stepSize} points, data len: {data.shape} points")
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
                runsEnd = self.configs['data']['limitRuns']
            else:
                runsEnd = data.shape[0] # How many runs in this subjects dataset
            #for run in range(data.shape[0]): # make sure the data is in order one run at a time
            windowsWithData = 0
            for run in range(runsEnd): # test with the first few dataums
                nWindows = 0
                firstDataPoint = True

                dataPtsAfterStomp = -1 # -1: no stomp yet, then = #since stomp
                if self.stompFromFile:
                    #print(f"Subject: {subject}, run: {run}")
                    sTime_sec = self.getStompTime(subject, run)
                    startPoint = sTime_sec * self.dataConfigs.sampleRate_hz
                else:
                    startPoint = 0 #points

                while True:
                    startPoint = int(startPoint)
                    nWindows += 1
                    endPoint = startPoint + self.windowLen
                    #logger.info(f"window run: {run},  startPoint: {startPoint}, windowlen: {self.windowLen}, endPoint: {endPoint}, dataLen: {self.dataConfigs.dataLen_pts}, step: {self.stepSize}")
                    if self.dataConfigs.dataLen_pts <= endPoint: break
                    if self.configs['data']['limitWindowLen'] > 0:
                        if windowsWithData >= self.configs['data']['limitWindowLen']: 
                            logger.info(f"Ending sub: {subject}, run: {run}, window: {nWindows}")
                            break

                    thisDataBlock = data[run, :, startPoint:endPoint]  # trial, sensor, dataPoint
                    #logger.info(f"window data shape: {thisDataBlock.shape}")

                    for i in range(thisDataBlock.shape[0]): # Each ch
                        rms_thisCh = np.sqrt(np.mean(np.square(thisDataBlock[i,:])))
                        try:              rms_allCh = np.append(rms_allCh, rms_thisCh)
                        except NameError: rms_allCh = rms_thisCh
                        #print(f"rms_allCh: {rms_allCh.shape}")

                    # Keep the RMS of time = 0 for a baseline
                    if firstDataPoint:
                        rms_BaseLine = rms_allCh.copy()
                        firstDataPoint = False
                    rms_ratio = rms_allCh/rms_BaseLine  # The ratio of the RMS of the data to the baseline for stomp and no step

                    # Look for stomp, and keeps track of how many windows since the stomp
                    # The detection of no step is done in getSubjecteLabel
                    if self.stompFromFile == False:
                        if self.stompThresh == 0: nSkips = 0
                        else                    : nSkips = 3
                        thisSubjectId, dataPtsAfterStomp = self.findDataStart(dataPtsAfterStomp, rms_ratio, nSkips)
                    else: 
                        dataPtsAfterStomp = 1
                        nSkips = 0
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
                            windowsWithData += 1
                            #print(f"using | subjectId: {thisSubjectId}, run:{run}, startTime: {thisStartTime}")
                            thisDataBlock = np.expand_dims(thisDataBlock, axis=0) # add the run dim back to append

                            # Append the data, labels, and all that junk
                            try:              windowedData = np.append(windowedData, thisDataBlock, axis=0) # append on trials, now trials/windows
                            except NameError: windowedData = thisDataBlock
                            try:              labels_speed = np.append(labels_speed, speed[run])
                            except NameError: labels_speed = speed[run]
                            try:              labels_subject = np.append(labels_subject, thisSubjectId)
                            except NameError: labels_subject = thisSubjectId
                            thisSubjectNumber = self.getSubjectNumber(subject) #Keep track of the subject number appart from the label
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


        return windowedData, labels_speed, labels_subject, subjects, runs, startTimes

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
            if self.downSample > 1:
                accelerometer_data,  = self.downSampleData(accelerometer_data)

            # Get just the sensors we want
            if self.dataConfigs.sampleRate_hz == 0: 
                # get the peramiters if needed
                # ex: Sample Freq
                self.getDataInfo(file) # get the sample rate from the file
                #if self.downSample > 1: # keep all the downsample calcs in one place
                #    self.dataConfigs.sampleRate_hz /= self.downSample
                #    logger.info(f"Downsampled rate: {self.dataConfigs.sampleRate_hz} {self.dataConfigs.units}")

                #logger.info(f"window len: {self.windowLen_s}, step size: {self.stepSize_s}, sample Rate: {self.dataConfigs.sampleRate_hz}")
                self.dataConfigs.dataLen_pts = accelerometer_data.shape[2]
                self.windowLen = int(self.windowLen_s * self.dataConfigs.sampleRate_hz)
                self.stepSize  = int(self.stepSize_s  * self.dataConfigs.sampleRate_hz)
                #logger.info(f"window len: {self.windowLen}, step size: {self.stepSize}")

        return accelerometer_data 

    def downSampleData(self, data):
        from scipy.signal import decimate

        #logger.info(f" dataLen from file: {self.dataConfigs.dataLen_pts}")
        #logger.info(f"Before downsample shape: {np.shape(data)} ")
        nTrials, nCh, timePoints = data.shape
        downSampled_data = np.empty((nTrials, nCh, timePoints // self.downSample))  
        for trial in range(nTrials):
            for ch in range(nCh):
                downSampled_data[trial, ch] = decimate(data[trial, ch], 
                                                       self.downSample, 
                                                       ftype='iir', 
                                                       zero_phase=True)

        self.dataConfigs.sampleRate_hz /= self.downSample
        logger.info(f"After downsample shape: {np.shape(downSampled_data)} ")
        logger.info(f"Downsampled rate: {self.dataConfigs.sampleRate_hz} {self.dataConfigs.units}")
        return downSampled_data, self.dataConfigs.sampleRate_hz / self.downSample

    def getSubjectLabel(self, subjectNumber, vals):
        #Vals is currently the RMS ratio of the data to the baseline
        # TODO: get from config
        label = 0

        #print(f"vals: {vals}")
        #If any ch is above the thresh, we call it a step
        for chVal in vals:
            if chVal > self.dataThresh:
                label = self.getSubjectNumber(subjectNumber)
                break
        return label

    def getSubjectNumber(self, subjectNumber):
        return(self.classes.index(subjectNumber))

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
    
    def getStompTimes(self, csv_file_name):
        with open(csv_file_name, mode='r') as stompTimesFile:
            stompTimes_reader = csv.reader(stompTimesFile)
            header = next(stompTimes_reader)
            data = []
            for row in stompTimes_reader:
                subjects = row[0]
                runs = int(row[1])
                times = float(row[2])
                data.append((subjects, runs, times))
        #startTimes_np = np.array(data)
        startTimes_np = np.array(data, dtype=[("name", "U50"), ("run", "i4"), ("startTime", "f4")])

        #print(startTimes_np)
        return startTimes_np

    def getStompTime(self, subject, run):
        mask = (self.stompTimes_np["name"] == subject) & (self.stompTimes_np["run"] == run)
        return self.stompTimes_np["startTime"][mask][0]

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
        self.dataConfigs.origSRate_hz = self.dataConfigs.sampleRate_hz
        self.dataConfigs.units = general_parameters[0]['units'].decode('utf-8')
        logger.info(f"Data cap rate: {self.dataConfigs.sampleRate_hz} {self.dataConfigs.units}")

        dataBlockSize = file['experiment/data'].shape 
        #logger.info(f"File Size: {dataBlockSize}")
        self.dataConfigs.origDataLen_pts = dataBlockSize[2] # do in getSubjecteData after downsample
        self.dataConfigs.nSensors = dataBlockSize[1]
        self.dataConfigs.nTrials = dataBlockSize[0]
        #logger.info(f"nsensor: {self.nSensors}, nTrials: {self.nTrials}")


    # If you don't send the normClass, it will calculate based on the data
    def scale_data(self, data, writeToLog=False, norm:normClass=None, scaler=None, scale=None, debug=False):
        if isinstance(data, torch.Tensor): #convert to numpy
            data = data.numpy()
            isTensor = True
        else: 
            isTensor = False

        if debug: 
            logger.info(f"scale_data: constants:{norm}")
            logger.info(f"Before scaling: min: {np.min(data)}, max: {np.max(data)}, shape: {data.shape}")
            #logger.info(f"{data[0:10, 0, 0]}")
            #logger.info(f"Complex data: {np.iscomplexobj(data)}")
        if scaler==None: scaler= norm.type

        if np.iscomplexobj(data):
            if scaler == "std": dataScaled, norm = self.std_complexData(data, writeToLog, norm, debug)
            else:               dataScaled, norm = self.norm_complexData(data, writeToLog, norm)
        else:
            if scaler == "std": dataScaled, norm = self.std_data(data, writeToLog, norm, debug)
            else:               dataScaled, norm = self.norm_data(data, writeToLog, norm, scale, debug)

        if debug: 
            logger.info(f"After scaling: min: {np.min(dataScaled)}, max: {np.max(dataScaled)}, {norm}")

        if isTensor: # and back to tensor
            dataScaled = torch.from_numpy(dataScaled)

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

    def std_complexData(self, data, writeToLog, norm:normClass, debug):
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

        if writeToLog:
            self.writeScalerToLog(self.logfile, self.dataNormConst, complex=True)
        #logger.info(f"newmin: {np.min(np.abs(normData))},  newmax: {np.max(np.abs(normData))}")
        if debug:
            logger.info(f"std_complexData done: {norm}")
        return normData, norm

    def std_data(self, data, writeToLog, norm:normClass, debug):
        # Normalize the data
        # float: from 0 to 1
        # 0.5 = center
        if norm == None:
            norm = normClass(type="std", mean=np.mean(data), std=np.std(data))

        # scale the data
        #logger.info(f"norm mean: {norm.mean}, std: {norm.std}")
        normData = (data - norm.mean)/norm.std # standardise

        if debug:
            logger.info(norm)
            #logger.info(f"Orig: \n{data[0:8]}")
            #logger.info(f"Norm: \n{normData[0:8]}")

        if writeToLog:
            self.writeScalerToLog(self.logfile, norm)
        #logger.info(f"newmin: {np.min(np.abs(normData))},  newmax: {np.max(np.abs(normData))}")
        return normData, norm

    def norm_complexData(self, data, writeToLog, norm:normClass):
        #L2 norm is frobenious_norm, saved as mean

        if norm == None:
            norm = normClass(type="norm", min=np.min(data), max=np.max(data), mean=np.linalg.norm(data) )

        normData = data/norm.mean

        if writeToLog:
            self.writeScalerToLog(self.logfile, norm)
        #logger.info(f"l2 norm: {self.dataNormConst.mean}, newmin: {np.min(normData)},  newmax: {np.max(normData)}")
        return normData, norm

    def norm_data(self, data, writeToLog, norm:normClass, scale, debug):
        #https://en.wikipedia.org/wiki/Feature_scaling
        if norm == None:
            norm = normClass(type="norm", min=np.min(data), max=np.max(data), mean=np.mean(data), scale=scale)

        if norm.type == 'meanNorm': 
            normTo = norm.mean
        elif norm.type == 'minMaxNorm': 
            normTo = norm.min

        normData = (data-normTo)/(norm.max - norm.min) 

        if debug:
            logger.info(f"normeddata:{data},  normTo: {normTo}, max: {norm.max}, min: {norm.min} | normData: {normData}")

        #Rescale for min/max
        if norm.scale!= 1:
            newMin = -norm.scale
            newMax = norm.scale
            normData = newMin + (newMin - newMax)*normData

        if debug:
            logger.info(f"After rescale: {normData}")

        if writeToLog:
            self.writeScalerToLog(self.logfile, norm)
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

    def logDataShape(self):
        logger.info(f"Writing to: {self.logfile}")
        trainDataSize = len(self.dataLoader_t.dataset)
        valDataSize = len(self.dataLoader_v.dataset)
        logger.info(f"Dataset train: {trainDataSize}, val: {valDataSize}")
        with open(self.logfile, 'a', newline='') as csvFile:
            writer = csv.writer(csvFile, dialect='unix')
            writer.writerow(['--------- Data Shape------------------'])
            writer.writerow(['TimeD Data Shape: windows/Ch/Time Points', 'train batch size', 'val batch size', 'train count', 'val count'])
            writer.writerow([self.timeDDataSet.shape, self.dataLoader_t.batch_size, self.dataLoader_v.batch_size, trainDataSize, valDataSize ] )
            if self.CWTDataSet is not None:
                writer.writerow(['Transformed Data Shape windows/Ch/Freqs/TimePoints'])
                writer.writerow([self.CWTDataSet.shape])
            writer.writerow(['---------------------------------------------'])


    def writeScalerToLog(self, logFile, scaler:normClass, data=True, complex=False, whoAmI=""):
        logger.info(f"Writing data scaler to: {logFile}")
        with open(logFile, 'a', newline='') as csvFile:
            writer = csv.writer(csvFile, dialect='unix')
            dataOrLabel = "Data"
            if data == False: dataOrLabel = "Label"
            writer.writerow([f'--------- Loaded:{whoAmI} {dataOrLabel}, Scaler Type:{scaler.type}, complex: {complex} -------'])
            writer.writerow(['min', 'max', 'mean', 'std', 'scale'])
            writer.writerow([scaler.min, scaler.max, scaler.mean, scaler.std, scaler.scale])
        logger.info(f"Data: min: {scaler.min}, max: {scaler.max}, mean: {np.abs(scaler.mean)}, std: {scaler.std}, scale: {scaler.scale}")
    
    def getThisWindowData(self, dataumNumber, ch=0 ):
        # If ch is 0, then we want all the channels

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
            fftData[ch] = freqData[0] # nomed to parseval in jFFT
            #fftData[ch] = freqData[0] / np.sqrt(len(freqData[0])) # Parsevals's theorem
            #fftData[ch] = freqData[0] * self.downSample # Correct for downsampleing

        return freqList, fftData

    def plotDataByWindow(self, cwt_class:"cwt", logScaleData:bool=False):
        # Plot the windowed data
        if self.configs['debugs']['generateTimeFreqWindowPlots']:
            yLim = configs['plts']['yLim_freqD']
            self.plotTime_FreqData(data=self.data_raw, freqYLim=yLim, folder="plots_byWindow")

    def writeToCWTDataFile(self, filename, data, label_speed, label_subject, subject, run, startTime):
        # Create HDF5 file with expandable datasets
        label_speed = label_speed.reshape(-1,1) #go from (num,) to (num,1)
        #logger.info(f"Data type: {type(data)}, shape {data.shape} ")
        #if np.iscomplexobj(data): data = np.abs(data)
        ## TODO: Save the complex data

        dataShape = data.shape
        with h5py.File(filename, "a") as cwtDataFile:
            if "data" not in cwtDataFile:
                # Create datasets with `maxshape=(None, ...)` to allow dynamic resizing
                    #cwtDataFile.create_dataset("data", shape=(0, dataShape[0], dataShape[1], dataShape[2]), maxshape=(None, dataShape[2], dataShape[1], dataShape[2]), dtype="complex64", chunks=True)
                cwtDataFile.create_dataset("data", shape=(0, dataShape[0], dataShape[1], dataShape[2]), maxshape=(None, dataShape[0], dataShape[1], dataShape[2]), dtype="float32", chunks=True)

                cwtDataFile.create_dataset("labelsSpeed", shape=(0,1), maxshape=(None,1), dtype="float32", chunks=True)
                cwtDataFile.create_dataset("labelsSubject", shape=(0,), maxshape=(None,), dtype="int32", chunks=True)
                cwtDataFile.create_dataset("subjects", shape=(0,), maxshape=(None,), dtype="int32", chunks=True)
                cwtDataFile.create_dataset("runs", shape=(0,), maxshape=(None,), dtype="int32", chunks=True)
                cwtDataFile.create_dataset("sTimes", shape=(0,), maxshape=(None,), dtype="float16", chunks=True)

            # Get datasets
            data_ds = cwtDataFile["data"]
            label_speed_ds = cwtDataFile["labelsSpeed"]
            label_subject_ds = cwtDataFile["labelsSubject"]
            subjectes_ds = cwtDataFile["subjects"]
            runs_ds = cwtDataFile["runs"]
            sTimes_ds = cwtDataFile["sTimes"]

            # Resize datasets to accommodate the new entry
            #new_index = data_ds.shape[0]  # current number of trials
            #data_ds.resize(new_index + 1, axis=0)
            data_ds.resize(data_ds.shape[0] + 1, axis=0)
            label_speed_ds.resize(label_speed_ds.shape[0] + 1, axis=0)
            label_subject_ds.resize(label_subject_ds.shape[0] + 1, axis=0)
            subjectes_ds.resize(subjectes_ds.shape[0] + 1, axis=0)
            runs_ds.resize(runs_ds.shape[0] + 1, axis=0)
            sTimes_ds.resize(sTimes_ds.shape[0] + 1, axis=0)
    
            # Append new data
            #new_index = data_ds.shape[0] - 1
            #data_ds[new_index] = data
            #data_ds[new_index:new_index+1, ...] = data[None, ...]

            data_ds[-1] = data
            label_speed_ds[-1] = label_speed
            label_subject_ds[-1] = label_subject
            subjectes_ds[-1] = subject
            runs_ds[-1] = run
            sTimes_ds[-1] = startTime


    def generateCWTDataByWindow(self, cwt_class:"cwt", logScaleData:bool=False):
            logger.info(f"  -----------------------   Generate CWT/Spectrogram Data  -------------------------")
            # plot the cwt data
            waveletDir = self.fileStruct.dataDirFiles.saveDataDir.waveletDir
            subFolder =waveletDir.waveletDir_name # We set up this dir when we inited the CWT

            # Generate the CWT data
            cwtFile = f"{waveletDir.waveletDir_name}/{waveletDir.cwtDataSet_Name}"
            logger.info(f"Checking for: {cwtFile}")
            filePath = Path(cwtFile)
            if filePath.exists() == False:
                self.generateCWT_calcNormTerms(cwt_class=cwt_class, saveFile=cwtFile)

            self.loadDataSet(dataSetFile=cwtFile, writeLog=True ) 
            logger.info(f"CWT Datashape: {self.CWTDataSet.shape}")

            # Plot the saved images
            if configs['debugs']['generateCWTPlots']:
                timeFFTCWT_dir= f"{subFolder}/{self.fileStruct.dataDirFiles.plotDirNames.time_fft_cwt}"
                if checkFor_CreateDir(timeFFTCWT_dir, echo=True) == False:
                    dataPlotter = saveCWT_Time_FFT_images(data_preparation=self, cwt_class=cwt_class, expDir=timeFFTCWT_dir)
                    dataPlotter.generateAndSaveImages(logScaleData)

    def specGramTransform(self, data, timeRes = 1.0, overlap = 0.95, debug=False):
        if debug:
            logger.info(f"SpectroGram Transform: sRate: {self.dataConfigs.sampleRate_hz}")
            logger.info(f"Transforming data: {type(data)}")
            #print(f"data block: {data.shape}")
            #print(f"N Ch: {data.shape[0]}")
        Sxx_list = []
        #for i, ch in enumerate(chList):
        for i  in range(0, data.shape[0]):
            #print(f" Data Block[i]: {i} {data[1].shape}")
            # nDataPoints in each window: Larger = better freq res, lower time res
            nperseg = int(timeRes * self.dataConfigs.sampleRate_hz) 
            noverlap = int(nperseg)*overlap # Overlap processing % overlap
            freqs, times, Sxx = spectrogram(data[i], fs=self.dataConfigs.sampleRate_hz, nperseg=nperseg, noverlap=noverlap)
            Sxx_list.append(Sxx)

        Sxx_np = np.stack(Sxx_list, axis=0)  # Shape: (n_channels, n_frequencies, n_time)

        # Trim the data to our fMin and fMax
        fMin = configs['cwt']['fMin']
        fMax = configs['cwt']['fMax']
        freq_mask = (freqs >= fMin) & (freqs <= fMax)
        freqs_trimmed = freqs[freq_mask]
        Sxx_np_trimmed = Sxx_np[:, freq_mask, :]  # preserves shape: (n_channels, n_selected_freqs, n_time)

        #logger.info(f"freqList: {freqs}")
        #logger.info(f"freqList trimmed: {freqs_trimmed}")
        #logger.info(f"Sxx shape: {Sxx_np_trimmed.shape}")

        return Sxx_np_trimmed, freqs_trimmed


    #TODO: Move to cwt
    def generateCWT_calcNormTerms(self, cwt_class:"cwt", saveFile ):
        # Can we transform the data in one shot? or dos this need a for loop?
        # Transform the RAW data. We do not actually have the data yet.
        timeStart = time()

        # Test the norm by loading the entire block and running the calcs there to compair
        # This takes too much mem for the entire set
        testCorrectness = self.configs['debugs']['testNormCorr']

        #self.dataNormConst = normClass() #This is bug, like 99% sure, but late to class
        self.dataNormConst.min = 10000
        self.dataNormConst.max = 0

        cwtData_raw = None
        cwtFrequencies = None
        #sum = 0
        mean = 0
        variance = 0
        mean_Real = 0
        mean_Imag = 0
        variance_Real = 0
        variance_Imag = 0
        nElements = 0

        for i, (data_tensor, label_speed, label_subject, subject, run, startTime) in  \
                  tqdm(enumerate(self.timeDDataSet), total= len(self.timeDDataSet), desc=f"Generating {cwt_class.wavelet_name} Data", unit="Window" ):
            data = data_tensor.numpy()
            #logger.info(f"Transforming data: {i}, {data.shape}")
            if cwt_class.wavelet_name == 'spectroGram':
                thisCwtData_raw, cwtFrequencies = self.specGramTransform(data, timeRes=1.0, overlap=0.9, debug=False)
                cwt_class.frequencies = cwtFrequencies
            else:
                thisCwtData_raw, cwtFrequencies = cwt_class.cwtTransform(data, debug=False)
            #logger.info(f"CWT data type: {type(thisCwtData_raw)}, shape: {thisCwtData_raw.shape}")
            #logger.info(f"CWT max: {np.max(thisCwtData_raw)}")
            self.writeToCWTDataFile(saveFile, thisCwtData_raw, label_speed, label_subject, subject, run, startTime)

            nElements += thisCwtData_raw.size
            min = np.min(thisCwtData_raw)
            max = np.max(thisCwtData_raw)
            #sum += np.sum(thisCwtData_raw)/thisCwtData_raw.size

            if np.iscomplexobj(data):
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
                variance_Real += np.sum((real - mean_Real) *delta_real)
                variance_Imag += np.sum((imag - mean_Imag) *delta_imag)

            else:
                this_mean = np.mean(thisCwtData_raw)
                delta = this_mean - mean
                mean += delta/(i+1)
                variance += np.sum((thisCwtData_raw - mean) *delta)

            if min < self.dataNormConst.min: self.dataNormConst.min = min
            if max > self.dataNormConst.max: self.dataNormConst.max = max
                #logger.info(f"#{i}: {self.dataNormConst.min}, max: {self.dataNormConst.max}, running Sum: {sum}")

            if testCorrectness:
                thisCwtData_raw = np.expand_dims(thisCwtData_raw, axis=0) # add the run dim back to append
                #logger.info(f"thisCwtData_raw: {type(thisCwtData_raw)}, {thisCwtData_raw.shape}")
                if cwtData_raw is None: cwtData_raw = thisCwtData_raw.copy()
                else:                        cwtData_raw = np.append(cwtData_raw, thisCwtData_raw, axis=0)

        #For testing to make sure we have the right thing
        if testCorrectness:
            cwtData_raw = np.transpose(cwtData_raw, (1, 2, 0, 3))           # we want: windows, ch, freqs, timepoints
            cwtTransformTime = time() - timeStart

        if np.iscomplexobj(data):
            mean = mean_Real + 1j * mean_Imag
            std  = np.sqrt(variance_Real/(nElements-1)) + 1j * np.sqrt(variance_Imag/(nElements-1)) 
        else:
            #print("NOT COMPLEX")
            std = np.sqrt(variance/(nElements-1))
        self.dataNormConst.mean = mean
        self.dataNormConst.std = std

        self.saveHDF5MetaData(saveFile)

        #self.dataNormConst.mean = sum/(i+1)
        logger.info(f"GenerateCWT, Norm stats : {self.dataNormConst}")
        if testCorrectness:
            logger.info(f"           | min: {np.min(cwtData_raw)}, max: {np.max(cwtData_raw)}, mean: {np.mean(cwtData_raw)}")
            logger.info(f"        | {np.std(np.real(cwtData_raw))}, + {np.std(np.imag(cwtData_raw))}i")


        if testCorrectness:
            logger.info(f"cwtData: {type(cwtData_raw)}, {cwtData_raw.shape}, cwtFrequencies: {type(cwtFrequencies)}, {cwtFrequencies.shape}, time: {cwtTransformTime:.2f}s")

    def getPeramsFromHDF5Dataset(self, dataSet:HDF5Dataset, dataSetFile=False, debug=False):
        HDF5Configs = dataSet.getConfig()
        self.dataConfigs.origSRate_hz = HDF5Configs["orig_sample_rate"]
        self.dataConfigs.sampleRate_hz = HDF5Configs["sample_rate"]
        self.dataConfigs.units = HDF5Configs["units"]
        self.dataConfigs.origDataLen_pts = HDF5Configs["orig_dataLen_pts"]
        self.dataConfigs.dataLen_pts = HDF5Configs["dataLen_pts"]
        self.dataConfigs.nSensors = HDF5Configs["nSensors"]
        self.dataConfigs.nTrials = HDF5Configs["nTrials"]
        self.dataConfigs.chList = HDF5Configs["chList"]
        logger.info(f"Loaded Peraimiters: srate: {self.dataConfigs.sampleRate_hz}")

        #TODO: Log the min, max, mean, std
        self.setNormConst(isData=True, norm=self.dataNormConst, dataSetFile=dataSetFile,
                          min=dataSet.data_min, max=dataSet.data_max, mean=dataSet.data_mean, std=dataSet.data_std)
        self.setNormConst(isData=False, norm=self.labNormConst,  dataSetFile=dataSetFile,
                          min=dataSet.lab_min, max=dataSet.lab_max, mean=dataSet.lab_mean, std=dataSet.lab_std)

        if debug:
            logger.info(f"Data: {self.dataNormConst}")
            logger.info(f"Labels: {self.labNormConst}")

    def setNormConst(self, isData, norm:normClass, dataSetFile, min, max, mean, std):
        norm.min = min
        norm.max = max
        norm.mean = mean
        norm.std = std
        self.writeScalerToLog(self.logfile, norm, data=isData, whoAmI=dataSetFile)
