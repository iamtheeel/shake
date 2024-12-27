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
import torch
import torch.nn.functional as tFun

#import math
from sklearn.model_selection import train_test_split #pip install scikit-learn

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Which sensor is which ch
#sensorChList = [[1], [2], [3], [4], [5], [6], [7], [8, 9, 10], [11, 12, 13], [14], [15], [16], [17], [18], [19], [20] ]
sensorZChList = [1, 2, 3, 4, 5, 6, 7, 8, 11, 14, 15, 16, 17, 18, 19, 20 ] # Just the Z chans


# Look for "stomp, wait a sec, then go"
# End block on???  all ch below a point?

# add batch size

# TODO: downsample 

class dataLoader:
    def __init__(self, config):
        # Load up the dataset info
        self.dataPath = config['data']['dataPath']# Where the data is
        self.test = config['data']['test']         # e.x. "Test_2"
        self.valPercen = config['data']['valSplitPercen']
        self.sensorList = config['data']['sensorList']

        #TODO: Get from file
        logger.info(f"data path: {self.dataPath}")

        self.classes = [0, 1, 2]
        self.nClasses = 3

        self.samRate_hz = 0
        self.dataLen_pts = 0
        self.windowLen_s = config['data']['windowLen'] 
        self.stepSize_s = config['data']['stepSize']
        self.windowLen = None
        self.stepSize = None

        self.dataPoints = 0 #99120
        self.nSensors = 0
        self.nTrials = 0

    def get_data(self):
        # Load all the data to a 3D numpy matrix:
        # 0 = trial: around 20
        # 1 = channels: 20
        # 2 = dataPoints: 99120

        # The data is [1,3]
        # Each image is 2

        # The labels are an array:
        # labels = subject/run


        self.subjects = self.getSubjects()
        for subjectNumber in self.subjects:
            data_file_hdf5, label_file_csv = self.getFileName(subjectNumber)
            logger.info(f"\n-----------------------------------------")
            logger.info(f"Dataloader, datafile: {data_file_hdf5}")
            logger.info(f"Dataloader, lablefile: {label_file_csv}")

            # Load data file
            subjectData = self.getSubjectData(data_file_hdf5) # Only the chans we are interested, in the order we want
            logger.info(f"Subject: {subjectNumber}, subject shape: {np.shape(subjectData)}")

            # Window the data
            windowedBlock = self.windowData(subjectData, self.windowLen, self.stepSize)

            # Append the data to the set
            try:              data = np.append(data, windowedBlock, axis=0)  # or should this be a torch tensor?
            except NameError: data = windowedBlock

            #labels = np.append(labels, self.getSpeedLabels(label_file_csv))
            thisSubLabels = self.getSubjectLabels(subjectNumber, windowedBlock.shape[0]) # on lable per run/window
            #logger.info(f"Labels: {thisSubLabels}")
            try:              labels = torch.cat((labels, thisSubLabels), 0) 
            except NameError: labels = thisSubLabels

            logger.info(f"Up to: {subjectNumber}, Labels, data shapes: {thisSubLabels.shape}, {data.shape}")

        logger.info(f"Final Labels, data shapes: {labels.shape}, {data.shape}")

        # normalize the data
        data = self.std_data(data)
        #data = self.norm_data(data)
        data = torch.tensor(data, dtype=torch.float32)

        return data, labels
    
    def windowData(self, data, window_len, step_len):
        logger.info(f"Window length: {window_len}, step: {step_len}, data len: {data.shape}")

        # Strip the head/tails

        dataLen = data.shape[2]
        startPoint = 200 #points

        # do while endPoint <= thisPoint + data len
        while True:
            endPoint = startPoint + window_len
            if dataLen <= endPoint: break

            thisDataBlock = data[:, :, startPoint:endPoint]  # trial, sensor, dataPoint

            #logger.info(f"Find End Point: {thisDataBlock.shape}")
            #logger.info(f"Find Start Point: step: {startPoint}")
            # Look for stomp

            #logger.info(f"Find End Point: {endPoint}")
            # Look for rms < val
            # or rather if rms < val, don't use this block

            # append the data
            try:              windowedData = np.append(windowedData, thisDataBlock, axis=0) # append on trials, now trials/windows
            except NameError: windowedData = thisDataBlock


            #logger.info(f"Data Block: {windowedData.shape}")
            startPoint += window_len

        return windowedData




    def getSubjectData(self, data_file_name):
        with h5py.File(data_file_name, 'r') as file:
            # Get the data from the datafile
            logger.info(f"Sensor List: {self.sensorList}")
            for sensor in self.sensorList:
                ch = sensorZChList[sensor]-1 
                #print(f"sensors: {sensor}, ch: {ch}")
                thisChData = file['experiment/data'][:, ch-1, :]  # trial, sensor, dataPoint
                thisChData = np.expand_dims(thisChData, axis=1)

                try:              accelerometer_data = np.append(accelerometer_data, thisChData, axis=1)
                except NameError: accelerometer_data = thisChData

            logger.info(f"data shape: {np.shape(accelerometer_data)} ")

            # Get just the sensors we want

            if self.samRate_hz == 0: 
                # get the peramiters if needed
                # ex: Sample Freq
                self.getDataInfo(file) # get the sample rate from the file

                logger.info(f"window len: {self.windowLen_s}, step size: {self.stepSize_s}, sample Rate: {self.samRate_hz}")
                self.windowLen = self.windowLen_s * self.samRate_hz
                self.stepSize  = self.stepSize_s  * self.samRate_hz
                logger.info(f"window len: {self.windowLen}, step size: {self.stepSize}")

        return accelerometer_data

    def getSubjectLabels(self, subjectNumber, nRuns):
        if(subjectNumber == '001'): label = 0
        if(subjectNumber == '002'): label = 1
        if(subjectNumber == '003'): label = 2

        labelListArr =  np.full((nRuns), label, dtype=int)
        print(f"labelList: {labelListArr.shape}")
        labelListTens = torch.from_numpy(labelListArr)
        
        # make 0 = [1,0,0], 1 = [0,1,0]... etc
        labelList = tFun.one_hot(labelListTens, num_classes=self.nClasses)

        return labelList

    def getSpeedLabels(self, csv_file_name ):
        with open(csv_file_name, mode='r') as labelFile:
            labelFile_csv = csv.DictReader(labelFile)
            for line_number, row in enumerate(labelFile_csv, start=1):
                speed_L = float(row['Gait - Lower Limb - Gait Speed L (m/s) [mean]'])
                speed_R = float(row['Gait - Lower Limb - Gait Speed R (m/s) [mean]'])
                #logger.info(f"Line {line_number}: mean: L={speed_L}, R={speed_R}(m/s) ")
                aveSpeed = (speed_L+speed_R)/2
                labelList = np.append(labelList, aveSpeed)

        print(f"Labels: {labelList}")
        return labelList


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
        units = general_parameters[0]['units'].decode('utf-8')
        logger.info(f"Data cap rate: {self.samRate_hz} {units}")

        dataBlockSize = file['experiment/data'].shape 
        self.dataLen_pts = dataBlockSize[2]
        #logger.info(f"File Size: {dataBlockSize}")
        #nSensors = 19 # There are 20 acceleromiters
        self.nSensors = dataBlockSize[1]
        self.nTrials = dataBlockSize[0]
        logger.info(f"nsensor: {self.nSensors}, nTrials: {self.nTrials}, dataPoints: {self.dataLen_pts} ")

    def std_data(self, data):
        # Normalize the data
        # float: from 0 to 1
        # 0.5 = center
        mean = np.mean(data)
        std = np.std(data)
        #logger.info(f"Orig: {data[0:3, 0:5, 0:2]}")

        # scale the data
        normData = (data - mean)/std # standardise

        logger.info(f"Data: Mean = {mean}, std = {std}")
        return normData

    def norm_data(self, data):
        # Normalize the data
        # float: from 0 to 1
        max = np.max(data)
        #logger.info(f"Orig: {data[0:3, 0:5, 0:2]}")

        # scale the data
        data = data/(2*1.1*max) # 2 time the max to leave room for the shift, and give 10% headroom
        mean = np.average(data)
        data = data - mean + 0.5 # shift to the mean = 0.5

        #logger.info(f"Data: Mean = {mean}, Max = {max}")
        return data


    def split_trainVal(self, data, labels ):
        logger.info(f"Splitting Test/Train: {self.valPercen}:1")

        splitSeed = 86 
        train_data, test_data = train_test_split(data, test_size=self.valPercen, random_state=splitSeed)
        train_label, test_label = train_test_split(labels, test_size=self.valPercen, random_state=splitSeed)
        '''
        train_data, test_data = train_test_split(data.reshape(-1,1), test_size=self.valPercen, random_state=splitSeed)
        train_label, test_label = train_test_split(labels.reshape(-1,1), test_size=self.valPercen, random_state=splitSeed)
        train_data = train_data.reshape(-1)
        test_data = test_data.reshape(-1)
        train_label = train_label.reshape(-1)
        test_label = test_label.reshape(-1)
        '''

        return train_data, test_data, train_label, test_label