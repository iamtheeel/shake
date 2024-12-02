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

dataPoints = 99120


class dataLoader:
    def __init__(self, config):
        # Load up the dataset info
        self.dataPath = config['data']['dataPath']# Where the data is
        self.test = config['data']['test']         # e.x. "Test_2"
        self.valPercen = config['data']['valSplitPercen']

        #TODO: Get from file
        self.classes = [0, 1, 2]

        logger.info(f"data path: {self.dataPath}")

    def get_data(self):
        # Load all the data to a 3D numpy matrix:
        # 0 = trial: around 20
        # 1 = channels: 20
        # 2 = dataPoints: 99120

        # The data is [1,3]
        # Each image is 2

        # The labels are an array:
        # labels = subject/run

        #TODO: open the first and get the dataPoints

        data = np.empty([0, 20, dataPoints], dtype=float) 
        labels =  torch.empty([0,3])
        #labels =  np.empty([0], dtype= float)
        logger.info(f"dataShape: {np.shape(data)}")
        logger.info(f"labelShape: {np.shape(labels)}")

        self.subjects = self.getSubjects()

        for subjectNumber in self.subjects:
            data_file_hdf5, label_file_csv = self.getFileName(subjectNumber)
            logger.info(f"Dataloader, datafile: {data_file_hdf5}")
            logger.info(f"Dataloader, lablefile: {label_file_csv}")

            # Load data file
            subjectData = self.getSubjectData(data_file_hdf5)
            logger.info(f"Subject: {subjectNumber}, subject shape: {np.shape(subjectData)} ")
            data = np.append(data, subjectData, axis=0)

            # Load label file
            nRuns = np.shape(subjectData)[0]
            #logger.info(f"{subjectNumber}: {nRuns}")
            thisSubLabels = self.getSubjectLabels(subjectNumber, nRuns)
            labels = torch.cat((labels, thisSubLabels), 0)
            #labels = np.append(labels, self.getSpeedLabels(label_file_csv))
            logger.info(f"Labels, data shapes: {thisSubLabels.shape}, {subjectData.shape}")
            #logger.info(f"Labels: {thisSubLabels}")

        # Strip the head/tails

        # normalize the data
        data = self.norm_data(data)


        return data, labels
    
    
    def getSubjectData(self, data_file_name):
        with h5py.File(data_file_name, 'r') as file:
            # get the peramiters if needed
            # ex: Sample Freq

            # Get the data from the datafile
            accelerometer_data = file['experiment/data'][:, :, :]  # Nth trial, Mth accelerometer
            dataBlockSize = file['experiment/data'].shape 
            #logger.info(f"File Size: {dataBlockSize}")
            #nSensors = 19 # There are 20 acceleromiters
            nSensors = dataBlockSize[1]
            nTrials = dataBlockSize[0]
            logger.info(f"nsensor: {nSensors}, nTrials: {nTrials}, dataType: {type(accelerometer_data)}, shape: {np.shape(accelerometer_data)} ")

        return accelerometer_data

    def getSubjectLabels(self, subjectNumber, nRuns):
        if(subjectNumber == '001'): label = 0
        if(subjectNumber == '002'): label = 1
        if(subjectNumber == '003'): label = 2
        labelListArr =  np.full((nRuns), label, dtype=int)
        print(f"labelList: {labelListArr.shape}")
        labelListTens = torch.from_numpy(labelListArr)
        
        # make 0 = [1,0,0], 1 = [0,1,0]... etc
        labelList = tFun.one_hot(labelListTens, num_classes=3)
        print(f"label List: {labelList.shape}")
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


    def norm_data(self, data):
        # Normalize the data
        # float: from 0 to 1
        # 0.5 = center
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