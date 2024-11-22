###
# trainer_main.py
# Joshua Mehlman
# MIC Lab
# Fall, 2024
###
# Data Loader
###

import h5py, csv
import numpy

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class dataLoader:
    def __init__(self, config):
        # Load up the dataset info
        self.dataPath = config['dataPath']# Where the data is
        self.test = config['test']         # e.x. "Test_2"
        logger.info(f"data path: {self.dataPath}")

    def get_data(self):
        # Load all the data to a 3D numpy matrix:
        # x = datapoints
        # y = channels: 20
        # z = subject/run: subject*run ~60. This is our "image"
        # The labels are an array:
        # labels = subject/run
        data = None
        labels = None

        self.subjects = self.getSubjects()

        for subjectNumber in self.subjects:
            trial_str, csv_str = self.getFileName(subjectNumber)
            print(f"Dataloader, datafile: {trial_str}")
            print(f"Dataloader, lablefile: {csv_str}")

            # Load data file

            # Load label file

            # Glue the data together

            '''
            # Path th the .csv label file (the labels)
            if(self.test==2):
                csv_path = f'{self.dataPath}/Test_{self.test}/{trial_str}.csv'
                with open(csv_path, mode='r') as labelFile:
                    labelFile_csv = csv.DictReader(labelFile)
                    #labelFile_csv = csv.reader(labelFile)
                    labelList = [] 
                    for line_number, row in enumerate(labelFile_csv, start=1):
                        speed_L = float(row['Gait - Lower Limb - Gait Speed L (m/s) [mean]'])
                        speed_R = float(row['Gait - Lower Limb - Gait Speed R (m/s) [mean]'])
                        #print(f"Line {line_number}: mean: L={speed_L}, R={speed_R}(m/s) ")
                        #thisLabel = (speed_L, speed_R, (speed_L+speed_R)/2)
                        labelList.append((speed_R+speed_L)/2)
            '''
        return data, labels

    def getSubjects(self):
        if(self.test == "Test_2"):
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
        elif(self.test == "Test_4"):
            trial_str = f"walking_hallway_classroom_{subjectNumber}"
            #TestData/Test_4/data/walking_hallway_classroom_three_people_ritght_after_the_other_001_002_003.hdf5
        else:
            print(f"ERROR: No such subject, test: {self.test}, {subjectNumber}")
        
        trial_str = f"{self.dataPath}/{self.test}/data/{trial_str}.hdf5"
        csv_str = f"{self.dataPath}/{self.test}/{csv_str}.csv"

        return trial_str, csv_str


    def norm_data(self, data):
        # Normalize the data
        # float: from 0 to 1
        # 0.5 = center
        logger.info(f"Normalizing Data")


    def split_trainVal(self, splitRatio):
        logger.info(f"Splitting Test/Train: {splitRatio}:1")
        self.splitRatio = splitRatio