###
# Footfall
# Joshua Mehlman
# MIC Lab
# Spring, 2025
###
# File Structure
###

#There are 3 directory locations
# 1) Input data
# 2) Experiment Tracking Information
# 3) Output Data

from dataclasses import dataclass
### Input Data
'''
Root dir (From Config):
'''

### Experiment Tracking 
'''
Root dir (From Config):
    <Time and Date of Experiment>
        <timedate>_dataTrack.csv: Sumary of each run and results in one line format
        <timeDate>_log.csv:       Description of peramiters used for experiment

        run-<number>:
            <time Date>_modelinfo.txt
            run-<num>_log.csv
            valResults.csv
'''
@dataclass
class expTrackFiles_class:
    #Root dir (From Config)/<timeDate>:
    expTrackDir_name:str 
    expTrackDateTime:str = ""

    #Files:
    expTrack_sumary_file:str = ""
    expTrack_log_file:str = ""

    class expNum_class:
        expTrackDir_Name:str =""

        #Files
        #modelInfo_file:str = ""  # <modelName>_modelInfo.txt
        expTrackLog_file:str ="" #run-<num>_log.csv
        # trining loss figure
        # Validation results figure
        # Validateion results sumary
    expNumDir = expNum_class()

### Output Data 
@dataclass
class dataDirFiles_class:
    #Root dir (From Config):
    dataOutDir_name:str # The root dir

    #Time Domain Data Dir (<Regresion or Classification>_<list of channesl>_<Run Count limit>_<Stomp Thresh>_<Data thresh>):
    class saveDataDir_class:
        #The dir name
        saveDataFolder_name:str = ""
        saveDataDir_name:str = ""

        #Files: 
        timeDDataSave:str = 'timeD_data.npy' #File with the time domain data selected for this run
        timeDDataSumary:str = 'timeD_data.csv' #File with the time domain data selected for this run

        #Wavelet Data dir (<wavelet name>_<Center frequency>_<Bandwidth>):
        class waveletDir_class:
            waveletFolder_name:str
            waveletDir_name:str
            
            ## Files:
            #<name>_<f0>_<bw>_freqD.jpg #Plot of wavelet, frequency domain
            #<name>_<f0>_<bw>_timeD.jpg #Plot of wavelet, time domain

            #Data Scaling dir (<Scaling Type>_<normilize to>):
            #normPerams_<wavelet name>.pkl
            class dataNormDir_class:
                dataNormFolder_name:str
                dataNormDir_name:str

                #Files:
                normPeramsFile_name:str = "normPerams.pkl"

            dataNormDir = dataNormDir_class()

            #Time, Frequency, CWT Transformed Data images:
            #    The images generated.png
        waveletDir = waveletDir_class()
    saveDataDir = saveDataDir_class()

    class plotDirNames_class:
        #Image Dirs:
        baseDir: str= "images"
        time_fft_cwt:str = "time_fft_cwt"
        time:str = "time"
        freq:str = "freq"
    plotDirNames = plotDirNames_class()


import os
from ConfigParser import ConfigParser
config = ConfigParser(os.path.join(os.getcwd(), 'config.yaml'))
configs = config.get_config()

from cwtTransform import cwt
from dataLoader import normClass

from pathlib import Path

import logging
from pathlib import Path
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class fileStruct:
    def __init__(self):
        self.dataDirFiles  = dataDirFiles_class(dataOutDir_name=configs['data']['dataOutDir'])
        self.expTrackFiles = expTrackFiles_class(expTrackDir_name=configs['expTrackDir'])

    def makeDir(self, dir_str):
        '''
        Create the dir if its not already there
        '''
        dir_path = Path(dir_str)
        dir_path.mkdir(parents=True, exist_ok=True)

    def setExpTrack_dir(self, dateTime_str=None ):
        '''
        '''
        self.expTrackFiles.expTrackDir_name = f"{configs['expTrackDir']}/{dateTime_str}"
        if dateTime_str != None: self.expTrackFiles.expTrackDateTime = dateTime_str
        #The experiment tracking
        self.expTrackFiles.expTrack_log_file = f"{dateTime_str}_DataTrack_Log.csv"
        self.expTrackFiles.expTrack_sumary_file = f"{dateTime_str}_DataTrack_Sumary.csv"

        # The individual experiments
        self.expTrackFiles.expNumDir.expTrackSum_file = f"{dateTime_str}_DataTrack_Log.csv"
        self.expTrackFiles.expNumDir.expTrackLog_file = f"{dateTime_str}_DataTrack_Sumary.csv"

        self.makeDir(self.expTrackFiles.expTrackDir_name)
        logger.info(f"Experiment Track Dir: {self.expTrackFiles.expTrackDir_name}")

    def setExpTrack_run(self, expNum):
        self.expTrackFiles.expNumDir.expTrackDir_Name = f"{self.expTrackFiles.expTrackDir_name}/exp-{expNum}"
        self.makeDir(self.expTrackFiles.expNumDir.expTrackDir_Name)
        logger.info(f"This Experiment Dir: {self.expTrackFiles.expNumDir.expTrackDir_Name}")

    def setData_dir(self, dataConfigs):
        '''
        '''
        dataFolder =self.dataDirFiles.saveDataDir.saveDataFolder_name 
        self.regression = configs['model']['regression']
        #Set up a string for saving the dataset so we can see if we have already loaded this set
        chList_str = "_".join(map(str, dataConfigs.chList))
        #TODO: add more info:
        # windows limit, runs limit
        if self.regression: regClas = "regression"
        else:               regClas = "classification"
        dataFolder = f"{regClas}_chList-{chList_str}"
        runLimit = configs['data']['limitRuns']
        if runLimit > 0: dataFolder = f"{dataFolder}_runLim-{runLimit}"
        winLimit = configs['data']['limitWindowLen']
        if winLimit > 0: self.dataFolder = f"{dataFolder}_winCountLim-{winLimit}"
        dataFolder = f"{dataFolder}_StompThresh-{configs['data']['stompThresh']}"
        dataFolder = f"{dataFolder}_DataThresh-{configs['data']['dataThresh']}"

        self.dataDirFiles.saveDataDir.saveDataDir_name = f"{self.dataDirFiles.dataOutDir_name}/{dataFolder}"
        self.dataDirFiles.saveDataDir.saveDataFolder_name = "dataFolder"
        self.makeDir(self.dataDirFiles.saveDataDir.saveDataDir_name)

        logger.info(f"Data save folder: {self.dataDirFiles.saveDataDir.saveDataDir_name}")

    def setCWT_dir(self, cwtClass:cwt):
        '''
        '''
        waveletFolder_name = f"{cwtClass.wavelet_name}"
        if cwtClass.wavelet_name != 'None':
            waveletFolder_name = f"{waveletFolder_name}_fMin-{cwtClass.min_freq}_fMax-{cwtClass.max_freq}"

        if cwtClass.useLogScaleFreq: 
            logScFreq_st = "_logScaleFreq"
            waveletFolder_name = f"normPerams_{cwtClass.wavelet_name}{logScFreq_st}"
        #logger.info(f"Wavelet time from: {cwtClass.wavelet_Time[0]} to {cwtClass.wavelet_Time[-1]}")

        self.dataDirFiles.saveDataDir.waveletDir.waveletFolder_name = waveletFolder_name
        self.dataDirFiles.saveDataDir.waveletDir.waveletDir_name = \
            f"{self.dataDirFiles.saveDataDir.saveDataDir_name}/{waveletFolder_name}"
        
        logger.info(f"Wavelet Dir: {self.dataDirFiles.saveDataDir.waveletDir.waveletDir_name}")
        self.makeDir(self.dataDirFiles.saveDataDir.waveletDir.waveletDir_name)


    def setDataNorm_dir(self, norm:"normClass", logData=False):
        '''
        '''
        dataNormFolder = f"dataScaler-{norm.type}"
        if norm.scale != 1: 
            dataNormFolder = f"{dataNormFolder}_dataScale-{norm.scale}"

        dataNormDir = f"{self.dataDirFiles.saveDataDir.waveletDir.waveletDir_name}/{dataNormFolder}"

        self.dataDirFiles.saveDataDir.waveletDir.dataNormDir.dataNormFolder_name = dataNormFolder
        self.dataDirFiles.saveDataDir.waveletDir.dataNormDir.dataNormDir_name = dataNormDir
        logger.info(f"Data Norm Dir: {dataNormDir}")
        self.makeDir(dataNormDir)

        return self.dataDirFiles.saveDataDir.waveletDir.dataNormDir.dataNormDir_name 