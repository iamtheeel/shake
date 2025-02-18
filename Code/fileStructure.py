###
# Footfall
# Joshua Mehlman
# MIC Lab
# Spring, 2025
###
# File Structure
###

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
        self.dataOutDir = f"{configs['data']['dataOutDir']}"
        
        self.dataOutDir = f"{configs['data']['dataOutDir']}"
        self.dataSaveDir = None  # Folder that had the timedomain data  that was loaded

    def makeDir(self, dir_str):
        dir_path = Path(dir_str)
        dir_path.mkdir(parents=True, exist_ok=True)


    def setData_dir(self, dataConfigs):
        self.regression = configs['model']['regression']
        #Set up a string for saving the dataset so we can see if we have already loaded this set
        chList_str = "_".join(map(str, dataConfigs.chList))
        #TODO: add more info:
        # windows limit, runs limit
        if self.regression: regClas = "regression"
        else:               regClas = "classification"
        self.dataFolder = f"{regClas}_chList-{chList_str}"
        runLimit = configs['data']['limitRuns']
        if runLimit > 0: self.dataFolder = f"{self.dataFolder}_runLim-{runLimit}"
        winLimit = configs['data']['limitWindowLen']
        if winLimit > 0: self.dataFolder = f"{self.dataFolder}_winCountLim-{winLimit}"
        self.dataFolder = f"{self.dataFolder}_StompThresh-{configs['data']['stompThresh']}"
        self.dataFolder = f"{self.dataFolder}_DataThresh-{configs['data']['dataThresh']}"

        self.dataSaveDir = f"{self.dataOutDir}/{self.dataFolder}"
        self.makeDir(self.dataSaveDir)

        logger.info(f"Data save folder: {self.dataSaveDir}")

    def setCWT_dir(self, cwtClass:cwt):
        self.waveletFolder = f"{cwtClass.wavelet_name}_fMin-{cwtClass.min_freq}_fMax-{cwtClass.max_freq}"
        if cwtClass.useLogScaleFreq: 
            logScFreq_st = "_logScaleFreq"
            self.wavletDir = f"normPerams_{cwtClass.wavelet_name}{logScFreq_st}"
        #logger.info(f"Wavelet time from: {cwtClass.wavelet_Time[0]} to {cwtClass.wavelet_Time[-1]}")

        self.wavletDir = f"{self.dataSaveDir}/{self.waveletFolder}"
        
        logger.info(f"Wavelet Dir: {self.wavletDir}")
        self.makeDir(self.wavletDir)


    def setDataNorm_dir(self, norm:"normClass", logData=False):
        self.dataNormFolder = f"dataScaler-{norm.type}"
        if norm.scale != 1: 
            self.dataNormFolder = f"{self.dataNormFolder}_dataScale-{norm.scale}"

        self.dataNormDir = f"{self.wavletDir}/{self.dataNormFolder}"
        logger.info(f"Data Norm Dir: {self.dataNormDir}")
        self.makeDir(self.dataNormDir)