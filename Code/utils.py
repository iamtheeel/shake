###
# Footfall
# Joshua Mehlman
# MIC Lab
# Spring, 2025
###
# Some Useful utilities
###

from time import time
from pathlib import Path

import logging
from pathlib import Path
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class timeTaken:
    def __init__(self, sigFig=0):
        self.startTime_s = time()
        self.endTime_s = None
        self.timeTaken_s = None
        self.sigFig = sigFig

    def startTime(self):
        self.startTime_s = time()

    def endTime(self, echo=False, echoStr=""):
        self.endTime_s = time()
        self.timeTaken_s = self.endTime_s - self.startTime_s
        if echo:
            print(f"{echoStr}: {self.reportTime()}")

    def reportTime(self):
        return  f"{self.timeTaken_s:.{self.sigFig}f}s"


def checkFor_CreateDir(thisDir):
    '''
    Checks for the dir, returns true if it exists
    if not returns false and creates
    '''
    logger.info(f"Checking for: {thisDir}")
    dir_path = Path(thisDir)
    if dir_path.is_dir():
        return True
    else: 
        logger.info(f"Creating: {thisDir}")
        dir_path.mkdir(parents=True, exist_ok=True)  # Creates directory if missing
        return False