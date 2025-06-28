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


def checkFor_CreateDir(thisDir, echo=True):
    '''
    Checks for the dir, returns true if it exists
    if not returns false and creates
    '''
    if echo:
        logger.info(f"Checking for: {thisDir}")
    dir_path = Path(thisDir)
    if dir_path.is_dir():
        return True
    else: 
        logger.info(f"Creating: {thisDir}")
        dir_path.mkdir(parents=True, exist_ok=True)  # Creates directory if missing
        return False
    

class runStats():
    def __init__(self):
        self.mean = 0
        self.M2 = 0 # Sum of the squares of the differences from the mean
        self.nElements = 0
        self.min = float('inf')
        self.max = float('-inf')
        self.std = 0

    def addElement(self, value):
        self.nElements += 1

        if value < self.min: self.min = value
        if value > self.max: self.max = value

        delta = value - self.mean               # The diff from the current mean
        self.mean += delta/self.nElements       # the running mean
        self.M2 += (value - self.mean) * delta  # (the delta is before updating the mean)

    def finish(self):
        import math
        if self.nElements > 1: #Watch the div/0
            self.std = math.sqrt(self.M2/(self.nElements-1)) # n-1 for sample std