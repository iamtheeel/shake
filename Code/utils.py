###
# Footfall
# Joshua Mehlman
# MIC Lab
# Spring, 2025
###
# Some Useful utilities
###

from time import time

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