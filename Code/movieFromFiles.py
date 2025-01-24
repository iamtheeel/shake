###
# Footfal
# Joshua Mehlman
# MIC Lab
# Spring, 2025
###
# Models
###
import os
from pathlib import Path


from ConfigParser import ConfigParser
config = ConfigParser(os.path.join(os.getcwd(), 'config.yaml'))
configs = config.get_config()

## Logging
import logging
debug = configs['debugs']['debug']
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
if debug == False:
    logging.disable(level=logging.CRITICAL)
    logger.disabled = True

import cv2
import glob


                
fps = 5 #Frames per second, 2 fps corrupts the video

subject  = 2
runs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
#thisDir = 'plots_CH765'
#thisDir = 'plots_765_chFrameNorm'
thisDir = 'plots_765_maxNorm_0.008'

fileDir = f"{configs['plts']['animDir']}"
#fourcc = cv2.VideoWriter_fourcc(*'mp4v') #This makes green, and only green
fourcc = cv2.VideoWriter_fourcc(*'avc1')



for run in runs:
    outFile = None
    outFileName = f"{fileDir}/{thisDir}_subject-{subject}_run-{run}_fps-{fps}.mp4"
    #fileStr = f"{fileDir}/{thisDir}/*_subject-{subject}_run-{run}*.png" # Newer files
    fileStr = f"{fileDir}/{thisDir}/*_run-{run}_subject-{subject}*.png" # earlyer files
    logger.info(f"Processing: {fileStr}")

    try:            
        for file in sorted(glob.glob(fileStr)):
            logger.info(f"loading frame: {file}")
            frame = cv2.imread(file)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            if outFile is None:
                height, width, layers = frame.shape
                outFile = cv2.VideoWriter(outFileName, fourcc, fps, (width, height))
        
            if frame is not None:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                try:
                    outFile.write(frame)
                except cv2.error:
                    logger.error(f"Error writing frame: {file}")
            else:
                logger.error(f"Error reading frame: {file}")
    except Exception as e:
        logger.error(f"Error: {e}")

    finally:    
        if outFile is not None:
            outFile.release()
            logger.info(f"Saved video to: {outFileName}")