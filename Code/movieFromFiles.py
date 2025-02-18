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


                
fps = 3 #Frames per second, 2 fps corrupts the video

subject  = 2
#runs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
runs = [1, 2]
#thisDir = 'plots_CH765'
#thisDir = 'plots_765_chFrameNorm'
#thisDir = 'plots_765_maxNorm_0.008'
#thisDir = 'run-1_cmor-10-0.8_logScaleData-False_dataScaler-meanNorm_dataScale-1'
#thisDir = 'exp-1_cmor-10-0.8_logScaleData-False_dataScaler-meanNorm_dataScale-1'
#thisDir = 'exp-1_cmor-10-0.8_logScaleData-False_dataScaler-meanNorm_dataScale-1_ch654'
#thisDir = 'exp-2_mexh_logScaleData-False_dataScaler-meanNorm_dataScale-1_ch654'
#thisDir = 'exp-1_fstep-0.8125_logScaleData-False_dataScaler-meanNorm_dataScale-1_ch654'
#thisDir = 'exp-1_morl_logScaleData-False_dataScaler-meanNorm_dataScale-1'
#thisDir ='exp-2_cmorl-0.8125-6.0_logScaleData-False_dataScaler-meanNorm_dataScale-1'

### Fstep f_0 search
#thisDir = 'exp-1_fstep-0.5_logScaleData-False_dataScaler-meanNorm_dataScale-1'
#thisDir = 'exp-2_fstep-1_logScaleData-False_dataScaler-meanNorm_dataScale-1'
#thisDir = 'exp-3_fstep-1.5_logScaleData-False_dataScaler-meanNorm_dataScale-1'
#thisDir = 'exp-4_fstep-2_logScaleData-False_dataScaler-meanNorm_dataScale-1'
#thisDir = 'exp-1_fstep-2.14_logScaleData-False_dataScaler-meanNorm_dataScale-1_FreqLogScale'
#thisDir = 'exp-5_fstep-2.5_logScaleData-False_dataScaler-meanNorm_dataScale-1'
#thisDir = 'exp-1_fstep-3_logScaleData-False_dataScaler-meanNorm_dataScale-1'
#thisDir = 'exp-2_fstep-5_logScaleData-False_dataScaler-meanNorm_dataScale-1'
#thisDir = 'exp-3_fstep-10_logScaleData-False_dataScaler-meanNorm_dataScale-1'

### Fmin = 1
#thisDir = 'exp-1_fstep-2.14_logScaleData-False_dataScaler-meanNorm_dataScale-1_fmin_1'

### Lin
#thisDir = 'exp-1_fstep-2.14_logScaleData-False_dataScaler-meanNorm_dataScale-1_linFreq'

#thisDir = 'classification_chList-8_7_6_5_4_3_2_16_1_runLim-1_winCountLim-20_StompThresh-0_DataThresh-0_fstep-2.14_dataScaler-std'
thisDir ='classification_chList-8_7_6_5_4_3_2_16_1_runLim-1_winCountLim-20_StompThresh-0_DataThresh-0_cmor10-0.8_dataScaler-std'



fileDir = f"{configs['plts']['animDir']}"
#fourcc = cv2.VideoWriter_fourcc(*'mp4v') #This makes green, and only green
fourcc = cv2.VideoWriter_fourcc(*'avc1')



for run in runs:
    outFile = None
    outFileName = f"{fileDir}/{thisDir}/{thisDir}_subject-{subject}_run-{run}_fps-{fps}.mp4"
    #fileStr = f"{fileDir}/{thisDir}/*_run-{run}_subject-{subject}*.png" # earlyer files
    fileStr = f"{fileDir}/{thisDir}/images/*_subject-{subject}_run-{run}*.png" # Newer files
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