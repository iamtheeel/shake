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


#from ConfigParser import ConfigParser
#config = ConfigParser(os.path.join(os.getcwd(), 'config.yaml'))
#configs = config.get_config()

## Logging
import logging
#debug = configs['debugs']['debug']
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
#if debug == False:
#    logging.disable(level=logging.CRITICAL)
#    logger.disabled = True

import cv2 #opencv-python
import glob


                
fps = 30 #Frames per second, x fps sometimes corrupts the video

subject  = 2
#runs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
#runs = [1, 2]
runs = [0]
#imageDir = 'plots_CH765'
#imageDir = 'plots_765_chFrameNorm'
#imageDir = 'plots_765_maxNorm_0.008'
#imageDir = 'run-1_cmor-10-0.8_logScaleData-False_dataScaler-meanNorm_dataScale-1'
#imageDir = 'exp-1_cmor-10-0.8_logScaleData-False_dataScaler-meanNorm_dataScale-1'
#imageDir = 'exp-1_cmor-10-0.8_logScaleData-False_dataScaler-meanNorm_dataScale-1_ch654'
#imageDir = 'exp-2_mexh_logScaleData-False_dataScaler-meanNorm_dataScale-1_ch654'
#imageDir = 'exp-1_fstep-0.8125_logScaleData-False_dataScaler-meanNorm_dataScale-1_ch654'
#imageDir = 'exp-1_morl_logScaleData-False_dataScaler-meanNorm_dataScale-1'
#imageDir ='exp-2_cmorl-0.8125-6.0_logScaleData-False_dataScaler-meanNorm_dataScale-1'

### Fstep f_0 search
#imageDir = 'exp-1_fstep-0.5_logScaleData-False_dataScaler-meanNorm_dataScale-1'
#imageDir = 'exp-2_fstep-1_logScaleData-False_dataScaler-meanNorm_dataScale-1'
#imageDir = 'exp-3_fstep-1.5_logScaleData-False_dataScaler-meanNorm_dataScale-1'
#imageDir = 'exp-4_fstep-2_logScaleData-False_dataScaler-meanNorm_dataScale-1'
#imageDir = 'exp-1_fstep-2.14_logScaleData-False_dataScaler-meanNorm_dataScale-1_FreqLogScale'
#imageDir = 'exp-5_fstep-2.5_logScaleData-False_dataScaler-meanNorm_dataScale-1'
#imageDir = 'exp-1_fstep-3_logScaleData-False_dataScaler-meanNorm_dataScale-1'
#imageDir = 'exp-2_fstep-5_logScaleData-False_dataScaler-meanNorm_dataScale-1'
#imageDir = 'exp-3_fstep-10_logScaleData-False_dataScaler-meanNorm_dataScale-1'

### Fmin = 1
#imageDir = 'exp-1_fstep-2.14_logScaleData-False_dataScaler-meanNorm_dataScale-1_fmin_1'

### Lin
rootDir = '/Volumes/Data/thesis/data_out/'

#imageDir = 'exp-1_fstep-2.14_logScaleData-False_dataScaler-meanNorm_dataScale-1_linFreq'
#imageDir = 'classification_chList-8_7_6_5_4_3_2_16_1_runLim-1_winCountLim-20_StompThresh-0_DataThresh-0_fstep-2.14_dataScaler-std'
#imageDir = '20250316-131404_mNet_inLine_valSaveEveryEpoch/exp-1'
#imageDir = 'chList-6_5_3_DownSample-4x/regression_winLen-5_step-1_StompThresh-File_DataThresh-2_runLim-2/cmorl10-0.8_mag_fMin-1_fMax-100_scales-256/s2_r0'
#imageDir = 'chList-6_5_3_DownSample-4x/regression_winLen-5_step-1_StompThresh-0_DataThresh-0_runLim-2/cmorl10-0.8_mag_fMin-1_fMax-100_scales-256/s2_run0'
#imageDir = 'chList-6_5_3_DownSample-4x/regression_winLen-5_step-0.25_StompThresh-0_DataThresh-0_runLim-1/cmorl10-0.8_mag_fMin-1_fMax-100_scales-256/s2_r0'
#imageDir = 'chList-6_5_3_DownSample-4x/regression_winLen-5_step-0.05_StompThresh-0_DataThresh-0_runLim-1/cmorl10-0.8_mag_fMin-1_fMax-100_scales-256/s2_r1'
#imageDir = 'chList-6_5_3_DownSample-4x/regression_winLen-5_step-0.02_StompThresh-0_DataThresh-0_runLim-1/cmorl10-0.8_mag_fMin-1_fMax-100_scales-256/s1_r0'
imageDir = 'chList-6_5_3_DownSample-4x/regression_winLen-5_step-0.0333_StompThresh-0_DataThresh-0_runLim-1/cmorl10-0.8_mag_fMin-1_fMax-100_scales-256/time_fft_cwt_plots'



#rootDir = f"{configs['plts']['animDir']}"
#fourcc = cv2.VideoWriter_fourcc(*'mp4v') #This makes green, and only green
#fourcc = cv2.VideoWriter_fourcc(*'avc1') #.mp4  # Frequently produces pixellation errors
fourcc = cv2.VideoWriter_fourcc(*'MJPG') #.avi good quality



for run in runs:
    outFile = None
    #outFileName = f"{rootDir}/{imageDir}/{imageDir}_subject-{subject}_run-{run}_fps-{fps}.mp4"
    outFileName = f"{rootDir}/{imageDir}/validationAcc_fps-{fps}.avi"

    # Input files
    #fileStr = f"{fileDir}/{imageDir}/*_run-{run}_subject-{subject}*.png" # earlyer files
    #fileStr = f"{rootDir}/{imageDir}/images/*_subject-{subject}_run-{run}*.png" # Newer files
    fileStr = f"{rootDir}/{imageDir}/*.png" # Newer files
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