###
# plotWavelets.py
# Joshua Mehlman
# MIC Lab
# Fall, 2025
###
# Plot wavelets for reports and presentations
###

from pathlib import Path
import sys, os

parent = Path(__file__).resolve().parent.parent
sys.path.append(str(parent))

from cwtTransform import cwt
from fileStructure import fileStruct

# From MICLab
## Configuration
configFile = "config.yaml"
from ConfigParser import ConfigParser
config = ConfigParser(os.path.join(os.getcwd(), configFile))
configs = config.get_config()

fileStructure = fileStruct(configs=configs)
sRate = configs['data']['sampleRate']
#wavelet_base = "ricker"
#wavelet_base = "cmorl"
#center_freq = 0.8  # Hz
bandwidth = 10
wavelet_base = "cfstep"
center_freq = 2.14  # Hz

cwt_class = cwt(fileStructure=fileStructure, sampleRate=sRate, configs=configs)

cwt_class.setupWavelet(wavelet_base=wavelet_base, sampleRate_hz=sRate, f0=center_freq, bw=bandwidth, 
                       useLogForFreq=False, savePlots=False, showPlots=True, length=4096)