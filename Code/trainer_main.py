###
# trainer_main.py
# Joshua Mehlman
# MIC Lab
# Fall, 2024
###
# Train footfall data
###

# From MICLab
## Configuration
import os
from ConfigParser import ConfigParser
config = ConfigParser(os.path.join(os.getcwd(), 'config.yaml'))
configs = config.get_config()
t_configs = configs['data']

## Logging
import logging
debug = configs['debugs']['debug']
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
if debug == False:
    logging.disable(level=logging.CRITICAL)
    logger.disabled = True

## What platform are we running on
import platform
machine = platform.machine()
logger.info(f"machine: {machine}")
if machine == "aarch64":
    device = "tpu"
else:
    import torch
    device = "cpu"
    if torch.cuda.is_available(): device = "cuda"
    if torch.backends.mps.is_available() and torch.backends.mps.is_built(): device = "mps"


"""
Data Preparation
"""
print(f"INIT: Get Data")
from dataLoader import dataLoader
data_preparation = dataLoader(t_configs)

data, labels = data_preparation.get_data()