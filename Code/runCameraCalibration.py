import sys, os
sys.path.append(os.path.dirname(__file__))

from utils.camera_calibrator import CameraCalibrator
from utils.researcher_base import Researcher

# load researcher settings found in settings directory
RESEARCHER = "duy"
config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
researcher = Researcher(RESEARCHER, config_path=config_path)
cc = CameraCalibrator(researcher)

# select script to run
RUN_CALIBRATION = False
RUN_DISTORTION_REMOVAL = False
CALCULATE_PERSPECTIVE_TRANSFORM = True

"""
To calibrate new camera settings,
Create new folder in cameraCalibration directory,
update RESEARCHER_settings.py with new CAMERA_CALIBRATION_DIR settings,
drop all checkerboard images in the new folder,
update CAMERA_SETTINGS with new values,
and run this script.
"""
if RUN_CALIBRATION:
    # settings to be saved in calibration file
    CAMERA_SETTINGS = {
    "pan": -9.45,
    "tilt": -19.50,
    "zoom": 1,
    "focus": 875,
    "autofocus": "on"
    }
    cc.calibrate(CAMERA_SETTINGS)

# undistort the video specified in RESEARCHER_settings.py
# times are in seconds
if RUN_DISTORTION_REMOVAL:
    cc.remove_distortion(start_time=10, end_time=15)

# calculate perspective transform from CSV file
# the CSV file should contain pixel coordinates and corresponding real-world coordinates
if CALCULATE_PERSPECTIVE_TRANSFORM:
    cc.calculate_perspective_transform()