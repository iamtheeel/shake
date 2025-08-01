import numpy as np
import cv2 as cv
import glob
import os
import importlib
import pandas as pd
from utils.researcher_base import Researcher

class CameraCalibrator:
    def __init__(self, researcher: Researcher):
        # Initialize with a Researcher instance
        self.researcher = researcher
        self.settings = researcher.settings

        # Paths
        self.calibration_dir = self.settings["CAMERA_CALIBRATION_DIR"]
        self.param_file = os.path.join(self.calibration_dir, 'calibration.npz')

        # Checkerboard dimensions: (columns, rows)
        self.checkerboard_dims = (9, 6)

        # Calibration data containers
        self.camMatrix = None
        self.distCoeff = None
        self.rvecs = None
        self.tvecs = None
        self.repError = None
        self.camera_settings = None
        self.homography = None
        self.perspective_transform = None

    def calibrate(self, CAMERA_SETTINGS):
        imgPathList = glob.glob(os.path.join(self.calibration_dir, '*.jpg'))
        print(f'Found {len(imgPathList)} images for calibration.')

        nCols, nRows = self.checkerboard_dims
        termCriteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        worldPtsCur = np.zeros((nRows * nCols, 3), np.float32)
        worldPtsCur[:, :2] = np.mgrid[0:nCols, 0:nRows].T.reshape(-1, 2)

        worldPtsList = []
        imgPtsList = []

        for curImgPath in imgPathList:
            imgBGR = cv.imread(curImgPath)
            imgGray = cv.cvtColor(imgBGR, cv.COLOR_BGR2GRAY)
            cornersFound, cornersOrg = cv.findChessboardCorners(imgGray, (nCols, nRows), None)

            if cornersFound:
                worldPtsList.append(worldPtsCur)
                cornersRefined = cv.cornerSubPix(imgGray, cornersOrg, (11, 11), (-1, -1), termCriteria)
                imgPtsList.append(cornersRefined)

        if not worldPtsList or not imgPtsList:
            raise RuntimeError("No valid checkerboard corners found. Calibration failed.")

        self.repError, self.camMatrix, self.distCoeff, self.rvecs, self.tvecs = cv.calibrateCamera(
            worldPtsList, imgPtsList, imgGray.shape[::-1], None, None)
        self.camera_settings = CAMERA_SETTINGS

        R, _ = cv.Rodrigues(self.rvecs[0])
        T = self.tvecs[0].reshape(3, 1)
        RT = np.hstack((R[:, :2], T))
        H = self.camMatrix @ RT
        self.homography = np.linalg.inv(H)

        print('Camera Matrix:\n', self.camMatrix)
        print('Reprojection Error (pixels): {:.4f}'.format(self.repError))

        np.savez(self.param_file,
                 repError=self.repError,
                 camMatrix=self.camMatrix,
                 distCoeff=self.distCoeff,
                 rvecs=self.rvecs,
                 tvecs=self.tvecs,
                 camera_settings=CAMERA_SETTINGS,
                 homography=self.homography)

    def load_calibration(self):
        if not os.path.exists(self.param_file):
            raise FileNotFoundError(f"Calibration file not found: {self.param_file}")

        with np.load(self.param_file, allow_pickle=True) as data:
            self.camMatrix = data['camMatrix']
            self.distCoeff = data['distCoeff']
            self.rvecs = data['rvecs']
            self.tvecs = data['tvecs']
            self.repError = data['repError']
            self.camera_settings = data['camera_settings'].item()
            self.homography = data.get('homography', None)

        print("Calibration parameters loaded.")

    def pixel_to_meters(self, x_pixel, y_pixel, perspective=False):
        if perspective:
            if self.perspective_transform is None:
                self.load_calibration()
            assert self.perspective_transform is not None, "Perspective transform not available."
            transform = self.perspective_transform
        else:
            if self.homography is None:
                self.load_calibration()
            assert self.homography is not None, "Homography could not be loaded."
            transform = self.homography

        assert transform.shape == (3, 3), "Transform matrix must be 3x3."

        pixel_coords = np.array([float(x_pixel), float(y_pixel), 1.0])
        world_coords = transform @ pixel_coords
        world_coords /= world_coords[2]

        return world_coords[0], world_coords[1]

    def calculate_perspective_transform(self):
        if self.perspective_transform is not None:
            print("Perspective transform already calculated.")
            return

        if self.camMatrix is None:
            self.load_calibration()

        csv_path = os.path.join(self.calibration_dir, "perspectiveTransform.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        df = pd.read_csv(csv_path)
        required_cols = {'u_pixel', 'v_pixel', 'x_meters', 'y_meters'}
        if not required_cols.issubset(df.columns):
            raise ValueError(f"CSV must contain columns: {required_cols}")

        src_pts = df[['u_pixel', 'v_pixel']].values.astype(np.float32)
        dst_pts = df[['x_meters', 'y_meters']].values.astype(np.float32)

        if len(src_pts) < 4:
            raise ValueError("At least 4 point correspondences are required for homography.")

        H, status = cv.findHomography(src_pts, dst_pts)
        self.perspective_transform = H

        with np.load(self.param_file, allow_pickle=True) as data:
            save_dict = dict(data)
            save_dict['perspective_transform'] = self.perspective_transform

        np.savez(self.param_file, **save_dict)
        print("Perspective transform calculated and saved.")

    def remove_distortion(self, start_time=0, end_time=None, output_filename="undistorted_output.avi"):
        if self.camMatrix is None or self.distCoeff is None:
            self.load_calibration()

        vid = self.researcher.load_video()
        if vid is None:
            return

        meta = self.researcher.video_metadata
        fps = meta["fps"]
        width = meta["width"]
        height = meta["height"]
        total_frames = meta["total_frames"]

        start_frame = int(start_time * fps)
        end_frame = int(end_time * fps) if end_time else total_frames

        output_path = os.path.join(self.calibration_dir, output_filename)
        fourcc = cv.VideoWriter_fourcc(*'XVID')
        out = cv.VideoWriter(output_path, fourcc, fps, (width, height))

        vid.set(cv.CAP_PROP_POS_FRAMES, start_frame)
        camMatrixNew, _ = cv.getOptimalNewCameraMatrix(self.camMatrix, self.distCoeff, (width, height), 1, (width, height))

        print(f"Processing video from frame {start_frame} to {end_frame}...")

        while vid.isOpened():
            frame_id = int(vid.get(cv.CAP_PROP_POS_FRAMES))
            if frame_id >= end_frame:
                break

            ret, frame = vid.read()
            if not ret:
                break

            undistorted_frame = cv.undistort(frame, self.camMatrix, self.distCoeff, None, camMatrixNew)
            out.write(undistorted_frame)

        vid.release()
        out.release()
        print(f"Undistorted video saved to {output_path}")
