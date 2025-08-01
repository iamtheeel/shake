import os
import cv2 as cv
import yaml

class Researcher:
    def __init__(self, researcher: str, config_path: str = "config.yaml"):
        self.researcher = researcher
        self.config_path = config_path
        self.settings = self.load_settings_from_yaml()
        self.video_capture = None
        self.video_metadata = {}

    def load_settings_from_yaml(self):
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        with open(self.config_path, "r") as f:
            config = yaml.safe_load(f)

        if self.researcher not in config:
            raise KeyError(f"Researcher '{self.researcher}' not found in config.")

        return config[self.researcher]

    def load_video(self):
        if self.video_capture is not None:
            return self.video_capture  # already loaded

        video_path = os.path.join(self.settings["FOLDER_DIR"], self.settings["VIDEO_FILENAME"])
        vid = cv.VideoCapture(video_path)

        if not vid.isOpened():
            print("Error: Could not open video.")
            return None

        self.video_capture = vid
        self.video_metadata = {
            "fps": vid.get(cv.CAP_PROP_FPS),
            "width": int(vid.get(cv.CAP_PROP_FRAME_WIDTH)),
            "height": int(vid.get(cv.CAP_PROP_FRAME_HEIGHT)),
            "total_frames": int(vid.get(cv.CAP_PROP_FRAME_COUNT)),
            "path": video_path,
        }

        print(f"Video loaded: {video_path}")
        return vid
