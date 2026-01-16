from ultralytics import YOLO
import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np

project_root = Path(__file__).parent.parent if '__file__' in globals() else Path().resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

class PosePipeline:
    def __init__(self, path='yolov8n-pose.pt'):
        # model path, model set to None intially
        self.path = path
        self.model = None
        
        # Load the model during initialization
        self._load_model()
        
    # Loads YOLOv8 with error handling
    def _load_model(self):

        try:
            self.model = YOLO(self.path) 
            print("Model loaded successfully.") 
        except FileNotFoundError: 
            raise RuntimeError(f"Model file not found: {self.path}") 
        except Exception as e: 
            raise RuntimeError(f"Failed to load model from {self.path}: {e}")


    # Select main person closest to camera
    def select_main_person(self, frame):

        boxes = frame.boxes.xyxy.cpu().numpy()  # shape: (N, 4)

        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        main_idx = np.argmax(areas)

        return main_idx


    # Run YOLO pose estimation on a video and extract pose data
    def process_video(self, video_path: str):
        rows = []

        results = self.model(video_path, stream=True)

        for frame_idx, frame in enumerate(results):
            if frame.keypoints is None or len(frame.keypoints.xy) == 0:
                continue

            #select shooter
            person_idx = self.select_main_person(frame)

            keypoints = frame.keypoints.xy[person_idx].cpu().numpy()
            confs = frame.keypoints.conf[person_idx].cpu().numpy()

            for kp_idx, (x, y) in enumerate(keypoints):
                rows.append({
                    "frame": frame_idx,
                    "keypoint": kp_idx,
                    "x": float(x),
                    "y": float(y),
                    "confidence": float(confs[kp_idx]),
                })

        return pd.DataFrame(rows)

        




        