from ultralytics import YOLO
import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import cv2

from scripts.data_pipeline import compute_joint_angles

project_root = Path(__file__).parent.parent if '__file__' in globals() else Path().resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
    
COCO_SKELETON = [
    (5, 7), (7, 9),
    (6, 8), (8, 10),
    (5, 6),
    (11, 12),
    (5, 11), (6, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
]

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
        
        # Handle case when no person is detected
        if len(boxes) == 0:
            return None
        
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        main_idx = np.argmax(areas)

        return main_idx


    # Run YOLO pose estimation on a video and extract pose data
    def pose_data(self, video_path: str):
        rows = []

        results = self.model(video_path, stream=True)

        for frame_idx, frame in enumerate(results):
            if frame.keypoints is None or len(frame.keypoints.xy) == 0:
                continue

            #select shooter
            person_idx = self.select_main_person(frame)
            
            # Skip frame if no person detected
            if person_idx is None:
                continue

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

    def process_angles(self, video_path: str, side: str = "right"):
        df = self.pose_data(video_path) 
            
        angle_rows = [] # list to hold angle data for each frame
        current_frame = None # track current frame number
        kp = np.full((17, 2), np.nan, dtype=float) # initialize keypoint array with NaN
            
        # Go through entire dataframe and compute angles for each frame, handling missing keypoints with NaN
        for row in df.itertuples(index=False):
            # Extract frame number and keypoint index
            frame = int(row.frame)
            keypoint = int(row.keypoint)

            # If not current frame, set frame we found to current frame
            if current_frame is None:
                current_frame = frame
            # Otherwise, if we encounter a new frame, compute angles for the previous frame and reset keypoint array
            elif frame != current_frame:
                angles = compute_joint_angles(kp, side=side)
                angle_rows.append({"frame": current_frame, **angles})
                kp[:] = np.nan
                current_frame = frame

            # Only populate keypoint array if keypoint index is valid (0-16 for COCO format)
            if 0 <= keypoint < 17:
                kp[k, 0] = float(row.x)
                kp[k, 1] = float(row.y)

        # Compute angle for last frame
        if current_frame is not None:
            angles = compute_joint_angles(kp, side=side)
            angle_rows.append({"frame": current_frame, **angles})

        # Return angles
        return pd.DataFrame(angle_rows)
                        
        def _iter_frames(self, video_path: str, side: str = "right"):
            # Open video with OpenCV 
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Failed to open video: {video_path}")
            
            # Read FPS from video, default to 20 if not available
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps <= 1e-6:
                fps = 20.0
                
            writer = None
            
            try:
                while True:
                    ok, frame_bgr = cap.read()
                    # Stop if no frame is read (end of video)
                    if not ok:
                        break
                    
                    # Run YOLO pose estimation on the frame
                    results = self.model(frame_bgr, verbose=False)
                    r = results[0]

                    if r.keypoints is not None and len(r.keypoints.xy) > 0:
                        # Select main person and extract keypoints
                        person_idx = self.select_main_person(r)
                        
                        if person_idx is not None:    
                            keypoints = r.keypoints.xy[person_idx].cpu().numpy()
                            confs = r.keypoints.conf[person_idx].cpu().numpy()
                            
                            # Only draw a line if both endpoints have decent confidence
                            for a, b in COCO_SKELETON:
                                if confs[a] > 0.2 and confs[b] > 0.2:
                                    xa, ya = int(keypoints[a][0]), int(keypoints[a][1])
                                    xb, yb = int(keypoints[b][0]), int(keypoints[b][1])
                                    cv2.line(frame_bgr, (xa, ya), (xb, yb), (0, 255, 0), 2)
                                    
                            # Draw a small red dot for each keypoint
                            for i, (x, y) in enumerate(keypoints):
                                if confs[i] > 0.2:
                                    cv2.circle(frame_bgr, (int(x), int(y)), 4, (0, 0, 255), -1)
                                    
                            # Compute angles using helpers
                            angles = compute_joint_angles(keypoints, side=side)
                            
                            # Text placement in the top-left corner
                            x0, y0 = 20, 30   # starting text position
                            dy = 25           # vertical spacing per line
                            
                            for j, (name, val) in enumerate(angles.items()):
                                txt = f"{name}: {val:.1f}" if np.isfinite(val) else f"{name}: nan"
                                # Put text on screen with white color and black outline for better visibility
                                cv2.putText(
                                    frame_bgr,
                                    txt,
                                    (x0, y0 + j * dy),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.7,                 
                                    (255, 255, 255),   
                                    2,               
                                    cv2.LINE_AA,
                                )
                
                    # Saving path for video
                    out_path = "../data/output_videos"
                    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
                    h, w = frame_bgr.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

                    # Create writer that saves frames to out_path
                    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
                    
                    if writer is not None:
                        writer.write(frame_bgr)
                        
                    yield frame_bgr
            finally:
                # Close video and writer resources
                cap.release()
                if writer is not None:
                    writer.release()
                    
    def process_video(video_path: Path | str) -> tuple[pd.DataFrame, list[np.ndarray]]:
        """Returns data frame and one annotated frame; combined all previous steps"""
        # TODO