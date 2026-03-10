import json
import cv2
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any
import sys

project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from models.pose_pipeline import PosePipeline

COCO_SKELETON = [
    (5, 7), (7, 9),
    (6, 8), (8, 10),
    (5, 6),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
]

class InferenceService:
    def __init__(self, data_dir: str = "data/inference"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.videos_path = self.data_dir / "videos.json"
        self.results_path = self.data_dir / "results.json"

        self.upload_dir = self.data_dir / "uploads"
        self.upload_dir.mkdir(exist_ok=True)
        
        self.overlays_dir = self.data_dir / "overlays"
        self.overlays_dir.mkdir(exist_ok=True)

        self.video_store: Dict[str, str] = self._load_store(self.videos_path)
        self.results_store: Dict[str, Any] = self._load_store(self.results_path)

    def _load_store(self, path: Path) -> Dict:
        if path.exists():
            with open(path, "r") as f:
                return json.load(f)
        return {}

    def _save_store(self, path: Path, store: Dict):
        with open(path, "w") as f:
            json.dump(store, f, indent=2)

    def store_video(self, video_id: str, path: str):
        self.video_store[video_id] = path
        self._save_store(self.videos_path, self.video_store)

    def save_result(self, result_id: str, payload: Dict):
        self.results_store[result_id] = payload
        self._save_store(self.results_path, self.results_store)

    def get_video(self, video_id: str) -> str | None:
        return self.video_store.get(video_id)

    def get_result(self, result_id: str) -> Dict | None:
        return self.results_store.get(result_id)

    def _draw_pose_on_frame(self, frame: np.ndarray, frame_data: pd.DataFrame) -> np.ndarray:
        annotated_frame = frame.copy()
        
        if frame_data.empty:
            return annotated_frame
        
        for _, row in frame_data.iterrows():
            if pd.isna(row.x) or pd.isna(row.y) or row.confidence < 0.5:
                continue
            x, y = int(row.x), int(row.y)
            cv2.circle(annotated_frame, (x, y), 4, (0, 0, 255), -1)
        
        for a, b in COCO_SKELETON:
            pa = frame_data[frame_data["keypoint"] == a]
            pb = frame_data[frame_data["keypoint"] == b]
            
            if pa.empty or pb.empty:
                continue
            
            xa_val, ya_val = pa.x.values[0], pa.y.values[0]
            xb_val, yb_val = pb.x.values[0], pb.y.values[0]
            
            if pd.isna(xa_val) or pd.isna(ya_val) or pd.isna(xb_val) or pd.isna(yb_val):
                continue
            
            x1, y1 = int(xa_val), int(ya_val)
            x2, y2 = int(xb_val), int(yb_val)
            cv2.line(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        return annotated_frame

    def process_video_and_save_overlays(self, video_path: str, result_id: str, video_id: str):
        model_path = project_root / "notebooks" / "yolov8n-pose.pt"
        if not model_path.exists():
            model_path = "yolov8n-pose.pt"
        
        pose_pipeline = PosePipeline(path=str(model_path))
        pose_df = pose_pipeline.process_video(video_path)
        
        result_overlay_dir = self.overlays_dir / result_id
        result_overlay_dir.mkdir(parents=True, exist_ok=True)
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video at: {video_path}")
        
        frame_paths = []
        frame_idx = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_data = pose_df[pose_df["frame"] == frame_idx]
                annotated_frame = self._draw_pose_on_frame(frame, frame_data)
                
                frame_file = result_overlay_dir / f"frame_{frame_idx:04d}.jpg"
                cv2.imwrite(str(frame_file), annotated_frame)
                frame_paths.append(str(frame_file.relative_to(self.data_dir)))
                
                frame_idx += 1
        finally:
            cap.release()
        
        result_data = {
            "video_id": video_id,
            "score": 75,
            "overlay_frames": frame_paths,
            "total_frames": frame_idx
        }
        
        self.results_store[result_id] = result_data
        self._save_store(self.results_path, self.results_store)
    
    def get_overlay(self, result_id:str, frame: int=0) -> np.ndarray | None:
        result = self.get_result(result_id)
        if not result:
            return None
        overlay_frames = result.get("overlay_frames", [])
        if frame < 0 or frame >= len(overlay_frames):
            return None
        
        frame_path = self.data_dir / overlay_frames[frame]
        if not frame_path.exists():
            return None
        
        img = cv2.imread(str(frame_path))
        return img
        