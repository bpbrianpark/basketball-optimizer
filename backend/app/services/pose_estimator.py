"""Pose estimation service layer stubs."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import cv2
import numpy as np
from fastapi import UploadFile
from scripts.score_data import score_shot

try:
    from ultralytics import YOLO 
except ImportError:  
    YOLO = None 


# @dataclass
# class PoseAnalysisResult:
#     score: float
#     strengths: list[str]
#     weaknesses: list[str]
#     metadata: dict[str, float]


class PoseEstimatorService:
    """High-level interface for orchestrating pose estimation and scoring."""

    def __init__(self, model_path: str | None = None) -> None:
        self.model_path = model_path or "yolov8n-pose.pt"
        self._model: Any | None = None

    def _load_model(self) -> None:
        if self._model is None:
            if YOLO is None:
                raise RuntimeError(
                    "ultralytics is not installed. Install dependencies to enable pose estimation."
                )
            self._model = YOLO(self.model_path)

    def _iter_frames(self, video_path: str, frame_skip: int = 3) -> Iterator[np.ndarray]:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video at: {video_path}.")
        
        try:
            frame_index = 0
            while True:
                ret, frame = cap.read()
                if (not ret):
                    break

                # Yield every 3 frames by default
                if frame_index % frame_skip == 0:
                    yield self._extract_keypoints(frame)
                frame_index +=1
        finally:
            cap.release()
    
    def _extract_keypoints(self, frame: np.ndarray) -> np.ndarray | None:
        """
            Extracts the keypoint coordinates + confidence scores from a single frame.
        """

        results = self._model(frame)
        if (not results or len(results) == 0):
            return None
        
        result = results[0]

        # If no keypoints / person is detected
        if (not hasattr(result, 'keypoints') or result.keypoints == None):
            print("There are no person detected.")
            return None
        
        keypoints_data = result.keypoints.data
        if (keypoints_data == None or len(keypoints_data) == 0):
            print("The detected person has no keypoint data.")
            return None

        return keypoints_data


    async def process_upload(self, video: UploadFile) -> dict:
        """Persist the uploaded file and delegate to main analysis pipeline."""
        temp_path = Path("/tmp") / video.filename
        content = await video.read()
        temp_path.write_bytes(content)
        return self.analyze_video(temp_path)

    def analyze_video(self, video_path: Path) -> dict:
        """Run pose estimation and scoring on a video file.

        This is a placeholder implementation that should be replaced with the
        real pose extraction and scoring pipeline.
        """
        # self._load_model()

        # TODO: integrate OpenCV
        _ = np.array([0.0])  

        # Placeholder response
        
        sample_data = {"elbow_angle":30}
        return score_shot(sample_data)

        # Kuan: commented out after score_shot implementation
        # return PoseAnalysisResult(
        #     score=0.0,
        #     strengths=["Consistent release point"],
        #     weaknesses=["Incomplete follow-through"],
        #     metadata={"frames_analyzed": 0.0},
        # )
