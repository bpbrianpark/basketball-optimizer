import os
import sys
import cv2
from pathlib import Path

TARGET_FPS = 30
TARGET_WIDTH = 1280
TARGET_HEIGHT = 720
TARGET_EXT = ".mp4"

def get_video_properties(video_path: str) -> dict | None:
    """
    Returns a dict with FPS, resolution, frame count, and duration (seconds)
    for the given video_path. Returns None if file is missing or cannot be read.
    """
    if not os.path.exists(video_path):
        return None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cap.release()
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    duration = frame_count / fps if fps and fps > 0 else 0.0

    cap.release()

    return {
        "path": video_path,
        "fps": float(fps),
        "frame_count": int(frame_count),
        "width": int(width),
        "height": int(height),
        "duration_sec": float(duration),
    }

STANDARDIZED_DIR = Path("data/processed/standardized")

def ensure_standardized_dir() -> None:
    STANDARDIZED_DIR.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":    
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
    else:
        video_path = ""
    
    props = get_video_properties(video_path)
    
    if props is None:
        print(f"Error: Could not read video file: {video_path}")
        sys.exit(1)
    
    print(f"Video Properties for: {props['path']}")
    print(f"  FPS: {props['fps']}")
    print(f"  Resolution: {props['width']}x{props['height']}")
    print(f"  Frame Count: {props['frame_count']}")
    print(f"  Duration: {props['duration_sec']:.2f} seconds")