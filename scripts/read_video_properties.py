import os
import sys
import cv2
from pathlib import Path

TARGET_FPS = 30
TARGET_WIDTH = 1280
TARGET_HEIGHT = 720
TARGET_EXT = ".mp4"

STANDARDIZED_DIR = Path("data/processed/standardized")
RAW_DIR = Path("data/raw")
OUTPUT_DIR = Path("data/processed")

EXTENSIONS = (".mp4", ".mov", ".avi", ".mkv", ".webm")

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

def ensure_standardized_dir():
    STANDARDIZED_DIR.mkdir(parents=True, exist_ok=True)
    
def ensure_raw_dir():
    RAW_DIR.mkdir(parents=True, exist_ok=True)

def ensure_output_dir():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
def video_conversion():
    """Resize/resamples frames and write to output file to standardize videos"""
    
    # Checks if directories exist or not
    if not RAW_DIR.exists():
        ensure_raw_dir()
    if not OUTPUT_DIR.exists():
        ensure_output_dir()
    
    # Gets all video files in the raw directory
    video_files = sorted(p for p in RAW_DIR.iterdir() if p.is_file() and p.suffix.lower() in EXTENSIONS)
    
    if not video_files:
        print("No video files found in the raw directory.")
        return

    # TODO - process sample videos, ensure names preserved and verify all videos are processed

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