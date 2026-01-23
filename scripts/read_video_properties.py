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

    processed = 0
    
    # Processes each video file in data/raw
    for video in video_files:
        # Open each video with VideoCapture
        cap = cv2.VideoCapture(str(video))
        # Skip video if not able to open
        if not cap.isOpened():
            print(f"Warning: Could not open video file: {video}")
            continue
        
        # Original video metadata
        input_fps = cap.get(cv2.CAP_PROP_FPS)
        input_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        input_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Decide how many frames to skip depending on what we are targetting
        frame_skip = max(1, round(input_fps / TARGET_FPS))
        
        # Output path
        output_path = OUTPUT_DIR / (video.stem + TARGET_EXT)
        
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        
        # Write the video with standardized properties
        writer = cv2.VideoWriter(
            str(output_path),
            fourcc,
            TARGET_FPS,
            (TARGET_WIDTH, TARGET_HEIGHT)
        )
        
        # For FPS downsampling
        frame_id = 0
        
        while True:
            ret, frame = cap.read()
            
            # Stop when no frames are available
            if not ret:
                break
            
            # If the current frame is to be kept
            if frame_id % frame_skip == 0:
                # Resize and write to output video
                resized_frame = cv2.resize(frame, (TARGET_WIDTH, TARGET_HEIGHT))
                writer.write(resized_frame)
            frame += 1
        
        # Release to avoid memory leaks or issues
        cap.release()
        writer.release()
        
        # Tracks processed videos
        processed += 1
        
    print(f"Processed {processed}/{len(video_files)} videos to standardized format")
    
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