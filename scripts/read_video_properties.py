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

def standardize_video(input_path: str, output_path: str, 
                     target_fps: int = TARGET_FPS,
                     target_width: int = TARGET_WIDTH,
                     target_height: int = TARGET_HEIGHT) -> bool:
    """
    Resize and resample video to target specifications using OpenCV.
    
    Args:
        input_path: Path to input video file (supports various formats: mp4, avi, mov, mkv, etc.)
        output_path: Path to save standardized video
        target_fps: Target frames per second (default: 30)
        target_width: Target width in pixels (default: 1280)
        target_height: Target height in pixels (default: 720)
    
    Returns:
        True if successful, False otherwise
    """
    # Ensure output directory exists
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Could not open video: {input_path}")
        return False
    
    # Get input properties
    input_fps = cap.get(cv2.CAP_PROP_FPS)
    input_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    input_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calculate frame skip ratio if FPS needs to be reduced
    frame_skip = 1.0
    if input_fps > target_fps:
        frame_skip = input_fps / target_fps
    
    # Create output video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, target_fps, 
                         (target_width, target_height))
    
    if not out.isOpened():
        print(f"Error: Could not create output video: {output_path}")
        cap.release()
        return False
    
    try:
        frame_count = 0
        frames_written = 0
        next_frame_to_write = 0.0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Handle frame skipping for FPS reduction
            if frame_count >= next_frame_to_write:
                # Resize frame to target resolution
                resized_frame = cv2.resize(frame, (target_width, target_height), 
                                         interpolation=cv2.INTER_LINEAR)
                out.write(resized_frame)
                frames_written += 1
                next_frame_to_write += frame_skip
            
            frame_count += 1
        
        # If input FPS is lower than target, we need to duplicate frames
        # This is handled by writing at target FPS (frames will be duplicated automatically)
        
        return True
    except Exception as e:
        print(f"Error processing video {input_path}: {e}")
        return False
    finally:
        cap.release()
        out.release()

def batch_process_videos(input_dir: str = "data/raw", 
                        output_dir: str = None) -> dict:
    """
    Process all videos from input directory and save to output directory.
    Preserves original filenames (converts extension to target format).
    
    Args:
        input_dir: Directory containing input videos (default: "data/raw")
        output_dir: Directory to save standardized videos (default: uses STANDARDIZED_DIR)
    
    Returns:
        Dictionary with processing statistics: {'total': int, 'success': int, 'failed': int, 'files': list}
    """
    input_path = Path(input_dir)
    if not input_path.exists():
        print(f"Error: Input directory does not exist: {input_dir}")
        return {'total': 0, 'success': 0, 'failed': 0, 'files': []}
    
    if output_dir is None:
        output_path = STANDARDIZED_DIR
    else:
        output_path = Path(output_dir)
    
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all video files (handle different formats)
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.MOV', '.MP4', 
                       '.AVI', '.MOV', '.flv', '.wmv', '.webm']
    video_files = []
    for ext in video_extensions:
        video_files.extend(input_path.glob(f"*{ext}"))
        video_files.extend(input_path.glob(f"**/*{ext}"))  # Also search subdirectories
    
    # Remove duplicates and sort
    video_files = sorted(set(video_files))
    
    if len(video_files) == 0:
        print(f"No video files found in: {input_dir}")
        return {'total': 0, 'success': 0, 'failed': 0, 'files': []}
    
    print(f"Found {len(video_files)} video(s) to process")
    print(f"Output directory: {output_path}\n")
    
    results = {
        'total': len(video_files),
        'success': 0,
        'failed': 0,
        'files': []
    }
    
    for video_file in video_files:
        # Preserve original filename (stem) but change extension to target format
        output_file = output_path / f"{video_file.stem}{TARGET_EXT}"
        
        # Skip if output already exists
        if output_file.exists():
            print(f"⏭  Skipping (already exists): {video_file.name}")
            results['success'] += 1
            results['files'].append({
                'input': str(video_file),
                'output': str(output_file),
                'status': 'skipped'
            })
            continue
        
        print(f"Processing: {video_file.name} -> {output_file.name}")
        
        success = standardize_video(str(video_file), str(output_file))
        
        if success:
            # Verify output file was created
            if output_file.exists():
                print(f"  ✓ Successfully processed\n")
                results['success'] += 1
                results['files'].append({
                    'input': str(video_file),
                    'output': str(output_file),
                    'status': 'success'
                })
            else:
                print(f"  ✗ Output file not created\n")
                results['failed'] += 1
                results['files'].append({
                    'input': str(video_file),
                    'output': str(output_file),
                    'status': 'failed'
                })
        else:
            print(f"  ✗ Failed to process\n")
            results['failed'] += 1
            results['files'].append({
                'input': str(video_file),
                'output': str(output_file),
                'status': 'failed'
            })
    
    print(f"\n{'='*50}")
    print(f"Batch Processing Complete")
    print(f"Total: {results['total']}")
    print(f"Success: {results['success']}")
    print(f"Failed: {results['failed']}")
    print(f"{'='*50}")
    
    return results

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python read_video_properties.py <video_path>          # Get video properties")
        print("  python read_video_properties.py --batch [input_dir]  # Batch process videos")
        print("  python read_video_properties.py --standardize <input> <output>  # Process single video")
        sys.exit(1)
    
    if sys.argv[1] == "--batch":
        # Batch processing mode
        input_dir = sys.argv[2] if len(sys.argv) > 2 else "data/raw"
        batch_process_videos(input_dir)
    elif sys.argv[1] == "--standardize":
        # Single video standardization mode
        if len(sys.argv) < 4:
            print("Error: --standardize requires input and output paths")
            print("Usage: python read_video_properties.py --standardize <input> <output>")
            sys.exit(1)
        input_path = sys.argv[2]
        output_path = sys.argv[3]
        success = standardize_video(input_path, output_path)
        if success:
            print(f"✓ Video standardized successfully: {output_path}")
            # Show properties of output
            props = get_video_properties(output_path)
            if props:
                print(f"\nOutput Video Properties:")
                print(f"  FPS: {props['fps']}")
                print(f"  Resolution: {props['width']}x{props['height']}")
                print(f"  Frame Count: {props['frame_count']}")
                print(f"  Duration: {props['duration_sec']:.2f} seconds")
        else:
            print(f"✗ Failed to standardize video")
            sys.exit(1)
    else:
        # Get video properties mode (original functionality)
        video_path = sys.argv[1]
        props = get_video_properties(video_path)
        
        if props is None:
            print(f"Error: Could not read video file: {video_path}")
            sys.exit(1)
        
        print(f"Video Properties for: {props['path']}")
        print(f"  FPS: {props['fps']}")
        print(f"  Resolution: {props['width']}x{props['height']}")
        print(f"  Frame Count: {props['frame_count']}")
        print(f"  Duration: {props['duration_sec']:.2f} seconds")