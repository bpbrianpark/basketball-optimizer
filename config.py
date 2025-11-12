"""Configuration settings for the basketball shooting form analyzer."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_VIDEOS_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
LABELS_DIR = DATA_DIR / "labels"
STANDARDIZED_VIDEOS_DIR = PROCESSED_DIR / "standardized"

MODELS_DIR = PROJECT_ROOT / "models"
CLASSIFIER_MODEL_PATH = MODELS_DIR / "shot_classifier.joblib"
MODEL_METADATA_PATH = MODELS_DIR / "model_metadata.json"

YOLO_MODEL_PATH = "yolov8n-pose.pt"

TARGET_FPS = 30
TARGET_RESOLUTION = (1280, 720)  
TARGET_FORMAT = "mp4"

FRAME_SAMPLE_RATE = 3  # Process every Nth frame (3 = every 3rd frame)
MIN_KEYPOINT_CONFIDENCE = 0.5  # Minimum confidence for keypoint detection

API_HOST = "0.0.0.0"
API_PORT = 8000

# Scoring thresholds (for heuristic scoring - will be replaced by ML model)
GOOD_SHOT_THRESHOLD = 0.5  # Probability threshold for "good" shot classification

