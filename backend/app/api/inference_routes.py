from fastapi import APIRouter, UploadFile, File, HTTPException
from backend.app.services.inference_service import InferenceService
from pathlib import Path
import uuid
from typing import Dict
import numpy as np
import cv2
from fastapi.responses import Response

router = APIRouter()
service = InferenceService()

MAX_FILE_SIZE = 100 * 1024 * 1024
ALLOWED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}

@router.post("/upload")
async def upload_video(file: UploadFile = File(...)) -> Dict[str, str]:
    """Upload a video file for analysis."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is required")
    
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
        )
    
    content = await file.read()
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size: {MAX_FILE_SIZE // (1024*1024)}MB"
        )
    
    service.upload_dir.mkdir(parents=True, exist_ok=True)
    
    video_id = str(uuid.uuid4())
    file_path = service.upload_dir / f"{video_id}{file_ext}"
    
    try:
        with open(file_path, "wb") as f:
            f.write(content)
        service.store_video(video_id, str(file_path))
        return {"video_id": video_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")

@router.post("/analyze/{video_id}")
async def analyze_video(video_id: str) -> Dict[str, str]:
    """Analyze a video and return result ID."""
    video_path = service.get_video(video_id)
    if not video_path:
        raise HTTPException(status_code=404, detail="Video not found")
    
    result_id = str(uuid.uuid4())
    try:
        service.process_video_and_save_overlays(video_path, result_id, video_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process video: {str(e)}")
    
    return {"result_id": result_id}

@router.get("/results/{result_id}")
async def get_result(result_id: str) -> Dict:
    """Returns the inference result for a given result ID"""
    result = service.get_result(result_id)
    if not result:
        raise HTTPException(status_code=404, detail="Result not found")
    return result

@router.get("/results/{result_id}/overlay")
async def get_overlay(result_id: str, frame: int = 0):
    """Return the overlay image for a given result ID and frame index."""
    overlay = service.get_overlay(result_id, frame)
    if overlay is None:
        raise HTTPException(status_code=404, detail="Overlay not found")
    success, buffer = cv2.imencode(".jpg", overlay)
    if not success:
        raise HTTPException(status_code = 500, detail= "Failed to encode overlay image")
    
    return Response(content=buffer.tobytes(), media_type="image/jpeg")
