# backend/app/api/videos.py
from fastapi import APIRouter, UploadFile, HTTPException
from pathlib import Path
import uuid
import shutil

router = APIRouter()

DATA_RAW = Path(__file__).resolve().parent.parent.parent.parent / "data" / "raw"
DATA_RAW.mkdir(parents=True, exist_ok=True)

@router.post("/videos", tags=["videos"])
async def upload_video(file: UploadFile):
    if not file.filename.endswith(".mp4"):
        raise HTTPException(status_code=400, detail="Only MP4 files are allowed")
    
    video_id = str(uuid.uuid4())
    file_path = DATA_RAW / f"{video_id}.mp4"
    
    with file_path.open("wb") as f:
        shutil.copyfileobj(file.file, f)
    
    return {"video_id": video_id}