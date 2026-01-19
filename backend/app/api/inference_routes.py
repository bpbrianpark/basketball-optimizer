from fastapi import APIRouter, UploadFile, File, HTTPException
from backend.app.services.inference_service import InferenceService
from pathlib import Path
import uuid

router = APIRouter()
service = InferenceService()
UPLOAD_DIR = Path("data/inference/uploads")

@router.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    video_id = str(uuid.uuid4())
    file_path = service.upload_dir / f"{video_id}.mp4"
    with open(file_path, "wb") as f:
        f.write(await file.read())
    service.store_video(video_id, str(file_path))
    return {"video_id": video_id}

@router.post("/analyze/{video_id}")
def analyze_video(video_id: str):
    video_path = service.get_video(video_id)
    if not video_path:
        raise HTTPException(status_code=404, detail="Video not found")
    # result = service.process_video(video_path)
    fake_result = {"score": 75}
    result_id = str(uuid.uuid4())
    service.save_result(result_id, fake_result)
    return {"result_id": result_id}

@router.get("/results/{result_id}")
def get_result(result_id: str):
    result = service.get_result(result_id)
    if not result:
        raise HTTPException(status_code=404, detail="Result not found")
    return result