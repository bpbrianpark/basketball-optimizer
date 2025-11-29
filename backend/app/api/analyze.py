"""API router for pose analysis requests."""
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, UploadFile
from pydantic import BaseModel

from backend.app.services.pose_estimator import PoseEstimatorService

router = APIRouter()


class AnalyzeResponse(BaseModel):
    score: float
    strengths: list[str]
    weaknesses: list[str]
    metadata: dict[str, float]


async def get_pose_service() -> PoseEstimatorService:
    """Dependency that constructs the pose estimator service."""
    return PoseEstimatorService()


@router.post("/", response_model=AnalyzeResponse)
async def analyze_shot(
    video: UploadFile,
    service: Annotated[PoseEstimatorService, Depends(get_pose_service)],
) -> AnalyzeResponse:
    """Analyze an uploaded shooting video and return qualitative feedback."""
    if not video.filename:
        raise HTTPException(status_code=400, detail="Video file is required")

    result = await service.process_upload(video)
    return AnalyzeResponse(
        score=result['score'],
        strengths=result['strengths'],
        weaknesses=result['weaknesses'],
        metadata=result['metadata'],
    )
