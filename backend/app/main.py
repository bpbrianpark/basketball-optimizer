"""Application entrypoint for the basketball shooting form analyzer."""
from fastapi import FastAPI

from backend.app.api import analyze_router
from backend.app.api.inference_routes import router as inference_router


def create_app() -> FastAPI:
    """FastAPI application factory."""
    app = FastAPI(title="Basketball Shooting Form Analyzer", version="0.1.0")

    app.include_router(analyze_router, prefix="/api/analyze", tags=["analyze"])
    app.include_router(inference_router, prefix="/api/inference", tags=["inference"])

    @app.get("/health", tags=["system"])
    async def health_check() -> dict[str, str]:
        """Simple health endpoint for readiness checks."""
        return {"status": "ok"}

    return app


app = create_app()
