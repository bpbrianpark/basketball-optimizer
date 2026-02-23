"""Application entrypoint for the basketball shooting form analyzer."""
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from backend.app.api import analyze_router
from backend.app.api.inference_routes import router as inference_router


def create_app() -> FastAPI:
    """FastAPI application factory."""
    app = FastAPI(title="Basketball Shooting Form Analyzer", version="0.1.0")

    app.include_router(analyze_router, prefix="/api/analyze", tags=["analyze"])
    app.include_router(inference_router, prefix="/api/inference", tags=["inference"])

    # a mobile app cannot access files in this repo - so this serves static files via HTTP
    app.mount("/overlays", StaticFiles(directory="data/inference/overlays"), name="overlays")

    @app.get("/health", tags=["system"])
    async def health_check() -> dict[str, str]:
        """Simple health endpoint for readiness checks."""
        return {"status": "ok"}

    return app


app = create_app()
