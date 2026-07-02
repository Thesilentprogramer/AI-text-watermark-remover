"""
SynthID Watermark Remover — FastAPI Main Application
"""

import os
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from app.schemas import WatermarkRequest, WatermarkResponse, HealthResponse
from app.model_loader import init_engines, get_paraphrase_engine, get_detector_engine
from app.pipeline import AttackPipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("main")

pipeline_instance: AttackPipeline = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline_instance
    logger.info("Application starting up: initializing engines...")
    init_engines()
    pipeline_instance = AttackPipeline()
    logger.info("Startup complete: ready to process requests.")
    yield
    logger.info("Application shutting down.")


app = FastAPI(
    title="SynthID Watermark Remover",
    description="Adversarial ML pipeline for reversing SynthID text watermarks using Gemma 4 E2B-it",
    version="1.0.0",
    lifespan=lifespan
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/remove-watermark", response_model=WatermarkResponse)
async def remove_watermark(request: WatermarkRequest):
    global pipeline_instance
    if not request.text or len(request.text.strip()) == 0:
        raise HTTPException(status_code=400, detail="Input text cannot be empty.")

    try:
        if pipeline_instance is None:
            pipeline_instance = AttackPipeline()
        return pipeline_instance.run(request)
    except Exception as e:
        logger.error(f"Error during watermark removal pipeline: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Pipeline execution error: {str(e)}")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    paraphraser = get_paraphrase_engine()
    return HealthResponse(
        status="ok",
        model_loaded=paraphraser.loaded if paraphraser else False,
        device=paraphraser.device if paraphraser else "unknown",
        engine="ExtendedDetector (synthid-text) + Gemma 4 E2B"
    )


# Static Files Setup for Frontend
frontend_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend")
if os.path.exists(frontend_dir):
    app.mount("/static", StaticFiles(directory=frontend_dir), name="static")

    @app.get("/")
    async def serve_index():
        index_path = os.path.join(frontend_dir, "index.html")
        if os.path.exists(index_path):
            return FileResponse(index_path)
        return {"message": "Index.html not found"}

    @app.get("/app")
    async def serve_app():
        app_path = os.path.join(frontend_dir, "app.html")
        if os.path.exists(app_path):
            return FileResponse(app_path)
        return {"message": "App.html not found"}
