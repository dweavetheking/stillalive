import logging
import shutil
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.config import settings
from app.routes import generate, health, upload
from app.routes.health import install_log_buffer
from app.services.generation_queue import generation_queue
from app.services.pipeline import pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
install_log_buffer()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting Still Alive backend...")
    settings.ensure_dirs()

    # Clean orphaned working dirs from crashed runs
    working = Path(settings.working_dir)
    if working.exists():
        shutil.rmtree(working, ignore_errors=True)
        working.mkdir(parents=True, exist_ok=True)

    # Load the ML pipeline
    pipeline.load()
    generation_queue.start()
    logger.info("Ready to generate!")

    yield

    # Shutdown
    logger.info("Shutting down...")
    generation_queue.stop()
    pipeline.unload()


app = FastAPI(
    title="Still Alive",
    description="EchoMimic V3 Flash Studio Backend",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS — allow frontend origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount routes
app.include_router(health.router, tags=["health"])
app.include_router(upload.router, tags=["upload"])
app.include_router(generate.router, tags=["generate"])

# Serve output files (videos, thumbnails) as static files
output_dir = Path(settings.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)
app.mount("/outputs", StaticFiles(directory=str(output_dir)), name="outputs")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=settings.host, port=settings.port)
