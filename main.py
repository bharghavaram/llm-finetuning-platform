"""LLM Fine-Tuning Platform – FastAPI Application Entry Point."""
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes.finetune import router as finetune_router
from app.core.config import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s – %(message)s")

app = FastAPI(
    title="LLM Fine-Tuning Platform",
    description="Production-grade PEFT/LoRA fine-tuning pipeline with QLoRA (4-bit) support, MLflow experiment tracking, and automated evaluation for Mistral-7B, Llama-3, and other HuggingFace models.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(finetune_router, prefix="/api/v1")

@app.get("/")
async def root():
    return {
        "service": "LLM Fine-Tuning Platform",
        "version": "1.0.0",
        "description": "PEFT/LoRA Fine-Tuning Pipeline with MLflow Tracking",
        "docs": "/docs",
        "supported_techniques": ["LoRA", "QLoRA (4-bit)", "PEFT", "SFT (Supervised Fine-Tuning)"],
        "supported_models": ["mistralai/Mistral-7B-v0.1", "meta-llama/Meta-Llama-3-8B", "google/gemma-7b"],
        "features": [
            "QLoRA 4-bit quantization for memory efficiency",
            "MLflow experiment tracking & comparison",
            "Async training job management",
            "Automated model evaluation",
            "HuggingFace Hub integration",
        ],
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host=settings.APP_HOST, port=settings.APP_PORT, reload=True)
