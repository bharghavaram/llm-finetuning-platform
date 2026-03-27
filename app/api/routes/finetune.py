"""LLM Fine-Tuning Platform – API routes."""
from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
from pydantic import BaseModel
from typing import Optional, List
from app.services.finetuning_service import FineTuningService, get_finetuning_service

router = APIRouter(prefix="/finetune", tags=["Fine-Tuning"])

class TrainingRequest(BaseModel):
    base_model: str = "mistralai/Mistral-7B-v0.1"
    dataset_name: str
    task_type: str = "causal_lm"
    lora_r: Optional[int] = None
    lora_alpha: Optional[int] = None
    learning_rate: Optional[float] = None
    num_epochs: Optional[int] = None
    use_4bit: Optional[bool] = None

class EvalRequest(BaseModel):
    job_id: str
    test_prompts: List[str]

class InferenceRequest(BaseModel):
    model_path: str
    prompt: str
    max_tokens: int = 256

@router.post("/start")
async def start_training(req: TrainingRequest, bg: BackgroundTasks, svc: FineTuningService = Depends(get_finetuning_service)):
    config = svc.create_training_config(
        req.base_model, req.dataset_name, req.task_type,
        req.lora_r, req.lora_alpha, req.learning_rate, req.num_epochs, req.use_4bit,
    )
    job = svc.start_training(config)
    return job.to_dict()

@router.get("/jobs")
async def list_jobs(svc: FineTuningService = Depends(get_finetuning_service)):
    return {"jobs": svc.list_jobs()}

@router.get("/jobs/{job_id}")
async def get_job(job_id: str, svc: FineTuningService = Depends(get_finetuning_service)):
    job = svc.get_job(job_id)
    if not job:
        raise HTTPException(404, f"Job {job_id} not found")
    return job.to_dict()

@router.post("/evaluate")
async def evaluate(req: EvalRequest, svc: FineTuningService = Depends(get_finetuning_service)):
    return svc.evaluate_model(req.job_id, req.test_prompts)

@router.post("/inference")
async def inference(req: InferenceRequest, svc: FineTuningService = Depends(get_finetuning_service)):
    return svc.generate_inference(req.model_path, req.prompt, req.max_tokens)

@router.get("/health")
async def health():
    return {"status": "ok", "service": "LLM Fine-Tuning Platform – PEFT/LoRA Pipeline"}
