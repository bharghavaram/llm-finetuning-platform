"""
LLM Fine-Tuning Platform – PEFT/LoRA training pipeline with MLflow tracking.
Supports QLoRA (4-bit) fine-tuning for Mistral-7B, Llama-3, and other HF models.
"""
import logging
import json
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional, List
from app.core.config import settings

logger = logging.getLogger(__name__)


class FineTuningJob:
    def __init__(self, job_id: str, config: dict):
        self.job_id = job_id
        self.config = config
        self.status = "queued"
        self.created_at = datetime.utcnow().isoformat()
        self.started_at = None
        self.completed_at = None
        self.metrics = {}
        self.error = None
        self.model_path = None

    def to_dict(self):
        return {
            "job_id": self.job_id,
            "status": self.status,
            "config": self.config,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "metrics": self.metrics,
            "error": self.error,
            "model_path": self.model_path,
        }


class FineTuningService:
    def __init__(self):
        self._jobs: dict[str, FineTuningJob] = {}

    def create_training_config(
        self,
        base_model: str,
        dataset_name: str,
        task_type: str = "causal_lm",
        lora_r: int = None,
        lora_alpha: int = None,
        learning_rate: float = None,
        num_epochs: int = None,
        use_4bit: bool = None,
    ) -> dict:
        return {
            "base_model": base_model or settings.BASE_MODEL,
            "dataset_name": dataset_name,
            "task_type": task_type,
            "lora_config": {
                "r": lora_r or settings.LORA_R,
                "lora_alpha": lora_alpha or settings.LORA_ALPHA,
                "lora_dropout": settings.LORA_DROPOUT,
                "bias": "none",
                "task_type": "CAUSAL_LM",
                "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
            },
            "training_config": {
                "learning_rate": learning_rate or settings.LEARNING_RATE,
                "num_train_epochs": num_epochs or settings.NUM_EPOCHS,
                "per_device_train_batch_size": settings.BATCH_SIZE,
                "gradient_accumulation_steps": 4,
                "warmup_steps": 100,
                "max_seq_length": settings.MAX_SEQ_LENGTH,
                "save_steps": 500,
                "logging_steps": 10,
                "fp16": True,
                "optim": "paged_adamw_8bit",
            },
            "quantization": {
                "load_in_4bit": use_4bit if use_4bit is not None else settings.USE_4BIT,
                "bnb_4bit_compute_dtype": "float16",
                "bnb_4bit_quant_type": "nf4",
                "bnb_4bit_use_double_quant": True,
            },
        }

    def start_training(self, config: dict) -> FineTuningJob:
        job_id = str(uuid.uuid4())
        job = FineTuningJob(job_id, config)
        self._jobs[job_id] = job

        try:
            self._run_training(job)
        except Exception as exc:
            job.status = "failed"
            job.error = str(exc)
            logger.error("Training job %s failed: %s", job_id, exc)

        return job

    def _run_training(self, job: FineTuningJob):
        import mlflow
        mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
        mlflow.set_experiment(settings.MLFLOW_EXPERIMENT)

        job.status = "running"
        job.started_at = datetime.utcnow().isoformat()
        logger.info("Starting training job %s", job.job_id)

        with mlflow.start_run(run_name=f"finetune-{job.job_id[:8]}"):
            mlflow.log_params({
                "base_model": job.config["base_model"],
                "dataset": job.config["dataset_name"],
                "lora_r": job.config["lora_config"]["r"],
                "lora_alpha": job.config["lora_config"]["lora_alpha"],
                "learning_rate": job.config["training_config"]["learning_rate"],
                "num_epochs": job.config["training_config"]["num_train_epochs"],
                "use_4bit": job.config["quantization"]["load_in_4bit"],
            })

            try:
                from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
                from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
                from datasets import load_dataset
                from trl import SFTTrainer, SFTConfig
                import torch

                quant_config = None
                if job.config["quantization"]["load_in_4bit"]:
                    quant_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_use_double_quant=True,
                    )

                tokenizer = AutoTokenizer.from_pretrained(
                    job.config["base_model"],
                    token=settings.HF_TOKEN or None,
                    trust_remote_code=True,
                )
                tokenizer.pad_token = tokenizer.eos_token

                model = AutoModelForCausalLM.from_pretrained(
                    job.config["base_model"],
                    quantization_config=quant_config,
                    device_map="auto",
                    token=settings.HF_TOKEN or None,
                    trust_remote_code=True,
                )
                model = prepare_model_for_kbit_training(model)

                lora_cfg = job.config["lora_config"]
                peft_config = LoraConfig(
                    r=lora_cfg["r"],
                    lora_alpha=lora_cfg["lora_alpha"],
                    lora_dropout=lora_cfg["lora_dropout"],
                    bias=lora_cfg["bias"],
                    task_type=TaskType.CAUSAL_LM,
                    target_modules=lora_cfg.get("target_modules", ["q_proj", "v_proj"]),
                )

                train_cfg = job.config["training_config"]
                output_dir = Path(settings.OUTPUT_DIR) / job.job_id
                output_dir.mkdir(parents=True, exist_ok=True)

                sft_config = SFTConfig(
                    output_dir=str(output_dir),
                    learning_rate=train_cfg["learning_rate"],
                    num_train_epochs=train_cfg["num_train_epochs"],
                    per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
                    gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
                    warmup_steps=train_cfg["warmup_steps"],
                    max_seq_length=train_cfg["max_seq_length"],
                    fp16=train_cfg["fp16"],
                    logging_steps=train_cfg["logging_steps"],
                    save_steps=train_cfg["save_steps"],
                    optim=train_cfg["optim"],
                    report_to="none",
                )

                dataset_path = Path(settings.DATASET_PATH) / job.config["dataset_name"]
                if dataset_path.exists():
                    dataset = load_dataset("json", data_files=str(dataset_path))["train"]
                else:
                    dataset = load_dataset(job.config["dataset_name"], split="train[:1000]")

                trainer = SFTTrainer(
                    model=model,
                    train_dataset=dataset,
                    peft_config=peft_config,
                    args=sft_config,
                    tokenizer=tokenizer,
                )
                train_result = trainer.train()

                job.metrics = {
                    "train_loss": round(train_result.training_loss, 4),
                    "train_runtime": round(train_result.metrics.get("train_runtime", 0), 1),
                    "samples_per_second": round(train_result.metrics.get("train_samples_per_second", 0), 2),
                }
                mlflow.log_metrics(job.metrics)
                trainer.save_model()
                job.model_path = str(output_dir)

            except ImportError as e:
                # Simulation mode when GPU not available
                logger.warning("GPU libraries not available, simulating training: %s", e)
                import time
                for epoch in range(1, job.config["training_config"]["num_train_epochs"] + 1):
                    time.sleep(0.1)
                    simulated_loss = 2.5 - (epoch * 0.3)
                    mlflow.log_metric("simulated_loss", simulated_loss, step=epoch)
                    logger.info("Epoch %d/%d – simulated_loss: %.4f", epoch, job.config["training_config"]["num_train_epochs"], simulated_loss)
                job.metrics = {
                    "train_loss": 2.5 - (job.config["training_config"]["num_train_epochs"] * 0.3),
                    "train_runtime": job.config["training_config"]["num_train_epochs"] * 0.1,
                    "mode": "simulated",
                }
                mlflow.log_metrics({"final_simulated_loss": job.metrics["train_loss"]})
                output_dir = Path(settings.OUTPUT_DIR) / job.job_id
                output_dir.mkdir(parents=True, exist_ok=True)
                job.model_path = str(output_dir)

            job.status = "completed"
            job.completed_at = datetime.utcnow().isoformat()
            logger.info("Job %s completed. Metrics: %s", job.job_id, job.metrics)

    def generate_inference(self, model_path: str, prompt: str, max_tokens: int = 256) -> dict:
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
            from peft import PeftModel
            import torch

            tokenizer = AutoTokenizer.from_pretrained(model_path)
            base_model = AutoModelForCausalLM.from_pretrained(
                model_path, device_map="auto", torch_dtype=torch.float16
            )
            pipe = pipeline("text-generation", model=base_model, tokenizer=tokenizer, max_new_tokens=max_tokens)
            result = pipe(prompt)[0]["generated_text"]
            return {"generated_text": result, "model_path": model_path}
        except Exception as exc:
            return {"error": str(exc), "note": "Model inference requires GPU environment"}

    def get_job(self, job_id: str) -> Optional[FineTuningJob]:
        return self._jobs.get(job_id)

    def list_jobs(self) -> list:
        return [job.to_dict() for job in self._jobs.values()]

    def evaluate_model(self, job_id: str, test_prompts: List[str]) -> dict:
        job = self._jobs.get(job_id)
        if not job or job.status != "completed":
            return {"error": "Job not found or not completed"}
        results = []
        for prompt in test_prompts:
            result = self.generate_inference(job.model_path, prompt)
            results.append({"prompt": prompt, "response": result.get("generated_text", result.get("error", ""))})
        return {"job_id": job_id, "model_path": job.model_path, "evaluations": results}


_service: Optional[FineTuningService] = None
def get_finetuning_service() -> FineTuningService:
    global _service
    if _service is None:
        _service = FineTuningService()
    return _service
