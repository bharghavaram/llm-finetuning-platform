import os
from dotenv import load_dotenv
load_dotenv()

class Settings:
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    HF_TOKEN: str = os.getenv("HF_TOKEN", "")
    BASE_MODEL: str = os.getenv("BASE_MODEL", "mistralai/Mistral-7B-v0.1")
    OUTPUT_DIR: str = os.getenv("OUTPUT_DIR", "models/finetuned")
    DATASET_PATH: str = os.getenv("DATASET_PATH", "datasets/")
    MLFLOW_TRACKING_URI: str = os.getenv("MLFLOW_TRACKING_URI", "mlruns")
    MLFLOW_EXPERIMENT: str = os.getenv("MLFLOW_EXPERIMENT", "llm-finetuning")
    LORA_R: int = int(os.getenv("LORA_R", "16"))
    LORA_ALPHA: int = int(os.getenv("LORA_ALPHA", "32"))
    LORA_DROPOUT: float = float(os.getenv("LORA_DROPOUT", "0.05"))
    LEARNING_RATE: float = float(os.getenv("LEARNING_RATE", "2e-4"))
    NUM_EPOCHS: int = int(os.getenv("NUM_EPOCHS", "3"))
    BATCH_SIZE: int = int(os.getenv("BATCH_SIZE", "4"))
    MAX_SEQ_LENGTH: int = int(os.getenv("MAX_SEQ_LENGTH", "2048"))
    USE_4BIT: bool = os.getenv("USE_4BIT", "true").lower() == "true"
    APP_HOST: str = os.getenv("APP_HOST", "0.0.0.0")
    APP_PORT: int = int(os.getenv("APP_PORT", "8000"))

settings = Settings()
