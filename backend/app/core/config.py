import sys
from pathlib import Path
from typing import List
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))


class Settings(BaseSettings):
    PROJECT_NAME: str = "Generals AI Server"
    VERSION: str = "1.0.0"
    API_PREFIX: str = "/game"
    DEBUG: bool = False

    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    WORKERS: int = 1

    # CORS
    BACKEND_CORS_ORIGINS: List[str] = ["*"]

    # Paths
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent
    DATA_DIR: Path = PROJECT_ROOT / "data"
    CHECKPOINT_PATH: Path = DATA_DIR / "checkpoints" / "model_latest.pth"

    # Model Config (Must match training config)
    MODEL_ACTION_DIM: int = 10004
    MODEL_CHANNELS: int = 196
    MODEL_RES_BLOCKS: int = 7

    # Inference Config
    BATCH_SIZE: int = 32
    TIMEOUT: float = 0.01
    INFERENCE_MODE: str = "torch"  # "torch" or "onnx"
    ONNX_PATH: Path = DATA_DIR / "checkpoints" / "model_latest.onnx"
    ONNX_PROVIDER: str = "auto"  # "auto", "cuda", "dml", "cpu"

    model_config = SettingsConfigDict(
        case_sensitive=True, env_file=".env", extra="ignore"
    )


settings = Settings()
